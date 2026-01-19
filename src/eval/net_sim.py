from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.predictors.base import Obs, Predictor
from src.predictors.dr_predictor import DRPredictor
from src.utils.angles import circular_abs_error_deg, circular_diff_deg


@dataclass(frozen=True)
class State:
    tic: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float


@dataclass(frozen=True)
class DRConfig:
    update_interval_tics: int
    pos_eps: float
    yaw_eps_deg: float
    pitch_eps_deg: float


@dataclass(frozen=True)
class NetConfig:
    """
    Network model (offline simulator).

    - latency_ms: constant one-way latency
    - jitter_ms: uniform jitter in [-jitter_ms, +jitter_ms]
    - loss_rate: iid packet loss probability in [0, 1]
    - bandwidth_kbps: if >0, caps throughput and creates queueing delay
    - header_bytes/payload_bytes_*: message size model (bytes)
    - rng_seed: ensures deterministic results under Tier A
    """

    tics_per_second: float
    latency_ms: float
    jitter_ms: float
    loss_rate: float
    bandwidth_kbps: float
    header_bytes: int
    payload_bytes_state: int
    rng_seed: int


@dataclass(frozen=True)
class SendEvent:
    send_tic: int
    kind: str  # "initial" | "periodic" | "correction"
    state: State


@dataclass(frozen=True)
class ArrivalEvent:
    arrival_time_s: float
    kind: str
    state: State
    size_bytes: int
    dropped: bool
    send_time_s: float


@dataclass(frozen=True)
class NetSummary:
    sent_packets: int
    delivered_packets: int
    dropped_packets: int
    sent_bytes: int
    delivered_bytes: int
    avg_one_way_delay_ms: float
    max_queue_delay_ms: float


@dataclass(frozen=True)
class ClientErrorSummary:
    pos_mae: float
    yaw_mae_deg: float
    pitch_mae_deg: float
    pos_max: float
    yaw_max_deg: float
    pitch_max_deg: float


@dataclass(frozen=True)
class DRNetSummary:
    n_tics: int
    dr_send_packets: int
    dr_send_bytes: int
    baseline_send_packets: int
    baseline_send_bytes: int
    savings_ratio_packets: float
    savings_ratio_bytes: float
    net: NetSummary
    client_error: ClientErrorSummary


def _euclid3(dx: float, dy: float, dz: float) -> float:
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _tic_to_time_s(tic: int, tps: float) -> float:
    return float(tic) / float(tps)


def baseline_traffic(states: Sequence[State], net_cfg: NetConfig) -> tuple[int, int]:
    # Baseline: send full state each tic (same size model)
    baseline_packets = len(states)
    msg_size = int(net_cfg.header_bytes + net_cfg.payload_bytes_state)
    baseline_bytes = baseline_packets * msg_size
    return baseline_packets, baseline_bytes


def _estimate_rates(prev: State | None, last: State) -> tuple[float, float, float, float, float]:
    if prev is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    dt = last.tic - prev.tic
    if dt <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    vx = (last.x - prev.x) / dt
    vy = (last.y - prev.y) / dt
    vz = (last.z - prev.z) / dt

    dyaw = circular_diff_deg(last.yaw, prev.yaw)
    dpitch = circular_diff_deg(last.pitch, prev.pitch)

    return vx, vy, vz, dyaw / dt, dpitch / dt


def _state_to_obs(s: State, action_mask: int) -> Obs:
    return Obs(
        tic=s.tic,
        x=s.x,
        y=s.y,
        z=s.z,
        yaw=s.yaw,
        pitch=s.pitch,
        action_mask=action_mask,
    )


def _predict(from_state: State, rates: tuple[float, float, float, float, float], tic: int) -> State:
    vx, vy, vz, yaw_rate, pitch_rate = rates
    dt = tic - from_state.tic
    if dt < 0:
        dt = 0

    return State(
        tic=tic,
        x=from_state.x + vx * dt,
        y=from_state.y + vy * dt,
        z=from_state.z + vz * dt,
        yaw=from_state.yaw + yaw_rate * dt,
        pitch=from_state.pitch + pitch_rate * dt,
    )


def plan_dr_sends(states: Sequence[State], cfg: DRConfig) -> list[SendEvent]:
    """
    Produce the server-side DR send schedule using:
    - periodic updates every K tics since last send
    - correction send if prediction error exceeds thresholds
    """
    if cfg.update_interval_tics <= 0:
        raise ValueError("update_interval_tics must be > 0")
    if len(states) < 2:
        raise ValueError("Need at least 2 states")

    for i in range(1, len(states)):
        if states[i].tic <= states[i - 1].tic:
            raise ValueError("States must be strictly increasing by tic")

    events: list[SendEvent] = []

    prev_auth: State | None = None
    last_auth: State = states[0]
    last_send_tic: int = states[0].tic

    events.append(SendEvent(send_tic=last_send_tic, kind="initial", state=last_auth))
    rates = _estimate_rates(prev_auth, last_auth)

    for i in range(1, len(states)):
        gt = states[i]
        pred = _predict(last_auth, rates, gt.tic)

        pos_err = _euclid3(pred.x - gt.x, pred.y - gt.y, pred.z - gt.z)
        yaw_err = circular_abs_error_deg(pred.yaw, gt.yaw)
        pitch_err = circular_abs_error_deg(pred.pitch, gt.pitch)

        need_periodic = (gt.tic - last_send_tic) >= cfg.update_interval_tics
        need_correction = (
            pos_err > cfg.pos_eps or yaw_err > cfg.yaw_eps_deg or pitch_err > cfg.pitch_eps_deg
        )

        if need_correction:
            events.append(SendEvent(send_tic=gt.tic, kind="correction", state=gt))
            last_send_tic = gt.tic
            prev_auth = last_auth
            last_auth = gt
            rates = _estimate_rates(prev_auth, last_auth)
            continue

        if need_periodic:
            events.append(SendEvent(send_tic=gt.tic, kind="periodic", state=gt))
            last_send_tic = gt.tic
            prev_auth = last_auth
            last_auth = gt
            rates = _estimate_rates(prev_auth, last_auth)

    return events


def plan_predictor_sends(
    states: Sequence[State],
    action_masks: Sequence[int],
    predictor: Predictor,
    cfg: DRConfig,
) -> list[SendEvent]:
    """
    Generic send schedule planner:
    - simulate client prediction tick-by-tick using predictor + actions
    - send periodic updates every K tics since last send
    - send correction if prediction error exceeds thresholds
    """
    if len(states) != len(action_masks):
        raise ValueError("states and action_masks must have equal length")
    if len(states) < 2:
        raise ValueError("Need at least 2 states")

    predictor.reset()

    events: list[SendEvent] = []
    last_send_tic = states[0].tic

    # Initial authoritative send
    events.append(SendEvent(send_tic=states[0].tic, kind="initial", state=states[0]))

    # Client starts from the authoritative initial state
    predictor.observe(_state_to_obs(states[0], action_masks[0]))

    for i in range(1, len(states)):
        gt = states[i]

        # advance predictor exactly one tic using action at previous tic
        pred_obs = predictor.step(action_masks[i - 1])

        pos_err = _euclid3(pred_obs.x - gt.x, pred_obs.y - gt.y, pred_obs.z - gt.z)
        yaw_err = circular_abs_error_deg(pred_obs.yaw, gt.yaw)
        pitch_err = circular_abs_error_deg(pred_obs.pitch, gt.pitch)

        need_periodic = (gt.tic - last_send_tic) >= cfg.update_interval_tics
        need_correction = (
            pos_err > cfg.pos_eps or yaw_err > cfg.yaw_eps_deg or pitch_err > cfg.pitch_eps_deg
        )

        if need_correction:
            events.append(SendEvent(send_tic=gt.tic, kind="correction", state=gt))
            last_send_tic = gt.tic
            predictor.reset()
            predictor.observe(_state_to_obs(gt, action_masks[i]))
            continue

        if need_periodic:
            events.append(SendEvent(send_tic=gt.tic, kind="periodic", state=gt))
            last_send_tic = gt.tic
            predictor.reset()
            predictor.observe(_state_to_obs(gt, action_masks[i]))
            continue
    return events
    # If no send, continue predicting next tic on next loop
    # Ensure predictor has correct action_mask for current predicted obs
    # (pred_obs already stores action_mask used for the step)


def simulate_network_delivery(
    events: Sequence[SendEvent], cfg: NetConfig
) -> tuple[list[ArrivalEvent], NetSummary]:
    """
    Apply latency/jitter/loss and optional bandwidth cap.
    Bandwidth cap creates queueing delay: packets transmit sequentially.

    Returns:
      arrivals: list of arrival events (including dropped flags)
      net_summary: aggregated stats
    """
    if cfg.tics_per_second <= 0:
        raise ValueError("tics_per_second must be > 0")
    if not (0.0 <= cfg.loss_rate <= 1.0):
        raise ValueError("loss_rate must be in [0,1]")
    if cfg.header_bytes < 0 or cfg.payload_bytes_state < 0:
        raise ValueError("message sizes must be non-negative")

    rng = random.Random(int(cfg.rng_seed))

    bandwidth_bps = 0.0
    if cfg.bandwidth_kbps > 0:
        bandwidth_bps = float(cfg.bandwidth_kbps) * 1000.0

    next_tx_free_time_s = 0.0
    max_queue_delay_s = 0.0

    arrivals: list[ArrivalEvent] = []

    sent_packets = 0
    delivered_packets = 0
    dropped_packets = 0
    sent_bytes = 0
    delivered_bytes = 0
    delay_sum_ms = 0.0
    delay_count = 0

    for ev in events:
        send_time_s = _tic_to_time_s(ev.send_tic, cfg.tics_per_second)
        size_bytes = int(cfg.header_bytes + cfg.payload_bytes_state)

        tx_start_s = max(send_time_s, next_tx_free_time_s)
        queue_delay_s = tx_start_s - send_time_s
        max_queue_delay_s = max(max_queue_delay_s, queue_delay_s)

        tx_dur_s = 0.0
        if bandwidth_bps > 0.0:
            tx_dur_s = (size_bytes * 8.0) / bandwidth_bps

        tx_end_s = tx_start_s + tx_dur_s
        next_tx_free_time_s = tx_end_s

        # Jitter: uniform in [-j, +j]
        jitter_s = 0.0
        if cfg.jitter_ms > 0:
            jitter_s = (rng.uniform(-cfg.jitter_ms, cfg.jitter_ms)) / 1000.0

        latency_s = cfg.latency_ms / 1000.0
        arrival_time_s = tx_end_s + latency_s + jitter_s

        dropped = rng.random() < cfg.loss_rate

        sent_packets += 1
        sent_bytes += size_bytes

        if dropped:
            dropped_packets += 1
        else:
            delivered_packets += 1
            delivered_bytes += size_bytes
            one_way_delay_ms = (arrival_time_s - send_time_s) * 1000.0
            delay_sum_ms += one_way_delay_ms
            delay_count += 1

        arrivals.append(
            ArrivalEvent(
                arrival_time_s=arrival_time_s,
                kind=ev.kind,
                state=ev.state,
                size_bytes=size_bytes,
                dropped=dropped,
                send_time_s=send_time_s,
            )
        )

    arrivals.sort(key=lambda a: a.arrival_time_s)

    avg_delay_ms = (delay_sum_ms / delay_count) if delay_count > 0 else 0.0
    net_summary = NetSummary(
        sent_packets=sent_packets,
        delivered_packets=delivered_packets,
        dropped_packets=dropped_packets,
        sent_bytes=sent_bytes,
        delivered_bytes=delivered_bytes,
        avg_one_way_delay_ms=avg_delay_ms,
        max_queue_delay_ms=max_queue_delay_s * 1000.0,
    )
    return arrivals, net_summary


def simulate_client_error(
    states: Sequence[State], arrivals: Sequence[ArrivalEvent], tps: float
) -> ClientErrorSummary:
    """
    Client applies delivered authoritative updates on arrival_time_s.
    Between updates, client predicts from last received state using constant rates
    estimated from the last two received authoritative updates.

    Error is measured against ground truth states at each tic.
    """
    if len(states) < 2:
        raise ValueError("Need at least 2 states")
    if tps <= 0:
        raise ValueError("tics_per_second must be > 0")

    # Prepare arrival cursor
    idx = 0
    delivered: list[ArrivalEvent] = [a for a in arrivals if not a.dropped]
    delivered.sort(key=lambda a: a.arrival_time_s)

    prev_recv: State | None = None
    last_recv: State | None = None
    rates = (0.0, 0.0, 0.0, 0.0, 0.0)

    # Error accumulators
    pos_sum = 0.0
    yaw_sum = 0.0
    pitch_sum = 0.0
    cnt = 0

    pos_max = 0.0
    yaw_max = 0.0
    pitch_max = 0.0

    for gt in states:
        now_s = _tic_to_time_s(gt.tic, tps)

        # Apply all arrivals up to now
        while idx < len(delivered) and delivered[idx].arrival_time_s <= now_s:
            upd = delivered[idx].state
            prev_recv = last_recv
            last_recv = upd
            rates = _estimate_rates(prev_recv, last_recv)
            idx += 1

        if last_recv is None:
            # No update has arrived yet; client has no state ->
            # treat as predicting from first gt (idealized start).
            # This keeps the metric finite and isolates network effects after first delivery.
            last_recv = states[0]
            prev_recv = None
            rates = (0.0, 0.0, 0.0, 0.0, 0.0)

        pred = _predict(last_recv, rates, gt.tic)

        pos_err = _euclid3(pred.x - gt.x, pred.y - gt.y, pred.z - gt.z)
        yaw_err = circular_abs_error_deg(pred.yaw, gt.yaw)
        pitch_err = circular_abs_error_deg(pred.pitch, gt.pitch)

        pos_sum += pos_err
        yaw_sum += yaw_err
        pitch_sum += pitch_err
        cnt += 1

        pos_max = max(pos_max, pos_err)
        yaw_max = max(yaw_max, yaw_err)
        pitch_max = max(pitch_max, pitch_err)

    return ClientErrorSummary(
        pos_mae=pos_sum / max(cnt, 1),
        yaw_mae_deg=yaw_sum / max(cnt, 1),
        pitch_mae_deg=pitch_sum / max(cnt, 1),
        pos_max=pos_max,
        yaw_max_deg=yaw_max,
        pitch_max_deg=pitch_max,
    )


def simulate_client_error_with_predictor(
    states: Sequence[State],
    action_masks: Sequence[int],
    arrivals: Sequence[ArrivalEvent],
    predictor: Predictor,
    tps: float,
) -> ClientErrorSummary:
    """
    Client applies delivered authoritative updates on arrival_time_s.
    Between updates, client predicts per tic using predictor.step(action_mask).

    We model "apply update at arrival time, then fast-forward to current tic"
    (no rollback reconciliation), consistent with our current evaluation simplification.
    """
    if len(states) != len(action_masks):
        raise ValueError("states and action_masks must have equal length")
    if len(states) < 2:
        raise ValueError("Need at least 2 states")
    if tps <= 0:
        raise ValueError("tics_per_second must be > 0")

    delivered = [a for a in arrivals if not a.dropped]
    delivered.sort(key=lambda a: a.arrival_time_s)

    # Index ground-truth by tic (assumes strictly increasing, 1-tic steps)
    tic0 = states[0].tic
    by_tic = {s.tic: s for s in states}
    mask_by_tic = {states[i].tic: action_masks[i] for i in range(len(states))}

    predictor.reset()

    # Initialize client at first ground-truth state (idealized start)
    cur_tic = tic0
    predictor.observe(_state_to_obs(by_tic[cur_tic], mask_by_tic[cur_tic]))
    client_obs = _state_to_obs(by_tic[cur_tic], mask_by_tic[cur_tic])

    idx = 0

    pos_sum = yaw_sum = pitch_sum = 0.0
    pos_max = yaw_max = pitch_max = 0.0
    cnt = 0

    for gt in states:
        now_s = _tic_to_time_s(gt.tic, tps)

        # Apply any arrivals that have arrived by now
        while idx < len(delivered) and delivered[idx].arrival_time_s <= now_s:
            upd_state = delivered[idx].state
            if upd_state.tic in by_tic:
                # Apply correction "now" using authoritative pose at upd_state.tic
                predictor.reset()
                cur_tic = upd_state.tic
                client_obs = _state_to_obs(by_tic[cur_tic], mask_by_tic.get(cur_tic, 0))
                predictor.observe(client_obs)
            idx += 1

        # Fast-forward client to current ground-truth tic
        while cur_tic < gt.tic:
            # Some logs/sessions may have missing tics (dropped lines).
            # Treat missing masks as "no input".
            am = mask_by_tic.get(cur_tic, 0)
            client_obs = predictor.step(am)
            cur_tic = client_obs.tic

        pos_err = _euclid3(client_obs.x - gt.x, client_obs.y - gt.y, client_obs.z - gt.z)
        yaw_err = circular_abs_error_deg(client_obs.yaw, gt.yaw)
        pitch_err = circular_abs_error_deg(client_obs.pitch, gt.pitch)

        pos_sum += pos_err
        yaw_sum += yaw_err
        pitch_sum += pitch_err
        cnt += 1

        pos_max = max(pos_max, pos_err)
        yaw_max = max(yaw_max, yaw_err)
        pitch_max = max(pitch_max, pitch_err)

    return ClientErrorSummary(
        pos_mae=pos_sum / max(cnt, 1),
        yaw_mae_deg=yaw_sum / max(cnt, 1),
        pitch_mae_deg=pitch_sum / max(cnt, 1),
        pos_max=pos_max,
        yaw_max_deg=yaw_max,
        pitch_max_deg=pitch_max,
    )


def evaluate_dr_under_network(
    states: Sequence[State],
    dr_cfg: DRConfig,
    net_cfg: NetConfig,
) -> DRNetSummary:
    """
    Full evaluation:
    - Plan DR send schedule (server-side)
    - Simulate delivery under network constraints
    - Compute client error vs ground truth
    - Compute packet/byte savings vs baseline "send full state each tic"
    """
    if len(states) < 2:
        raise ValueError("Need at least 2 states")

    dr_events = plan_dr_sends(states, dr_cfg)
    if not isinstance(dr_events, list):
        raise RuntimeError("plan_dr_sends did not return a list of SendEvent")

    arrivals, net_summary = simulate_network_delivery(dr_events, net_cfg)
    client_err = simulate_client_error(states, arrivals, net_cfg.tics_per_second)

    # Baseline: send full state each tic (same size model)
    baseline_packets = len(states)
    msg_size = int(net_cfg.header_bytes + net_cfg.payload_bytes_state)
    baseline_bytes = baseline_packets * msg_size

    dr_packets = net_summary.sent_packets
    dr_bytes = net_summary.sent_bytes

    savings_packets = 1.0 - (dr_packets / baseline_packets) if baseline_packets > 0 else 0.0
    savings_bytes = 1.0 - (dr_bytes / baseline_bytes) if baseline_bytes > 0 else 0.0

    action_masks = [0 for _ in states]
    dr_pred = DRPredictor()
    dr_events = plan_predictor_sends(states, action_masks, dr_pred, dr_cfg)
    arrivals, net_summary = simulate_network_delivery(dr_events, net_cfg)
    client_err = simulate_client_error_with_predictor(
        states, action_masks, arrivals, DRPredictor(), net_cfg.tics_per_second
    )

    # IMPORTANT: Make DR packet/byte counts consistent with the predictor-based run.
    dr_packets = net_summary.sent_packets
    dr_bytes = net_summary.sent_bytes

    savings_packets = 1.0 - (dr_packets / baseline_packets) if baseline_packets > 0 else 0.0
    savings_bytes = 1.0 - (dr_bytes / baseline_bytes) if baseline_bytes > 0 else 0.0

    return DRNetSummary(
        n_tics=len(states),
        dr_send_packets=dr_packets,
        dr_send_bytes=dr_bytes,
        baseline_send_packets=baseline_packets,
        baseline_send_bytes=baseline_bytes,
        savings_ratio_packets=savings_packets,
        savings_ratio_bytes=savings_bytes,
        net=net_summary,
        client_error=client_err,
    )


def evaluate_predictor_under_network_factory(
    states: Sequence[State],
    action_masks: Sequence[int],
    predictor_factory: Callable[[], Predictor],
    dr_cfg: DRConfig,
    net_cfg: NetConfig,
) -> DRNetSummary:
    if len(states) != len(action_masks):
        raise ValueError("states and action_masks must have equal length")
    if len(states) < 2:
        raise ValueError("Need at least 2 states")

    server_pred = predictor_factory()
    events = plan_predictor_sends(states, action_masks, server_pred, dr_cfg)
    if not isinstance(events, list):
        raise RuntimeError("plan_predictor_sends did not return a list of SendEvent")

    arrivals, net_summary = simulate_network_delivery(events, net_cfg)

    client_pred = predictor_factory()
    client_err = simulate_client_error_with_predictor(
        states=states,
        action_masks=action_masks,
        arrivals=arrivals,
        predictor=client_pred,
        tps=net_cfg.tics_per_second,
    )

    baseline_packets, baseline_bytes = baseline_traffic(states, net_cfg)

    dr_packets = net_summary.sent_packets
    dr_bytes = net_summary.sent_bytes

    savings_packets = 1.0 - (dr_packets / baseline_packets) if baseline_packets > 0 else 0.0
    savings_bytes = 1.0 - (dr_bytes / baseline_bytes) if baseline_bytes > 0 else 0.0

    return DRNetSummary(
        n_tics=len(states),
        dr_send_packets=dr_packets,
        dr_send_bytes=dr_bytes,
        baseline_send_packets=baseline_packets,
        baseline_send_bytes=baseline_bytes,
        savings_ratio_packets=savings_packets,
        savings_ratio_bytes=savings_bytes,
        net=net_summary,
        client_error=client_err,
    )


def evaluate_predictor_under_network(
    states: Sequence[State],
    action_masks: Sequence[int],
    predictor: Predictor,
    dr_cfg: DRConfig,
    net_cfg: NetConfig,
) -> DRNetSummary:
    """
    Generic evaluator: same structure as evaluate_dr_under_network, but uses:
    - action_masks for prediction steps
    - arbitrary Predictor implementation
    """
    if len(states) != len(action_masks):
        raise ValueError("states and action_masks must have equal length")
    if len(states) < 2:
        raise ValueError("Need at least 2 states")

    # 1) Plan server sends based on predictor divergence (corrections) + periodic updates
    events = plan_predictor_sends(states, action_masks, predictor, dr_cfg)
    if not isinstance(events, list):
        raise RuntimeError("plan_predictor_sends did not return a list of SendEvent")

    # 2) Deliver them through network model
    arrivals, net_summary = simulate_network_delivery(events, net_cfg)

    # 3) Simulate client error using predictor between delivered updates
    client_err = simulate_client_error_with_predictor(
        states=states,
        action_masks=action_masks,
        arrivals=arrivals,
        predictor=predictor.__class__(),  # fresh predictor instance for client sim
        tps=net_cfg.tics_per_second,
    )

    # 4) Baseline traffic (same definition as evaluate_dr_under_network)
    baseline_packets, baseline_bytes = baseline_traffic(states, net_cfg)

    # Use net_summary for sent bytes/packets (consistent with evaluate_dr_under_network)
    dr_packets = net_summary.sent_packets
    dr_bytes = net_summary.sent_bytes

    savings_packets = 1.0 - (dr_packets / baseline_packets) if baseline_packets > 0 else 0.0
    savings_bytes = 1.0 - (dr_bytes / baseline_bytes) if baseline_bytes > 0 else 0.0

    return DRNetSummary(
        n_tics=len(states),
        dr_send_packets=dr_packets,
        dr_send_bytes=dr_bytes,
        baseline_send_packets=baseline_packets,
        baseline_send_bytes=baseline_bytes,
        savings_ratio_packets=savings_packets,
        savings_ratio_bytes=savings_bytes,
        net=net_summary,
        client_error=client_err,
    )
