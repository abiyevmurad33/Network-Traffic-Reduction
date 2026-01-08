# Logger Design (v1.0)

## 0. Goal
Implement a deterministic, correct ViZDoom session logger that produces:
- data/raw_logs/<session_id>/states.csv
- data/raw_logs/<session_id>/session.json

It must satisfy:
- docs/data_schema.md v1.0
- docs/experiment_protocol.md v1.0
- docs/logger_requirements.md v1.0

v1.0 scope: single-player only.

---

## 1. Public entry point and files

### 1.1 Files to create
- src/logger/run_logger.py  (main entry point; CLI)
- src/utils/paths.py        (safe path helpers)
- src/utils/time_id.py      (session_id generator)
- src/utils/angles.py       (angle normalization + circular utilities)
- src/logger/action_space.py (bitmask encoding/decoding)
- src/logger/controller.py   (scripted deterministic controller)
- src/logger/vizdoom_env.py  (wrapper around DoomGame init/run)
- scripts/validate_log.py    (post-run validation script)
- scripts/smoke_run.ps1      (runs 10s Tier A session)

We will keep modules small and focused. No hidden globals.

---

## 2. CLI interface (run_logger.py)

### 2.1 Arguments
python -m src.logger.run_logger (or direct script execution) supports:

Required/Default:
- --scenario-config scenarios/basic_movement.cfg
- --duration-seconds 120
- --protocol-tier A
- --seed 12345 (required if Tier A)
- --session-id <auto> (optional; generated if absent)
- --output-root data/raw_logs
- --window-visible true|false (default: true for debugging)

### 2.2 Output contract
On success, it prints:
- session_id
- output directory path
- number of tics recorded

On failure, it prints a clear error and exits non-zero.

---

## 3. Initialization sequence (vizdoom_env.py)

### 3.1 Steps
1. Validate required files exist:
   - scenario config path
   - reedoom2.wad in repo root
   - scenario wad referenced by config (scenarios/basic.wad)
2. Construct zd.DoomGame()
3. game.load_config(<scenario_config_path>)
4. Apply runtime overrides (only if needed):
   - window visibility (optional)
5. game.init()
6. Start new episode:
   - game.new_episode()

Note: If ViZDoom exposes seed control, apply it according to Tier A before 
ew_episode(); otherwise seed any Python RNGs used by controller.

---

## 4. Per-tic loop (core logging)

### 4.1 Tic source of truth
We will use ViZDoom's episode time as tic counter:
- 	ic = game.get_episode_time()

We require monotonic integer tics. If not monotonic, abort.

### 4.2 Reading state
At each tic:
- state = game.get_state()
- Extract position and view angles from game variables or objects (exact API finalized during implementation; must be documented in code comments).
- Normalize angles to (-180, 180] via src/utils/angles.py.

### 4.3 Choosing and executing action
- Use deterministic scripted controller from controller.py to produce:
  - multi-binary list aligned with action labels order
- Encode bitmask using ction_space.py:
  - ction_mask = encode(button_bools)
- Call:
  - game.make_action(action_vector, 1) to advance exactly 1 tic.

### 4.4 Row generation
For each tic, write one row with required columns:
- session_id, tic, player_id="p1",
- pos_x,pos_y,pos_z,
- yaw_deg,pitch_deg,
- action_mask

Important ordering:
- We log the state *before* applying action for that tic (state at time t).
- This must be consistent across all sessions.

---

## 5. Action space module (action_space.py)

### 5.1 Canonical labels order (bit positions)
Bit 0: MOVE_FORWARD
Bit 1: MOVE_BACKWARD
Bit 2: MOVE_LEFT
Bit 3: MOVE_RIGHT
Bit 4: TURN_LEFT
Bit 5: TURN_RIGHT
Bit 6: ATTACK

Functions:
- encode(buttons: list[bool]) -> int
- decode(mask: int) -> list[bool]
- labels() -> list[str]

---

## 6. Scripted controller (controller.py)

### 6.1 Design goals
- Fully deterministic given seed.
- Exercises movement + turning + occasional attack.
- Must not require human input.

### 6.2 Initial policy (v1.0)
A repeating pattern over a fixed period, optionally seeded to choose between a small set of patterns.

Example pattern idea:
- Phase 1: move forward + turn left (N tics)
- Phase 2: strafe right + turn right (N tics)
- Phase 3: move backward (N tics)
- Phase 4: attack bursts while turning (N tics)
- Repeat

Implementation must document exact durations and ensure no contradictory buttons (e.g., forward+back simultaneously unless intentionally tested).

---

## 7. Writing outputs safely

### 7.1 Atomic writes
Write to temp files first:
- states.csv.tmp
- session.json.tmp

On successful completion:
- rename to final names.

If run fails:
- leave tmp files for debugging, but do not produce partial final outputs.

### 7.2 CSV writer
Use Python csv module with explicit header ordering.

---

## 8. session.json creation

### 8.1 Values
session.json must include:
- schema_version="1.0"
- session_id
- created_utc
- engine="vizdoom"
- scenario_name="basic_movement" (basename)
- scenario_config_path="scenarios/basic_movement.cfg"
- wad="freedoom2.wad"
- seed (Tier A required)
- tics_per_second (expected 35; validate after init if possible)
- duration_seconds (user arg)
- warmup_seconds=0
- protocol_tier
- ended_early, tics_recorded, end_reason (if early termination)
- players=[{"player_id":"p1","controller":"script","notes":""}]
- action_space={"type":"multi_binary_bitmask","labels":[...]}

---

## 9. Validation script (scripts/validate_log.py)

### 9.1 Checks
Given a session directory:
- states.csv exists
- session.json exists
- required columns exist in correct order
- tics monotonic
- angles within (-180, 180]
- action_mask integer >= 0
- action_space.labels matches canonical list

Exit code:
- 0 on pass
- non-zero on fail with message

---

## 10. Smoke run helper (scripts/smoke_run.ps1)

Runs:
- duration_seconds=10
- protocol_tier=A
- seed=12345
- window_visible=true

Then runs validate_log.py on produced directory.

---

## 11. Open questions resolved for v1.0
- Platform: Windows native
- Python: 3.11
- Log format: CSV + session.json
- Angle range: (-180, 180]
- Action encoding: bitmask
- v1 logger: single-player only
- Controller: deterministic scripted policy

