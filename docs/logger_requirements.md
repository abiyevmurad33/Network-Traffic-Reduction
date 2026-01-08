# Logger Requirements (v1.0)

## 0. Purpose
The logger produces a per-tic dataset from ViZDoom suitable for:
- training behavior predictors,
- evaluating prediction error (vs Dead Reckoning),
- simulated traffic accounting.

This is a *ground-truth logger*. It must be correct, stable, and reproducible.

Scope (v1.0):
- Single-player only.
- Uses the project scenario config: scenarios/basic_movement.cfg.

Non-goals (v1.0):
- Multiplayer logging.
- Real network capture.
- Any Zandronum integration.

---

## 1. Inputs

### 1.1 Required runtime inputs
The logger must accept (via CLI args or config; chosen later):
- scenario_config_path (string): default scenarios/basic_movement.cfg
- duration_seconds (int/float): default 120
- protocol_tier (string): default "A" (reproducible)
- seed (int): required when protocol_tier == "A"
- session_id (string): optional; if not provided, generate per protocol:
  session_<YYYY-MM-DDTHH-MM-SSZ>

### 1.2 Required environment dependencies
- Python 3.11
- izdoom installed and importable
- Base IWAD available locally (repo root):
  - reedoom2.wad (default)
- Scenario files present:
  - scenarios/basic.wad
  - scenarios/basic_movement.cfg

---

## 2. Outputs

### 2.1 Output directory layout
For each run, create:
- data/raw_logs/<session_id>/states.csv
- data/raw_logs/<session_id>/session.json

The logger must create missing directories as needed.

### 2.2 Output schema
states.csv must exactly follow docs/data_schema.md v1.0 required columns:
- session_id,tic,player_id,pos_x,pos_y,pos_z,yaw_deg,pitch_deg,action_mask

session.json must satisfy:
- docs/data_schema.md required keys
- docs/experiment_protocol.md required keys
Including:
- schema_version="1.0"
- engine="vizdoom"
- scenario_name (derived from cfg basename, e.g., "basic_movement")
- scenario_config_path (relative path)
- wad (string, e.g., "freedoom2.wad")
- seed (int or null per protocol tier)
- tics_per_second (from config; expected 35)
- duration_seconds
- protocol_tier ("A" or "B")
- warmup_seconds (default 0)
- players array: single element for v1.0
- action_space definition (bitmask labels order)

---

## 3. Action space (canonical v1.0)

### 3.1 Bitmask labels order (must be fixed)
For v1.0, the canonical bit order is:

Bit 0: MOVE_FORWARD  
Bit 1: MOVE_BACKWARD  
Bit 2: MOVE_LEFT  
Bit 3: MOVE_RIGHT  
Bit 4: TURN_LEFT  
Bit 5: TURN_RIGHT  
Bit 6: ATTACK  

This order must be recorded in session.json.action_space.labels and used consistently when encoding ction_mask.

Note: This list must match the enabled buttons in scenarios/basic_movement.cfg.

---

## 4. Angle normalization and error safety

### 4.1 Angle normalization
The logger must normalize:
- yaw_deg to (-180, 180]
- pitch_deg to (-180, 180]

Normalization must be consistent across all rows.

### 4.2 Numeric validity
The logger must ensure required numeric fields are finite (no NaN/Inf).
If invalid data is observed, the logger must:
- stop the run,
- write a clear error message,
- not produce partial ambiguous output.

---

## 5. Time base and row generation

### 5.1 Tic handling
- 	ic must be integer and monotonic.
- For v1.0 single-player:
  - exactly one row per tic.

### 5.2 Duration handling
Given duration_seconds and 	ics_per_second, the logger targets:
- 	arget_tics = duration_seconds * tics_per_second

If the episode terminates early:
- Logger must record this in session.json:
  - ended_early: true
  - 	ics_recorded: <int>
  - end_reason: <string> (best-effort)

---

## 6. Determinism (Protocol Tier A)

### 6.1 Tier A requirements
For Tier A sessions:
- seed must be provided.
- session.json.seed must equal the provided seed.
- Any relevant RNGs used by the logger must be seeded deterministically.
- Controller must be non-human (scripted actions) unless we later add input recording.

Tier B is allowed by protocol but not required for v1.0 implementation.

---

## 7. Controller policy (v1.0)

### 7.1 Default controller
To ensure reproducibility, v1.0 logger runs with a deterministic scripted controller:
- a small finite-state or pattern-based movement policy that exercises:
  - forward/back/strafe
  - turning left/right
  - occasional attack

The exact scripted policy will be defined in the logger implementation docstring and should remain stable once results collection begins.

---

## 8. Failure modes and required error messages

The logger must fail fast with clear messages if:
- izdoom import fails
- scenario config file not found
- scenario WAD missing
- IWAD missing (reedoom2.wad absent)
- ViZDoom initialization fails
- output directory cannot be created
- schema columns cannot be produced

Error messages must include:
- which file/path is missing
- which step failed (init, episode start, state read, write)

---

## 9. Validation plan (acceptance tests)

### 9.1 Smoke run
A smoke run is defined as:
- duration_seconds = 10
- Tier A seed = 12345
- produces states.csv and session.json under a new session directory
- states.csv contains at least 10 * tics_per_second - 5 rows (small tolerance for startup)

### 9.2 Schema validation
A validation check must confirm:
- required CSV columns exist in correct order
- ction_mask is non-negative integer
- angles are within (-180, 180]
- session.json contains required keys
- session.json.action_space.labels matches canonical bit order

These checks may be implemented as a small scripts/validate_log.py later.

---

## 10. Change control
Any change to:
- action_space labels order
- angle convention
- scenario config
must bump schema/protocol minor version or be explicitly documented in commit messages and reflected in session.json.

