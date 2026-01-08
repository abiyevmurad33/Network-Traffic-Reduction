# Data Schema (v1.0) — ViZDoom Behavior Logging

## 0. Purpose
This schema defines the canonical format for logging per-tic player behavior/state from ViZDoom so we can later:
- train behavior predictors (offline > online),
- evaluate prediction error vs baselines (Dead Reckoning),
- simulate network traffic (baseline vs DR vs AI+sync).

This schema is the *ground truth dataset format* (not the network protocol).

---

## 1. Time base and identity

### 1.1 Time unit
- **tic**: integer simulation step (Doom-style tic).
- Each record corresponds to one player at one tic.

Required fields:
- session_id (string): unique run identifier
- 	ic (int): non-negative, monotonically increasing within a session
- player_id (string): stable identifier per player within the session

Optional (derivable):
- 	_seconds (float): 	ic / tics_per_second (stored in metadata)

### 1.2 Tick rate
	ics_per_second must be stored in session.json. Do not repeat it in every row.

---

## 2. Files per session (canonical layout)

For each run, create a session directory:
data/raw_logs/<session_id>/

Inside it, write:

1) states.csv — per-tic, per-player rows
2) session.json — session metadata (scenario, seed, tick rate, action mapping)

v1 scope: **single-player logging** (one player_id), multiplayer added later.

---

## 3. states.csv (required columns)

### 3.1 Required columns (v1.0)
states.csv must contain these columns exactly:

- session_id (string)
- 	ic (int)
- player_id (string)
- pos_x (float)
- pos_y (float)
- pos_z (float)
- yaw_deg (float)
- pitch_deg (float)
- ction_mask (int)

### 3.2 Units and conventions
- Positions are in engine/world units as provided by ViZDoom.
- Angles are in **degrees**, normalized to **(-180, 180]**.
- ction_mask is a multi-binary button bitmask (see session.json).

### 3.3 Optional columns (allowed, not required)
May include for debugging:
- ction_id (int)
- ction_vector (string) e.g. "0,1,0,0,1"

Canonical action representation is ction_mask.

---

## 4. Action encoding

### 4.1 Bitmask definition
ction_mask is an integer bitmask encoding the pressed buttons for that tic.

- Bit positions are defined by session.json.action_space.labels.
- If labels[0] = "MOVE_FORWARD", then bit 0 corresponds to MOVE_FORWARD.

Example:
- labels = ["MOVE_FORWARD","MOVE_BACKWARD","MOVE_LEFT","MOVE_RIGHT","ATTACK"]
- action_mask = 1 means MOVE_FORWARD
- action_mask = 16 means ATTACK
- action_mask = 17 means MOVE_FORWARD + ATTACK

---

## 5. session.json (required keys)

Each session directory must include a session.json containing at minimum:

- schema_version: "1.0"
- session_id: string
- created_utc: ISO-8601 timestamp
- engine: "vizdoom"
- scenario_name: string (config/scenario identifier)
- wad: string (e.g., "freedoom2.wad" or "doom2.wad")
- seed: int (if deterministic; else null)
- 	ics_per_second: number
- players: array of objects:
  - { "player_id": "p1", "controller": "human|bot|script|agent", "notes": "" }

Action mapping:
- ction_space:
  - 	ype: "multi_binary_bitmask"
  - labels: ordered list of action labels; index = bit position

---

## 6. Data integrity rules (must hold)

Within one session_id:

1. 	ic is monotonic and integer.
2. For each (tic, player_id) there is at most one row.
3. Required numeric fields contain no NaN/Inf.
4. yaw_deg and pitch_deg are always normalized to (-180, 180].
5. ction_mask is non-negative and consistent with session.json.action_space.labels.

---

## 7. Example outputs

### 7.1 Example states.csv header
session_id,tic,player_id,pos_x,pos_y,pos_z,yaw_deg,pitch_deg,action_mask

### 7.2 Example row
session_2026-01-08T09-12-33Z,140,p1,1024.50,512.25,0.00,-90.00,0.00,17

### 7.3 Minimal session.json example
{
  "schema_version": "1.0",
  "session_id": "session_2026-01-08T09-12-33Z",
  "created_utc": "2026-01-08T09:12:33Z",
  "engine": "vizdoom",
  "scenario_name": "basic",
  "wad": "freedoom2.wad",
  "seed": 12345,
  "tics_per_second": 35,
  "players": [
    { "player_id": "p1", "controller": "script", "notes": "" }
  ],
  "action_space": {
    "type": "multi_binary_bitmask",
    "labels": ["MOVE_FORWARD","MOVE_BACKWARD","MOVE_LEFT","MOVE_RIGHT","ATTACK"]
  }
}

---

## 8. Versioning policy
- Backward-compatible additions increment minor (1.0 > 1.1).
- Breaking changes increment major (1.x > 2.0).
