# Experiment Protocol (v1.0)

## 0. Purpose
This document standardizes how we run ViZDoom sessions and store outputs so results are comparable and reproducible.

This protocol is designed to support:
- behavior logging (states.csv + session.json),
- offline training datasets,
- evaluation of prediction error,
- simulated traffic accounting.

---

## 1. Session definition
A **session** is one run of a fixed scenario configuration for a fixed duration, producing:
- data/raw_logs/<session_id>/states.csv
- data/raw_logs/<session_id>/session.json

A session may be controlled by:
- scripted policy (recommended for baseline experiments),
- bot/agent,
- human input (allowed; not used for strict reproducibility runs unless recorded).

---

## 2. Session identifiers and directory layout

### 2.1 Session ID format (canonical)
Use UTC timestamp-based IDs:
- session_<YYYY-MM-DDTHH-MM-SSZ>

Example:
- session_2026-01-08T09-12-33Z

### 2.2 Output directories
Per session:
- data/raw_logs/<session_id>/
  - states.csv
  - session.json

We do not commit raw logs to Git. Only small curated samples belong in:
- data/examples/

---

## 3. Scenario configuration and assets

### 3.1 Scenario config location
All scenario configs used by this project must live in:
- scenarios/

Each session references its scenario by:
- scenario_name in session.json (base name)
- optional scenario_config_path for traceability

### 3.2 WAD / asset policy
Default: Freedoom assets (redistributable).
If original Doom WADs are used, they must not be committed and must be referenced only by file name in session.json.

---

## 4. Determinism and seeds

### 4.1 Reproducibility tiers
We define two tiers:

**Tier A — Reproducible sessions (research baseline)**
- seed is required and recorded in session.json.
- Use scripted/bot/agent control.
- Used for comparisons between baseline vs DR vs AI.

**Tier B — Non-reproducible sessions (exploration)**
- seed may be null.
- Human input allowed.
- Used for qualitative evaluation only (not primary tables).

Unless explicitly stated, all reported metrics must come from Tier A sessions.

---

## 5. Duration, warm-up, and sampling

### 5.1 Default duration
Default session duration:
- 120 seconds (2 minutes)

We may adjust later, but any change must be recorded in session.json:
- duration_seconds

### 5.2 Warm-up window
If a warm-up period is used (e.g., model training without traffic reduction), it must be recorded:
- warmup_seconds

For initial logging sessions, warm-up is 0.

---

## 6. Required metadata keys (extension of data schema)

In addition to the required keys in docs/data_schema.md, session.json must include:

- duration_seconds: number
- protocol_tier: "A" or "B"
- 
otes: string (free text)
- scenario_config_path: string (relative path), when applicable

---

## 7. Execution record (manual checklist)
For every Tier A session, record:
- scenario name
- seed
- duration
- controller type

This information must be present in session.json. No separate lab notebook is required.

---

## 8. Acceptance criteria for a valid session
A session is considered valid if:
1. states.csv exists and has the required columns from docs/data_schema.md.
2. session.json exists and contains all required keys (schema + protocol).
3. states.csv contains at least duration_seconds * tics_per_second rows for single-player (allowing small deviations if episode ends early; must be recorded).
4. 	ic starts at 0 or 1 consistently (to be standardized when the logger is implemented) and is monotonic.
5. ction_mask is consistent with session.json.action_space.labels.

---

## 9. Change control
Any change to:
- action space labels order,
- angle convention,
- scenario config,
- duration defaults,
must be documented via commit message and reflected in session.json for affected sessions.

