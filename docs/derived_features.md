# Derived Features (v1)

This document defines features that are *not logged* in v1, but derived deterministically from states.csv + session.json.

## 1. Velocities (derived)
Given 	ics_per_second from session.json and per-player ordered rows by 	ic:

- el_x[t] = (pos_x[t] - pos_x[t-1]) * tics_per_second
- el_y[t] = (pos_y[t] - pos_y[t-1]) * tics_per_second
- el_z[t] = (pos_z[t] - pos_z[t-1]) * tics_per_second

For the first tic in a session for a player, define velocity as 0.0 or mark as missing (to be decided in evaluation code).
