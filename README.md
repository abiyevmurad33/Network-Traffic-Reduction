# Reducing Network Traffic in FPS Games via Predictive Player Modeling

This repository contains the experimental codebase for a master’s thesis investigating network traffic reduction in multiplayer first-person shooter (FPS) games using short-horizon player behavior prediction. The target engine for applicability is **Zandronum**, while **ViZDoom** is currently used as a controlled experimental environment for data collection, simulation, and evaluation.

The central research question is whether learned player behavior models can reduce server-to-client update traffic beyond classical Dead Reckoning approaches, while maintaining bounded state error through corrective mechanisms.

---

## Background and motivation

Multiplayer FPS games require frequent synchronization of player state to preserve a consistent game world across clients. Classical approaches either transmit authoritative state at a fixed rate or apply Dead Reckoning (DR), where the server sends updates only when the client’s locally extrapolated state exceeds an error threshold.

Dead Reckoning significantly reduces traffic compared to full replication, but it relies on simple kinematic assumptions and does not adapt to individual player behavior. This limitation motivates the use of learned predictors that exploit short-term temporal regularities in player motion and view control.

The hypothesis explored in this work is that **accurate short-horizon prediction** can further reduce update frequency beyond DR, provided that prediction is bounded by explicit correction and confidence mechanisms.

---

## Experimental setting

All experiments are currently conducted in ViZDoom. This environment allows deterministic scenario design, scripted player behavior, and controlled simulation of adverse network conditions, making it suitable for repeatable evaluation.

The experimental pipeline consists of:
- ViZDoom scenarios generating streams of player state (position and view orientation),
- A network simulation layer modeling latency, jitter, packet loss, and queueing delay,
- Multiple update policies applied to the same underlying state stream.

Architectural choices and timing assumptions are made with Zandronum’s networking model in mind, enabling later transfer of results.

---

## Current state of the implementation

At present, the repository provides a stable evaluation framework for baseline replication and Dead Reckoning under simulated network conditions. ViZDoom scenarios produce consistent state trajectories, which are processed by the network simulator and evaluated on the client side.

The following components are implemented and functional:
- **Scenario definitions** for controlled movement and view patterns,
- **Network simulation and evaluation logic** supporting baseline and DR policies,
- **Parameter sweeps** for Dead Reckoning thresholds,
- **Metric logging** to CSV for post-processing and visualization.

In parallel, initial machine learning components are implemented:
- Lightweight **MLP-based predictors** trained on short windows of player state,
- Scripts for training, evaluation, and learning-curve visualization.

These models are currently evaluated offline or in isolation. Full integration into the live network simulation loop is in progress, with emphasis placed first on validating baselines and evaluation methodology.

---

## Repository structure

The repository is organized around three main concerns: scenario specification, experimentation scripts, and evaluation logic.

- `scenarios/` contains ViZDoom configuration files defining player movement and view behavior.
- `scripts/` contains training, evaluation, parameter sweep, and plotting utilities.
- `src/eval/` contains the core network simulation and metric computation code.
- `models/` and `results/` store trained checkpoints and generated outputs during experimentation and may be excluded from version control depending on size.

---

## Running the experiments

The codebase is tested with Python 3.11. After creating and activating a virtual environment, dependencies can be installed via the provided requirements file.

Dead Reckoning evaluations can be run directly using the evaluation scripts by specifying the scenario configuration and network preset. Each run produces CSV files containing both network-level and accuracy-related metrics.

---

## Evaluation metrics

Each experiment records both traffic behavior and reconstruction accuracy.

Network-level metrics include:
- Total packets and bytes sent, delivered, and dropped,
- Simulated one-way delay and maximum queueing delay.

Accuracy is quantified using:
- Mean and maximum positional error,
- Mean and maximum yaw and pitch angular error.

Traffic savings are reported relative to baseline replication, enabling direct comparison between update strategies.

---

## Planned work

The immediate next phase focuses on integrating learned predictors directly into the network simulation loop, allowing controlled comparison between:
- baseline replication,
- Dead Reckoning,
- AI-assisted prediction.

Subsequent work will introduce:
- confidence-based gating to mitigate cold-start behavior,
- corrective packet strategies to bound prediction error,
- periodic synchronization of model parameters between server and clients.

The final stage will map the ViZDoom-based results onto Zandronum’s networking layer to evaluate feasibility under real engine constraints.

---

## Thesis context

This repository supports the master’s thesis *“Reducing Network Traffic in First-Person Shooter Games Using AI-Based Player Behavior Prediction.”* The work builds on established Dead Reckoning literature while exploring the practical integration of lightweight learning models into real-time networked systems.

---

## Author

Murad Abiyev  
Master’s student, Artificial Intelligence  
Supervisor: [Name]
