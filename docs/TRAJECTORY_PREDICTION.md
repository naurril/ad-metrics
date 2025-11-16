# Trajectory Prediction Metrics

This document provides comprehensive coverage of trajectory prediction metrics implemented in `admetrics`.

## Table of Contents

- [Trajectory Prediction Metrics](#trajectory-prediction-metrics)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Core Displacement Metrics](#core-displacement-metrics)
    - [1. ADE (Average Displacement Error)](#1-ade-average-displacement-error)
    - [2. FDE (Final Displacement Error)](#2-fde-final-displacement-error)
    - [3. Miss Rate (MR)](#3-miss-rate-mr)
  - [Multi-Modal Prediction Metrics](#multi-modal-prediction-metrics)
    - [4. minADE (Minimum ADE)](#4-minade-minimum-ade)
    - [5. minFDE (Minimum FDE)](#5-minfde-minimum-fde)
  - [Probabilistic Metrics](#probabilistic-metrics)
    - [6. Brier-FDE](#6-brier-fde)
    - [7. NLL (Negative Log-Likelihood)](#7-nll-negative-log-likelihood)
  - [Safety and Constraint Metrics](#safety-and-constraint-metrics)
    - [8. Collision Rate](#8-collision-rate)
    - [9. Drivable Area Compliance](#9-drivable-area-compliance)
  - [Comprehensive Metrics](#comprehensive-metrics)
    - [10. calculate\_trajectory\_metrics()](#10-calculate_trajectory_metrics)
  - [Metrics Comparison](#metrics-comparison)
  - [Benchmark Standards](#benchmark-standards)
    - [Argoverse Motion Forecasting](#argoverse-motion-forecasting)
    - [nuScenes Prediction](#nuscenes-prediction)
    - [Waymo Open Motion Dataset](#waymo-open-motion-dataset)
  - [Best Practices](#best-practices)
    - [Choosing Metrics](#choosing-metrics)
    - [Common Pitfalls](#common-pitfalls)
  - [Examples](#examples)
  - [Testing](#testing)
  - [References](#references)

## Overview

The trajectory prediction module (`admetrics.trajectory`) provides metrics for evaluating predicted future trajectories of dynamic objects (vehicles, pedestrians, cyclists). These are essential for motion forecasting models in autonomous driving.

## Core Displacement Metrics

### 1. ADE (Average Displacement Error)

**Definition:** Mean Euclidean distance between predicted and ground truth positions across all timesteps.

**Formula:**
```
ADE = (1/T) * Σ ||pred_t - gt_t||₂
```

Where T is the number of timesteps.

**Characteristics:**
- Measures average accuracy over entire trajectory
- Sensitive to errors throughout the prediction horizon
- Most commonly reported metric for trajectory prediction
- Units: meters

**Usage:**
```python
from admetrics.trajectory import calculate_ade

# pred: (T, 2) for 2D or (T, 3) for 3D
# gt: (T, 2) or (T, 3)
ade = calculate_ade(predicted_trajectory, ground_truth_trajectory)
print(f"ADE: {ade:.2f} meters")
```

**When to Use:**
- Comparing overall trajectory quality
- Understanding average prediction accuracy
- Standard metric for most trajectory benchmarks

---

### 2. FDE (Final Displacement Error)

**Definition:** Euclidean distance between final predicted position and final ground truth position.

**Formula:**
```
FDE = ||pred_T - gt_T||₂
```

**Characteristics:**
- Measures only endpoint accuracy
- Critical for planning (where will object end up?)
- Ignores intermediate trajectory quality
- Often more important than ADE for decision-making
- Units: meters

**Usage:**
```python
from admetrics.trajectory import calculate_fde

fde = calculate_fde(predicted_trajectory, ground_truth_trajectory)
print(f"FDE: {fde:.2f} meters")
```

**When to Use:**
- End-state prediction evaluation
- Long-horizon forecasting (3-6 seconds)
- Planning and collision avoidance

---

### 3. Miss Rate (MR)

**Definition:** Binary metric indicating if FDE exceeds a threshold.

**Formula:**
```
MR = 1 if FDE > threshold else 0
```

Common thresholds:
- Vehicles: 2.0 meters
- Pedestrians: 0.5 meters
- Cyclists: 1.0 meters

**Characteristics:**
- Binary success/failure metric
- Useful for safety-critical thresholds
- Aggregated across dataset as percentage
- Threshold depends on object type and application

**Usage:**
```python
from admetrics.trajectory import calculate_miss_rate

result = calculate_miss_rate(pred, gt, threshold=2.0)
print(f"Miss: {result['is_miss']}")  # True/False
print(f"FDE: {result['fde']:.2f}m")
```

**When to Use:**
- Safety-critical applications
- Setting acceptance criteria
- Comparing models at specific thresholds

---

## Multi-Modal Prediction Metrics

Most modern trajectory predictors output multiple possible futures (modes). Multi-modal metrics evaluate these predictions.

### 4. minADE (Minimum ADE)

**Definition:** Best ADE among all predicted modes.

**Formula:**
```
minADE = min_{k ∈ K} ADE(pred_k, gt)
```

**Characteristics:**
- Standard metric for multi-modal benchmarks
- Tests if model can predict correct scenario
- Does not penalize incorrect modes
- Optimistic evaluation (best-case)

**Usage:**
```python
from admetrics.trajectory import calculate_multimodal_ade

# preds: (K, T, 2) - K modes, T timesteps
# gt: (T, 2)
result = calculate_multimodal_ade(multimodal_preds, ground_truth)
print(f"minADE: {result['min_ade']:.2f}m")
print(f"Best mode: {result['best_mode']}")
print(f"Mean ADE: {result['mean_ade']:.2f}m")
```

---

### 5. minFDE (Minimum FDE)

**Definition:** Best FDE among all predicted modes.

**Formula:**
```
minFDE = min_{k ∈ K} FDE(pred_k, gt)
```

**Usage:**
```python
from admetrics.trajectory import calculate_multimodal_fde

result = calculate_multimodal_fde(multimodal_preds, ground_truth)
print(f"minFDE: {result['min_fde']:.2f}m")
```

**Standard Benchmarks:**
- **Argoverse**: Reports minADE₆ and minFDE₆ (6 modes, 3 sec horizon)
- **nuScenes**: Reports minADE₅ and minFDE₅ (5 modes, 6 sec horizon)
- **Waymo Open**: Reports minADE and mAP

---

## Probabilistic Metrics

### 6. Brier-FDE

**Definition:** Probability-weighted FDE across all modes.

**Formula:**
```
Brier-FDE = Σ π_k * FDE(pred_k, gt)
```

Where π_k is the predicted probability of mode k.

**Characteristics:**
- Evaluates probability calibration
- Penalizes confident incorrect predictions
- Proper scoring rule (incentivizes truthful probabilities)
- Requires mode probabilities

**Usage:**
```python
from admetrics.trajectory import calculate_brier_fde

# probabilities: (K,) summing to 1.0
result = calculate_brier_fde(preds, gt, probabilities)
print(f"Brier-FDE: {result['brier_fde']:.2f}m")
```

**When to Use:**
- Evaluating probabilistic forecasters
- Testing confidence calibration
- Comparing prediction uncertainty

---

### 7. NLL (Negative Log-Likelihood)

**Definition:** Negative log probability of ground truth under predicted distribution.

**Formula:**
For Gaussian mixture:
```
NLL = -log(Σ π_k * 𝒩(gt | μ_k, Σ_k))
```

**Characteristics:**
- Proper scoring rule
- Evaluates both accuracy AND uncertainty
- Requires covariance/uncertainty estimates
- Lower is better
- Standard for probabilistic models

**Usage:**
```python
from admetrics.trajectory import calculate_nll

# preds: (K, T, 2) - means
# covs: (K, T, 2, 2) - covariance matrices
# probs: (K,) - mixture weights
result = calculate_nll(preds, gt, covs, probs)
print(f"NLL: {result['nll']:.4f}")
```

**When to Use:**
- Evaluating uncertainty quantification
- Probabilistic deep learning models
- Gaussian mixture predictors

---

## Safety and Constraint Metrics

### 8. Collision Rate

**Definition:** Percentage of timesteps where predicted trajectory collides with obstacles.

**Formula:**
```
Collision Rate = (# collision timesteps) / (total timesteps)
```

**Characteristics:**
- Safety-critical metric
- Should be 0% for deployed systems
- Evaluates physical plausibility
- Requires obstacle information

**Usage:**
```python
from admetrics.trajectory import calculate_collision_rate

obstacles = [
    {'center': [10, 5], 'radius': 1.0},  # Static obstacle
]

result = calculate_collision_rate(pred, obstacles, safety_margin=0.5)
print(f"Collision rate: {result['collision_rate']:.1%}")
print(f"Collisions at: {result['collision_timesteps']}")
```

---

### 9. Drivable Area Compliance

**Definition:** Percentage of timesteps where predicted trajectory stays within drivable area.

**Characteristics:**
- Measures legal/physical feasibility
- Important for realistic predictions
- Supports rectangle and polygon areas
- Should be 100% for deployed systems

**Usage:**
```python
from admetrics.trajectory import calculate_drivable_area_compliance

# Rectangle area
area = {
    'type': 'rectangle',
    'x_min': 0, 'x_max': 50,
    'y_min': -5, 'y_max': 5
}

result = calculate_drivable_area_compliance(pred, area)
print(f"Compliance: {result['compliance_rate']:.1%}")

# Polygon area (lane boundaries)
area = {
    'type': 'polygon',
    'vertices': [[0,0], [50,0], [50,3.5], [0,3.5]]
}
```

---

## Comprehensive Metrics

### 10. calculate_trajectory_metrics()

Convenience function that calculates all relevant metrics at once.

**Usage:**
```python
from admetrics.trajectory import calculate_trajectory_metrics

# Single-modal
metrics = calculate_trajectory_metrics(
    pred, gt,
    miss_threshold=2.0,
    multimodal=False
)
# Returns: {ade, fde, miss_rate, is_miss}

# Multi-modal
metrics = calculate_trajectory_metrics(
    multimodal_preds, gt,
    miss_threshold=2.0,
    multimodal=True
)
# Returns: {ade, fde, miss_rate, mean_ade, mean_fde, num_modes, best_mode}
```

---

## Metrics Comparison

| Metric | Input | Output | Best Value | Use Case |
|--------|-------|--------|------------|----------|
| **ADE** | Single traj | Scalar | 0 | Overall accuracy |
| **FDE** | Single traj | Scalar | 0 | Endpoint accuracy |
| **Miss Rate** | Single traj | 0 or 1 | 0 | Safety threshold |
| **minADE** | K trajs | Scalar | 0 | Multi-modal best |
| **minFDE** | K trajs | Scalar | 0 | Multi-modal endpoint |
| **Brier-FDE** | K trajs + probs | Scalar | 0 | Calibration |
| **NLL** | K trajs + covs + probs | Scalar | -∞ | Uncertainty |
| **Collision Rate** | Traj + obstacles | % | 0% | Safety |
| **Compliance** | Traj + area | % | 100% | Feasibility |

---

## Benchmark Standards

### Argoverse Motion Forecasting
- **Horizon:** 3 seconds (30 timesteps @ 10 Hz)
- **Modes:** 6 trajectories
- **Metrics:** minADE₆, minFDE₆, Miss Rate @ 2m

### nuScenes Prediction
- **Horizon:** 6 seconds (12 timesteps @ 2 Hz)
- **Modes:** 5 trajectories  
- **Metrics:** minADE₅, minFDE₅, Miss Rate @ 2m
- **Additional:** Off-road rate, collision rate

### Waymo Open Motion Dataset
- **Horizon:** 8 seconds (80 timesteps @ 10 Hz)
- **Modes:** 6 trajectories
- **Metrics:** minADE, mAP (mean Average Precision)
- **Additional:** Soft metrics, overlap rate

---

## Best Practices

### Choosing Metrics

1. **For model development:**
   - Primary: ADE/FDE (quick feedback)
   - Secondary: Miss rate (threshold performance)

2. **For benchmark comparison:**
   - Multi-modal: minADE, minFDE
   - Probabilistic: Brier-FDE or NLL

3. **For deployment:**
   - Safety: Collision rate (must be 0%)
   - Feasibility: Drivable area compliance
   - Accuracy: FDE at relevant horizon

### Common Pitfalls

❌ **Don't:**
- Use only minADE/minFDE (ignores mode diversity)
- Ignore safety metrics for deployed systems
- Compare metrics across different horizons
- Mix 2D and 3D metrics

✅ **Do:**
- Report both ADE and FDE
- Include miss rate at application-relevant threshold
- Evaluate safety constraints for real-world deployment
- Specify prediction horizon and frequency
- Use consistent coordinate systems

---

## Examples

See `examples/trajectory_prediction.py` for comprehensive demonstrations:
- Single-modal vs multi-modal predictions
- Probabilistic metrics (Brier-FDE, NLL)
- Safety metrics (collision, compliance)
- Visualization of results

Run examples:
```bash
python examples/trajectory_prediction.py
```

## Testing

Comprehensive test suite in `tests/test_trajectory.py`:
- 26 test cases covering all metrics
- Edge cases (empty trajectories, single timestep)
- Multi-modal scenarios
- Probabilistic predictions
- Safety constraints

Run tests:
```bash
pytest tests/test_trajectory.py -v
```

---

## References

1. **Argoverse: 3D Tracking and Forecasting with Rich Maps**
   - Chang, M.F., Lambert, J., Sangkloy, P., Singh, J., Bak, S., Hartnett, A., Wang, D., Carr, P., Lucey, S., Ramanan, D., & Hays, J. (2019)
   - CVPR 2019
   - https://arxiv.org/abs/1911.02620
   - https://www.argoverse.org/av1.html
   - 300,000+ tracked scenarios with HD maps for motion forecasting

2. **CoverNet: Multimodal Behavior Prediction using Trajectory Sets** (nuScenes Prediction)
   - Phan-Minh, T., Grigore, E.C., Boulton, F.A., Beijbom, O., & Wolff, E.M. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1911.10298
   - https://www.nuscenes.org/prediction
   - Classification-based approach to multimodal trajectory prediction

3. **Large Scale Interactive Motion Forecasting for Autonomous Driving** (Waymo Open Motion)
   - Ettinger, S., Cheng, S., Caine, B., Liu, C., Zhao, H., Pradhan, S., Chai, Y., Sapp, B., Qi, C.R., Zhou, Y., et al. (2021)
   - IROS 2021
   - https://arxiv.org/abs/2104.10133
   - https://waymo.com/open/data/motion
   - Interactive prediction with diverse agent types

4. **TNT: Target-driveN Trajectory Prediction**
   - Zhao, H., Gao, J., Lan, T., Sun, C., Sapp, B., Varadarajan, B., Shen, Y., Shen, Y., Chai, Y., Schmid, C., et al. (2020)
   - CoRL 2020
   - https://arxiv.org/abs/2008.08294
   - Target-conditioned multimodal prediction

5. **Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data**
   - Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020)
   - ECCV 2020
   - https://arxiv.org/abs/2001.03093
   - https://github.com/StanfordASL/Trajectron-plus-plus
   - Graph-structured model with dynamic constraints and map integration
