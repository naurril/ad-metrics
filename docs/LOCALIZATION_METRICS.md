# Localization Metrics for Autonomous Driving

Comprehensive guide to ego vehicle localization metrics for evaluating pose estimation systems in autonomous driving.

## Table of Contents

- [Overview](#overview)
- [Position Accuracy Metrics](#position-accuracy-metrics)
- [Orientation Accuracy Metrics](#orientation-accuracy-metrics)
- [Convergence and Initialization Metrics](#convergence-and-initialization-metrics)
- [HD Map Alignment Metrics](#hd-map-alignment-metrics)
- [Use Cases](#use-cases)
- [Implementation Examples](#implementation-examples)
- [Benchmark Compatibility](#benchmark-compatibility)

---

## Overview

Localization metrics evaluate the accuracy of ego vehicle pose estimation systems that combine multiple sensors (GPS, IMU, LiDAR, cameras) and HD maps to determine the vehicle's position and orientation. These metrics are critical for:

- **GPS Systems**: Consumer-grade GPS, RTK-GPS, differential GPS
- **SLAM Systems**: Visual SLAM, LiDAR SLAM, visual-inertial odometry
- **Sensor Fusion**: GPS+IMU+LiDAR integration pipelines
- **HD Map Localization**: Lane-level positioning accuracy
- **Modular Autonomous Driving**: Evaluating localization components independently

### Key Concepts

**Pose Representations:**
- **3D Pose**: `(x, y, z)` - Position only
- **4D Pose**: `(x, y, z, yaw)` - Position + heading angle
- **7D Pose**: `(x, y, z, qw, qx, qy, qz)` - Position + quaternion orientation

**Coordinate Systems:**
- Global coordinates (e.g., WGS84, UTM)
- Local coordinates (e.g., relative to start position)
- Map coordinates (HD map reference frame)

---

## Position Accuracy Metrics

### Absolute Trajectory Error (ATE)

**Definition:** Mean Euclidean distance between predicted and ground truth positions across the entire trajectory.

**Formula:**
```
ATE = (1/N) * Σ ||p_pred[i] - p_gt[i]||₂
```

where:
- `N` = number of poses
- `p_pred[i]` = predicted position at timestamp i
- `p_gt[i]` = ground truth position at timestamp i

**Key Statistics:**
- `mean`: Average position error (meters)
- `std`: Standard deviation of error
- `max`: Maximum position error
- `rmse`: Root mean square error
- `median`: Median error (robust to outliers)

**Interpretation:**
- **ATE < 0.1m**: Excellent (RTK-GPS level)
- **ATE < 0.5m**: Good (high-quality SLAM)
- **ATE < 2.0m**: Acceptable (consumer GPS)
- **ATE > 5.0m**: Poor localization

**Use Cases:**
- GPS accuracy evaluation
- SLAM performance assessment
- Sensor fusion validation
- Comparing localization methods

**Alignment Option:**
When `align=True`, uses the Umeyama algorithm to find optimal similarity transformation (rotation, translation, scale) between trajectories. This is useful for:
- Evaluating SLAM systems where absolute position is arbitrary
- Removing initialization bias
- Focusing on relative accuracy rather than absolute accuracy

**Example:**
```python
from admetrics.localization import calculate_ate

# Evaluate GPS localization (no alignment)
ate_result = calculate_ate(
    predicted_poses,  # (N, 3) or (N, 7)
    ground_truth_poses,
    align=False
)
print(f"GPS ATE: {ate_result['mean']:.3f}m ± {ate_result['std']:.3f}m")

# Evaluate SLAM (with alignment)
ate_slam = calculate_ate(slam_poses, gt_poses, align=True)
print(f"SLAM ATE: {ate_slam['mean']:.3f}m (after alignment)")
```

---

### Relative Trajectory Error (RTE)

**Definition:** Measures drift over specific distances by comparing relative transformations between pose pairs.

**Formula:**
For each distance `d`:
```
RTE(d) = (1/M) * Σ ||Δp_pred - Δp_gt|| / d
```

where:
- `Δp_pred = p_pred[j] - p_pred[i]` (relative motion in prediction)
- `Δp_gt = p_gt[j] - p_gt[i]` (relative motion in ground truth)
- `d` = traveled distance from i to j
- Result is normalized by distance (percentage error)

**Typical Distances:**
- 10m: Short-term accuracy
- 50m: Medium-term drift
- 100m: Long-term drift (KITTI standard)
- 200m, 500m: Very long-term drift

**Interpretation:**
- **RTE < 1%**: Excellent drift characteristics
- **RTE < 5%**: Good (typical SLAM)
- **RTE < 10%**: Acceptable
- **RTE > 20%**: Significant drift problem

**Use Cases:**
- Detecting SLAM drift over time
- Evaluating loop closure effectiveness
- Comparing odometry methods
- KITTI odometry benchmark

**Example:**
```python
from admetrics.localization import calculate_rte

rte_result = calculate_rte(
    predicted_poses[:, :3],
    ground_truth_poses[:, :3],
    distances=[10, 50, 100, 200]
)

print(f"RTE @ 10m:  {rte_result['rte_10']:.4f} ({rte_result['rte_10']*100:.2f}%)")
print(f"RTE @ 100m: {rte_result['rte_100']:.4f} ({rte_result['rte_100']*100:.2f}%)")

# Detect drift
if rte_result['rte_100'] > 0.05:
    print("Warning: Significant drift detected")
```

---

### Lateral Error (Cross-Track Error)

**Definition:** Perpendicular distance from the predicted path to the ground truth path.

**Formula:**
For each position:
```
lateral_error[i] = ||offset - (offset · direction) * direction||₂
```

where:
- `offset = p_pred[i] - p_gt[i]`
- `direction = normalized tangent vector of ground truth path`

**Key Statistics:**
- `mean`: Average cross-track error
- `std`: Variability in lateral positioning
- `max`: Maximum deviation
- `lane_violation_rate`: Percentage of positions outside lane width/2
- `lane_violations`: Count of violations

**Interpretation:**
- **Lateral < 0.1m**: Excellent lane keeping
- **Lateral < 0.5m**: Good (within half lane width)
- **Lateral < 1.0m**: Acceptable
- **Lateral > 1.75m**: Lane violation (for 3.5m lane width)

**Use Cases:**
- Lane keeping assessment
- HD map-based localization
- Autonomous driving safety validation
- Path following accuracy

**Example:**
```python
from admetrics.localization import calculate_lateral_error

lateral_result = calculate_lateral_error(
    predicted_poses,
    ground_truth_poses,
    lane_width=3.5  # meters
)

print(f"Lateral Error: {lateral_result['mean']:.3f}m ± {lateral_result['std']:.3f}m")
print(f"Lane Violations: {lateral_result['lane_violation_rate']:.1%}")
print(f"Violation Count: {lateral_result['lane_violations']}")
```

---

### Longitudinal Error (Along-Track Error)

**Definition:** Distance ahead or behind the ground truth position along the path direction.

**Formula:**
```
longitudinal_error[i] = offset · direction
```

where:
- `offset = p_pred[i] - p_gt[i]`
- `direction = normalized tangent vector of ground truth path
- Positive = ahead of ground truth
- Negative = behind ground truth

**Key Statistics:**
- `mean`: Average along-track error (can be positive/negative)
- `std`: Variability
- `mean_abs`: Average absolute error
- `rmse`: Root mean square error
- `median`: Median error

**Use Cases:**
- Velocity estimation validation
- Time synchronization verification
- Along-path positioning accuracy

**Example:**
```python
from admetrics.localization import calculate_longitudinal_error

long_result = calculate_longitudinal_error(predicted_poses, ground_truth_poses)

print(f"Longitudinal Error: {long_result['mean']:.3f}m")
if long_result['mean'] > 0:
    print("System is ahead of ground truth")
else:
    print("System is behind ground truth")
```

---

## Orientation Accuracy Metrics

### Absolute Rotation Error (ARE)

**Definition:** Angular difference between predicted and ground truth heading angles.

**Supports:**
- **Yaw angle**: 4D poses `(x, y, z, yaw)`
- **Quaternions**: 7D poses `(x, y, z, qw, qx, qy, qz)`

**Formula:**
```
ARE = (1/N) * Σ |normalize_angle(yaw_pred[i] - yaw_gt[i])|
```

For quaternions, yaw is extracted using:
```
yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))
```

**Key Statistics:**
- `mean`: Average heading error (degrees)
- `std`: Standard deviation
- `max`: Maximum orientation error
- `rmse`: Root mean square error
- `median`: Median error

**Interpretation:**
- **ARE < 1°**: Excellent orientation
- **ARE < 5°**: Good (typical IMU quality)
- **ARE < 10°**: Acceptable
- **ARE > 20°**: Poor heading estimation

**Use Cases:**
- IMU calibration validation
- Heading sensor evaluation
- Orientation estimation quality
- Sensor fusion assessment

**Example:**
```python
from admetrics.localization import calculate_are

# With yaw angles (4D poses)
are_result = calculate_are(
    predicted_poses_4d,  # (N, 4): [x, y, z, yaw]
    ground_truth_poses_4d
)

# With quaternions (7D poses)
are_quat = calculate_are(
    predicted_poses_7d,  # (N, 7): [x, y, z, qw, qx, qy, qz]
    ground_truth_poses_7d
)

print(f"Heading Error: {are_result['mean']:.2f}° ± {are_result['std']:.2f}°")
print(f"Max Error: {are_result['max']:.2f}°")
```

---

## Convergence and Initialization Metrics

### Convergence Rate

**Definition:** Measures how quickly the localization system converges to accurate estimates after initialization or recovery from loss.

**Formula:**
```
time_to_convergence = first timestamp where error < threshold
convergence_rate = (initial_error - converged_error) / time_to_convergence
```

**Parameters:**
- `threshold`: Convergence threshold in meters (e.g., 0.5m)
- `timestamps`: Time series data

**Key Statistics:**
- `converged`: Boolean - whether system converged
- `time_to_convergence`: Time in seconds to reach threshold
- `convergence_rate`: Error reduction per second (m/s)
- `initial_error`: Starting position error
- `converged_error`: Error at convergence time

**Interpretation:**
- **Time < 1s**: Fast convergence (good initialization)
- **Time < 5s**: Acceptable convergence
- **Time > 10s**: Slow convergence (may need improvement)
- **Not converged**: System never reaches acceptable accuracy

**Use Cases:**
- Initialization quality assessment
- Recovery from GPS loss
- Tunnel exit performance
- System startup behavior

**Example:**
```python
from admetrics.localization import calculate_convergence_rate
import numpy as np

# Calculate position errors over time
position_errors = np.linalg.norm(
    predicted_poses[:, :3] - ground_truth_poses[:, :3], 
    axis=1
)

convergence = calculate_convergence_rate(
    position_errors,
    timestamps=timestamps,
    threshold=0.5  # 0.5m threshold
)

if convergence['converged']:
    print(f"Converged in {convergence['time_to_convergence']:.2f}s")
    print(f"Initial error: {convergence['initial_error']:.2f}m")
    print(f"Converged error: {convergence['converged_error']:.2f}m")
    print(f"Convergence rate: {convergence['convergence_rate']:.2f} m/s")
else:
    print("System did not converge within trajectory")
```

---

## HD Map Alignment Metrics

### Map Alignment Score

**Definition:** Measures how well the localization aligns with HD map features, particularly lane centerlines.

**Formula:**
```
For each position:
  min_distance[i] = min(distance to all lane centerlines)

alignment_rate = count(min_distance < threshold) / N
mean_distance = (1/N) * Σ min_distance
```

**Parameters:**
- `hd_map_lanes`: List of lane centerlines, each (M, 2) or (M, 3)
- `max_distance`: Threshold for considering alignment (default: 2.0m)

**Key Statistics:**
- `mean_distance_to_lane`: Average distance to nearest lane
- `median_distance_to_lane`: Median distance
- `max_distance_to_lane`: Maximum distance
- `alignment_rate`: Percentage within max_distance
- `aligned_poses`: Count of aligned positions
- `total_poses`: Total number of positions

**Interpretation:**
- **Mean < 0.3m**: Excellent map alignment
- **Mean < 0.5m**: Good lane-level accuracy
- **Mean < 1.0m**: Acceptable
- **Mean > 2.0m**: Poor map alignment

**Use Cases:**
- HD map-based localization validation
- Lane-level positioning accuracy
- Map matching quality
- Production deployment readiness

**Example:**
```python
from admetrics.localization import calculate_map_alignment_score

# Define HD map lane centerlines
lane_centerlines = [
    np.array([[x1, y1], [x2, y2], ...]),  # Lane 1
    np.array([[x1, y1], [x2, y2], ...]),  # Lane 2
    # ... more lanes
]

alignment = calculate_map_alignment_score(
    predicted_poses[:, :2],  # Use only x, y
    lane_centerlines,
    max_distance=2.0
)

print(f"Mean distance to lane: {alignment['mean_distance_to_lane']:.3f}m")
print(f"Alignment rate: {alignment['alignment_rate']:.1%}")
print(f"Aligned poses: {alignment['aligned_poses']}/{alignment['total_poses']}")
```

---

## Use Cases

### 1. GPS System Evaluation

**Consumer-grade GPS:**
```python
metrics = calculate_localization_metrics(
    gps_poses, ground_truth_poses,
    align=False  # Absolute positioning
)
# Expected: ATE ~1-2m, lateral ~0.5m
```

**RTK-GPS:**
```python
metrics = calculate_localization_metrics(
    rtk_poses, ground_truth_poses,
    align=False
)
# Expected: ATE ~0.02m, lateral ~0.01m
```

### 2. SLAM Drift Analysis

```python
# Use RTE to detect drift
rte = calculate_rte(slam_poses, gt_poses, distances=[100, 200, 500])

# Use ATE with alignment for relative accuracy
ate_aligned = calculate_ate(slam_poses, gt_poses, align=True)
```

### 3. Sensor Fusion Validation

```python
# Compare different sensor combinations
gps_only = calculate_localization_metrics(gps_poses, gt_poses)
gps_imu = calculate_localization_metrics(fusion_poses, gt_poses)

improvement = (gps_only['ate_mean'] - gps_imu['ate_mean']) / gps_only['ate_mean']
print(f"Fusion improves ATE by {improvement*100:.1f}%")
```

### 4. HD Map-Based Localization

```python
# Evaluate lane-level accuracy
lateral = calculate_lateral_error(poses, gt_poses, lane_width=3.5)
map_align = calculate_map_alignment_score(poses[:, :2], hd_map_lanes)

# Check production readiness
if lateral['lane_violation_rate'] < 0.01 and map_align['mean_distance_to_lane'] < 0.3:
    print("✓ Ready for lane-level autonomous driving")
```

### 5. Initialization Performance

```python
# Test cold start behavior
convergence = calculate_convergence_rate(errors, timestamps, threshold=0.5)

if convergence['time_to_convergence'] < 2.0:
    print("✓ Fast initialization suitable for production")
```

---

## Implementation Examples

### Complete Evaluation Pipeline

```python
from admetrics.localization import calculate_localization_metrics

# Load data
predicted_poses = load_poses('predicted.txt')  # (N, 7)
ground_truth_poses = load_poses('ground_truth.txt')
timestamps = load_timestamps('timestamps.txt')

# Comprehensive evaluation
results = calculate_localization_metrics(
    predicted_poses,
    ground_truth_poses,
    timestamps=timestamps,
    lane_width=3.5,
    align=False
)

# Print report
print("=" * 60)
print("Localization Evaluation Report")
print("=" * 60)
print(f"Position Accuracy:")
print(f"  ATE: {results['ate_mean']:.3f} ± {results['ate_std']:.3f} m")
print(f"  Max Error: {results['ate_max']:.3f} m")
print()
print(f"Path Following:")
print(f"  Lateral Error: {results['lateral_mean']:.3f} ± {results['lateral_std']:.3f} m")
print(f"  Lane Violations: {results['lateral_lane_violation_rate']:.1%}")
print(f"  Longitudinal Error: {results['longitudinal_mean']:.3f} m")
print()
print(f"Orientation:")
print(f"  Heading Error: {results['are_mean']:.2f}° ± {results['are_std']:.2f}°")
print()
if 'convergence_converged' in results:
    print(f"Initialization:")
    if results['convergence_converged']:
        print(f"  Converged: Yes ({results['convergence_time_to_convergence']:.2f}s)")
        print(f"  Convergence Rate: {results['convergence_convergence_rate']:.2f} m/s")
    else:
        print(f"  Converged: No")
```

### Comparing Localization Methods

```python
methods = {
    'GPS': gps_poses,
    'RTK-GPS': rtk_poses,
    'SLAM': slam_poses,
    'GPS+IMU+SLAM': fusion_poses
}

print(f"{'Method':<20} {'ATE (m)':<12} {'Lateral (m)':<12} {'ARE (°)':<12}")
print("-" * 60)

for name, poses in methods.items():
    metrics = calculate_localization_metrics(poses, gt_poses, align=False)
    print(f"{name:<20} {metrics['ate_mean']:<12.4f} "
          f"{metrics['lateral_mean']:<12.4f} {metrics['are_mean']:<12.2f}")
```

---

## Benchmark Compatibility

### KITTI Odometry Benchmark

The KITTI odometry benchmark uses:
- **RTE**: At 100m, 200m, ..., 800m intervals
- **ATE**: After sequence alignment
- **Evaluation**: Per sequence (00-21)

```python
# KITTI-style evaluation
rte = calculate_rte(poses, gt_poses, distances=[100, 200, 300, 400, 500, 600, 700, 800])
ate = calculate_ate(poses, gt_poses, align=True)

# Report KITTI metrics
print(f"Translation Error: {rte['rte_100']:.2%}")
print(f"Rotation Error: {are['mean']:.2f}°/100m")
```

### nuScenes Localization

nuScenes provides ego poses for evaluating localization:

```python
# Load nuScenes ego poses
ego_poses = load_nuscenes_ego_poses(scene_token)
gt_poses = load_nuscenes_gt_poses(scene_token)

# Evaluate
metrics = calculate_localization_metrics(ego_poses, gt_poses)
```

### Argoverse HD Maps

Argoverse provides HD maps for map-based localization:

```python
# Load Argoverse HD map
lane_centerlines = load_argoverse_lanes(log_id)

# Evaluate map alignment
alignment = calculate_map_alignment_score(poses[:, :2], lane_centerlines)
```

---

## Best Practices

### 1. Choose Appropriate Metrics

- **GPS evaluation**: Use ATE without alignment
- **SLAM evaluation**: Use RTE and ATE with alignment
- **Production systems**: Use lateral error and map alignment
- **Initialization**: Use convergence rate

### 2. Use Appropriate Ground Truth

- **RTK-GPS**: ±2cm accuracy for position
- **Ground truth SLAM**: From offline mapping
- **Simulation**: Perfect ground truth available
- **Benchmark datasets**: Use provided ground truth

### 3. Consider Coordinate Systems

- Ensure predicted and ground truth use same coordinate frame
- Account for different height references (WGS84 vs. MSL)
- Apply proper transformations if needed

### 4. Report Multiple Metrics

Don't rely on a single metric:
- Position: ATE, RTE, lateral, longitudinal
- Orientation: ARE
- System behavior: Convergence, map alignment

### 5. Visualize Results

```python
import matplotlib.pyplot as plt

# Plot trajectory
plt.figure(figsize=(12, 6))
plt.plot(gt_poses[:, 0], gt_poses[:, 1], 'g-', label='Ground Truth')
plt.plot(pred_poses[:, 0], pred_poses[:, 1], 'b--', label='Predicted')
plt.legend()
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectory Comparison')
plt.grid(True)
plt.show()

# Plot error over time
plt.figure()
errors = np.linalg.norm(pred_poses[:, :3] - gt_poses[:, :3], axis=1)
plt.plot(timestamps, errors)
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Localization Error Over Time')
plt.grid(True)
plt.show()
```

---

## Summary

Localization metrics provide comprehensive evaluation of ego vehicle pose estimation:

| Metric Category | Key Metrics | Primary Use Case |
|----------------|-------------|------------------|
| **Position** | ATE, RTE | Overall accuracy, drift detection |
| **Path Following** | Lateral, Longitudinal | Lane keeping, HD map alignment |
| **Orientation** | ARE | Heading accuracy |
| **Initialization** | Convergence Rate | System startup, recovery |
| **Map Alignment** | Map Alignment Score | Lane-level positioning |

**For more information:**
- Implementation: `admetrics/localization.py`
- Tests: `tests/test_localization.py`
- Example: `examples/localization_evaluation.py`
- Complete reference: `docs/METRICS_REFERENCE.md`

---

*Last Updated: November 14, 2025*
