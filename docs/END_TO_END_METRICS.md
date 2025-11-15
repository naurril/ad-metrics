# End-to-End Autonomous Driving Metrics

This document describes metrics for evaluating end-to-end autonomous driving models that directly map sensor inputs to driving actions (steering, acceleration, braking) or planned trajectories.

## Table of Contents

1. [Overview](#overview)
2. [Planning Accuracy Metrics](#planning-accuracy-metrics)
3. [Safety Metrics](#safety-metrics)
4. [Progress and Navigation Metrics](#progress-and-navigation-metrics)
5. [Control Accuracy Metrics](#control-accuracy-metrics)
6. [Comfort Metrics](#comfort-metrics)
7. [Composite Metrics](#composite-metrics)
8. [Imitation Learning Metrics](#imitation-learning-metrics)
9. [Benchmark References](#benchmark-references)
10. [Best Practices](#best-practices)

## Overview

End-to-end autonomous driving represents a paradigm shift from modular perception-planning-control pipelines to unified models that directly output driving actions. Examples include:

- **Tesla FSD**: Vision-based neural planner
- **Wayve**: End-to-end learned driving
- **nuPlan**: Closed-loop planning benchmark
- **CARLA**: Simulation-based evaluation

These metrics evaluate the quality, safety, and efficiency of end-to-end driving policies.

### Key Evaluation Dimensions

| Dimension | What It Measures | Key Metrics |
|-----------|------------------|-------------|
| **Planning Accuracy** | How well trajectories match expert/optimal paths | L2 distance, ADE, FDE |
| **Safety** | Collision avoidance and safe operation | Collision rate, time-to-collision |
| **Progress** | Task completion and navigation efficiency | Progress score, route completion |
| **Control** | Accuracy of path following and speed control | Lateral deviation, heading error, velocity error |
| **Comfort** | Smooth driving and passenger experience | Acceleration, jerk, comfort rate |
| **Composite** | Overall driving performance | Driving score (nuPlan/CARLA style) |

## Planning Accuracy Metrics

### L2 Distance

**Purpose**: Primary metric for trajectory matching quality.

**Formula**:
```
L2 = (1/N) * Σ w_i * ||p_i - g_i||_2
```
where:
- `p_i`: predicted waypoint at timestep i
- `g_i`: ground truth/expert waypoint at timestep i
- `w_i`: optional temporal weight (default: uniform)
- `N`: number of waypoints

**Usage**:
```python
from admetrics.planning import l2_distance

# Basic usage
dist = l2_distance(predicted_traj, expert_traj)

# Emphasize long-term planning (later timesteps weighted more)
weights = np.linspace(1.0, 2.0, len(predicted_traj))
dist_weighted = l2_distance(predicted_traj, expert_traj, weights=weights)
```

**Interpretation**:
- Lower is better
- Typical good performance: < 0.5m for urban driving
- Used in nuPlan, CARLA benchmarks

**Temporal Weighting Strategies**:
- **Uniform** (default): All timesteps equally important
- **Linear increasing**: Emphasize long-term planning accuracy
- **Exponential**: Strong emphasis on distant future
- **Custom**: Task-specific (e.g., emphasize near-goal states)

### Average Displacement Error (ADE) & Final Displacement Error (FDE)

**Purpose**: Standard metrics for trajectory prediction adapted to planning evaluation.

**Formulas**:
```
ADE = (1/T) * Σ w_t * ||p_t - g_t||_2
FDE = ||p_T - g_T||_2
```

**Usage**:
```python
from admetrics.planning import average_displacement_error_planning

# Uniform weighting
result = average_displacement_error_planning(pred_traj, expert_traj)

# Linear weighting (emphasize later timesteps)
result = average_displacement_error_planning(
    pred_traj, expert_traj, 
    timestep_weights='linear'
)

# Exponential weighting (strong emphasis on future)
result = average_displacement_error_planning(
    pred_traj, expert_traj,
    timestep_weights='exponential',
    alpha=0.1
)

print(f"ADE: {result['ADE']:.3f}m")
print(f"FDE: {result['FDE']:.3f}m")
```

**Parameters**:
- `timestep_weights`: 'uniform', 'linear', 'exponential', or array
- `alpha`: Exponential decay rate (for exponential weighting)

**Interpretation**:
- **ADE**: Overall trajectory accuracy
- **FDE**: Long-term planning capability
- Lower values indicate better performance
- FDE often more critical for goal-reaching tasks

## Safety Metrics

### Collision Rate

**Purpose**: Evaluate collision avoidance with static and dynamic obstacles.

**Formula**:
```
collision_rate = (# timesteps with collision) / (total timesteps)
```

**Usage**:
```python
from admetrics.planning import collision_rate

# Define obstacles
static_obstacles = [
    np.array([[5, 2]]),  # Parked car
    np.array([[10, -1]])  # Road barrier
]

dynamic_obstacles = [
    np.array([[8, 5], [8, 3], [8, 1]])  # Crossing pedestrian
]

obstacles = static_obstacles + dynamic_obstacles

result = collision_rate(
    ego_trajectory,
    obstacles,
    vehicle_size=(4.5, 2.0),  # length x width in meters
    safety_margin=0.5,  # additional safety buffer
    obstacle_size=(2.0, 2.0)  # default size for obstacles
)

print(f"Collision rate: {result['collision_rate']:.2%}")
print(f"Number of collisions: {result['num_collisions']}")
print(f"Collision timesteps: {result['collision_timesteps']}")
print(f"First collision: {result['first_collision']}")
```

**Parameters**:
- `vehicle_size`: (length, width) of ego vehicle
- `safety_margin`: Additional buffer around vehicle
- `obstacle_size`: Default size for obstacles

**Collision Detection**:
- Uses circular bounding approximation for efficiency
- Collision radius = `sqrt((L/2)^2 + (W/2)^2) + margin`
- Supports both static and dynamic obstacles

**Interpretation**:
- **Critical**: Should be 0.0% for safe operation
- Used as hard constraint in nuPlan (no collisions allowed)
- CARLA uses weighted collision penalty

## Progress and Navigation Metrics

### Progress Score

**Purpose**: Measure task completion along reference path.

**Usage**:
```python
from admetrics.planning import progress_score

result = progress_score(
    executed_trajectory,
    reference_path,
    goal_threshold=2.0  # meters
)

print(f"Progress: {result['progress']:.2f}m")
print(f"Progress ratio: {result['progress_ratio']:.2%}")
print(f"Goal reached: {result['goal_reached']}")
print(f"Distance to goal: {result['final_distance_to_goal']:.2f}m")
```

**Returns**:
- `progress`: Distance traveled along reference path (meters)
- `progress_ratio`: Fraction of path completed [0, 1]
- `goal_reached`: Boolean indicating goal achievement
- `final_distance_to_goal`: Euclidean distance to final goal

**Interpretation**:
- Higher progress ratio is better
- Critical for task completion evaluation
- Used in nuPlan progress score calculation

### Route Completion

**Purpose**: Evaluate waypoint-based navigation.

**Usage**:
```python
from admetrics.planning import route_completion

# Define waypoints (e.g., intersections, landmarks)
waypoints = np.array([
    [100, 50],   # Waypoint 1
    [200, 150],  # Waypoint 2
    [300, 200]   # Waypoint 3
])

result = route_completion(
    executed_trajectory,
    waypoints,
    completion_radius=3.0  # meters
)

print(f"Waypoints reached: {result['num_waypoints_reached']}/{result['total_waypoints']}")
print(f"Completion rate: {result['completion_rate']:.2%}")
print(f"Waypoint status: {result['waypoint_status']}")
```

**Waypoint Completion Logic**:
- Waypoint is "reached" if vehicle passes within `completion_radius`
- Must visit waypoints in order (cannot skip)
- Returns boolean status for each waypoint

**Interpretation**:
- 100% completion rate indicates successful navigation
- Used in CARLA route completion metric
- Critical for goal-directed driving scenarios

## Control Accuracy Metrics

### Lateral Deviation

**Purpose**: Measure cross-track error from reference path (lane keeping).

**Usage**:
```python
from admetrics.planning import lateral_deviation

result = lateral_deviation(
    actual_trajectory,
    reference_centerline,
    lane_width=3.5  # meters (default)
)

print(f"Mean lateral error: {result['mean_lateral_error']:.3f}m")
print(f"Max lateral error: {result['max_lateral_error']:.3f}m")
print(f"Std lateral error: {result['std_lateral_error']:.3f}m")
print(f"Lane keeping rate: {result['lane_keeping_rate']:.2%}")
```

**Calculation**:
- Projects each trajectory point onto nearest reference segment
- Computes perpendicular distance (cross-track error)
- Lane keeping: fraction of points within ±lane_width/2

**Interpretation**:
- Lower error indicates better path following
- Lane keeping rate should be close to 100%
- Critical for highway driving evaluation

### Heading Error

**Purpose**: Measure orientation/yaw accuracy.

**Usage**:
```python
from admetrics.planning import heading_error

# Headings in radians
predicted_headings = np.array([0.0, 0.1, 0.2, 0.3])  # rad
expert_headings = np.array([0.0, 0.12, 0.19, 0.31])  # rad

result = heading_error(predicted_headings, expert_headings)

print(f"Mean error: {result['mean_heading_error']:.4f} rad")
print(f"Mean error: {result['mean_heading_error_deg']:.2f}°")
print(f"Max error: {result['max_heading_error']:.4f} rad")
```

**Implementation Notes**:
- Properly handles angle wraparound ([-π, π])
- Uses `atan2` for correct quadrant calculation
- Returns both radians and degrees

**Interpretation**:
- Lower error indicates better orientation control
- Important for parallel parking, tight turns
- Typical good performance: < 5° mean error

### Velocity Error

**Purpose**: Measure speed control accuracy.

**Usage**:
```python
from admetrics.planning import velocity_error

result = velocity_error(predicted_velocities, expert_velocities)

print(f"Mean error: {result['mean_velocity_error']:.3f} m/s")
print(f"RMSE: {result['rmse_velocity']:.3f} m/s")
print(f"Max error: {result['max_velocity_error']:.3f} m/s")
```

**Interpretation**:
- Lower error indicates better speed tracking
- Important for traffic flow matching
- Typical good performance: < 1.0 m/s mean error

## Comfort Metrics

### Comfort Metrics (Acceleration & Jerk)

**Purpose**: Evaluate smoothness and passenger comfort.

**Usage**:
```python
from admetrics.planning import comfort_metrics

result = comfort_metrics(
    trajectory,
    timestamps,
    max_acceleration=2.0,  # m/s² threshold
    max_jerk=2.0  # m/s³ threshold
)

print(f"Mean acceleration: {result['mean_acceleration']:.3f} m/s²")
print(f"Max acceleration: {result['max_acceleration']:.3f} m/s²")
print(f"Mean jerk: {result['mean_jerk']:.3f} m/s³")
print(f"Max jerk: {result['max_jerk']:.3f} m/s³")
print(f"Comfort rate: {result['comfort_rate']:.2%}")
print(f"Violations: {result['comfort_violations']}")
```

**Calculation**:
- **Acceleration**: Computed from velocity profile
- **Jerk**: Derivative of acceleration (rate of change)
- **Comfort violations**: Count of timesteps exceeding thresholds

**Returns**:
- `mean_acceleration`: Average magnitude
- `max_acceleration`: Peak value
- `mean_jerk`: Average jerk
- `max_jerk`: Peak jerk
- `comfort_rate`: Fraction of timesteps within thresholds
- `comfort_violations`: Number of threshold violations

**Typical Thresholds**:
- **Comfortable**: acceleration < 2.0 m/s², jerk < 2.0 m/s³
- **Acceptable**: acceleration < 3.0 m/s², jerk < 3.0 m/s³
- **Aggressive**: acceleration > 4.0 m/s² or jerk > 4.0 m/s³

**Interpretation**:
- Higher comfort rate is better
- Used in nuPlan comfort score
- Critical for passenger acceptance

## Composite Metrics

### Driving Score

**Purpose**: Comprehensive evaluation combining planning, safety, progress, and comfort.

**Formula** (default weights):
```
Driving Score = 0.3 * Planning + 0.4 * Safety + 0.2 * Progress + 0.1 * Comfort
```

**Usage**:
```python
from admetrics.planning import driving_score

result = driving_score(
    predicted_trajectory,
    expert_trajectory,
    obstacles,
    reference_path,
    timestamps,
    weights={
        'planning': 0.3,
        'safety': 0.4,
        'progress': 0.2,
        'comfort': 0.1
    }
)

print(f"Driving Score: {result['driving_score']:.1f}/100")
print(f"\nComponent Scores:")
print(f"  Planning: {result['planning_accuracy']:.1f}/100")
print(f"  Safety: {result['safety_score']:.1f}/100")
print(f"  Progress: {result['progress_score']:.1f}/100")
print(f"  Comfort: {result['comfort_score']:.1f}/100")
```

**Component Calculations**:
1. **Planning Accuracy**: Based on L2 distance (normalized)
2. **Safety Score**: 100 if no collisions, 0 otherwise
3. **Progress Score**: Based on progress ratio × 100
4. **Comfort Score**: Based on comfort rate × 100

**Custom Weighting Examples**:

```python
# Safety-critical applications (autonomous shuttles)
safety_focused = {
    'planning': 0.2,
    'safety': 0.6,
    'progress': 0.1,
    'comfort': 0.1
}

# Efficiency-focused (delivery robots)
efficiency_focused = {
    'planning': 0.2,
    'safety': 0.3,
    'progress': 0.4,
    'comfort': 0.1
}

# Comfort-focused (passenger vehicles)
comfort_focused = {
    'planning': 0.25,
    'safety': 0.35,
    'progress': 0.15,
    'comfort': 0.25
}
```

**Interpretation**:
- Score range: 0-100 (higher is better)
- Inspired by nuPlan and CARLA benchmarks
- Adjust weights based on application requirements

## Imitation Learning Metrics

### Planning KL Divergence

**Purpose**: Measure how well learned policy matches expert behavior.

**Formula**:
```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```
where P is expert distribution, Q is learned policy distribution.

**Usage**:
```python
from admetrics.planning import planning_kl_divergence

# Expert action distribution (e.g., steering histogram)
expert_actions = np.array([0.1, 0.3, 0.4, 0.15, 0.05])

# Learned policy distribution
learned_actions = np.array([0.12, 0.28, 0.38, 0.16, 0.06])

kl_div = planning_kl_divergence(learned_actions, expert_actions)

print(f"KL divergence: {kl_div:.4f}")
```

**Application Contexts**:
- Behavioral cloning evaluation
- Policy distillation quality
- Expert demonstration matching
- Action distribution analysis

**Interpretation**:
- Lower KL divergence = better policy match
- KL = 0 means perfect match
- Sensitive to distribution mismatches
- Not symmetric: KL(P||Q) ≠ KL(Q||P)

## Benchmark References

### nuPlan

**Website**: https://www.nuscenes.org/nuplan

**Key Metrics**:
- **Closed-loop score**: Composite metric (0-100)
  - No collisions (pass/fail)
  - Progress (% of route completed)
  - Driving within lane (% of time)
  - Comfort (smooth acceleration/jerk)
- **Open-loop score**: L2 distance to expert

**Evaluation Protocol**:
- 14 scenario types (lane following, turning, parking, etc.)
- 40 scenarios per type
- Pass threshold: > 60% overall score

**Our Implementation**:
```python
# nuPlan-style evaluation
score = driving_score(
    pred_traj, expert_traj, obstacles, ref_path, timestamps,
    weights={'planning': 0.25, 'safety': 0.40, 'progress': 0.20, 'comfort': 0.15}
)
```

### CARLA

**Website**: https://leaderboard.carla.org/

**Key Metrics**:
- **Route completion**: % of route successfully completed
- **Infraction score**: Penalty-based system
  - Collision with pedestrian: -0.50
  - Collision with vehicle: -0.40
  - Collision with static: -0.25
  - Red light violation: -0.30
  - Stop sign violation: -0.20

**Driving Score Formula**:
```
DS = Route Completion × Π (1 - infraction_penalty_i)
```

**Our Implementation**:
```python
# CARLA-style evaluation
route_result = route_completion(traj, waypoints)
collision_result = collision_rate(traj, obstacles)

# Compute CARLA-like score
carla_score = route_result['completion_rate'] * (
    1.0 if collision_result['collision_rate'] == 0 else 0.0
)
```

### Waymo Open Dataset

**Website**: https://waymo.com/open/

**Planning Metrics**:
- **Displacement error**: L2 distance at multiple horizons (1s, 3s, 5s, 8s)
- **Heading error**: Angle difference in radians
- **Kinematic feasibility**: Check acceleration/jerk limits

**Sim Agents Challenge**:
- **Realism score**: How realistic is the behavior
- **Kinematic metrics**: Velocity, acceleration validity
- **Interaction quality**: Multi-agent scenario handling

### Argoverse 2

**Website**: https://www.argoverse.org/

**Scenario-based Evaluation**:
- **Unprotected turns**: Yielding behavior
- **Lane changes**: Gap acceptance, merge quality
- **Pedestrian crossing**: Stopping distance, yielding

**Metrics**:
- Scenario-specific success rates
- Interaction-aware metrics
- Multi-agent coordination

## Best Practices

### 1. Metric Selection

**Choose metrics based on application**:

| Application | Priority Metrics |
|-------------|-----------------|
| Urban robotaxi | Safety (collision), Comfort, Progress |
| Highway autopilot | Lateral deviation, Velocity error, Comfort |
| Parking assistant | L2 distance, Heading error, Collision |
| Delivery robot | Progress, Route completion, Safety |

### 2. Evaluation Protocol

**Comprehensive evaluation checklist**:

```python
# 1. Planning accuracy
l2 = l2_distance(pred, expert)
ade_fde = average_displacement_error_planning(pred, expert)

# 2. Safety (critical)
collision = collision_rate(pred, obstacles)
assert collision['collision_rate'] == 0.0, "Safety violation!"

# 3. Task completion
progress = progress_score(pred, ref_path)
route = route_completion(pred, waypoints)

# 4. Control quality
lateral = lateral_deviation(pred, ref_centerline)
heading = heading_error(pred_headings, expert_headings)

# 5. Comfort
comfort = comfort_metrics(pred, timestamps)

# 6. Overall assessment
overall = driving_score(pred, expert, obstacles, ref_path, timestamps)
```

### 3. Temporal Considerations

**Short-term vs. Long-term**:
- **Short-term** (< 1s): Reactive control, obstacle avoidance
- **Medium-term** (1-3s): Tactical planning, lane changes
- **Long-term** (3-8s): Strategic planning, route following

**Weighting strategies**:
```python
# Emphasize long-term planning
weights_longterm = np.linspace(1.0, 3.0, T)

# Emphasize near-term control
weights_nearterm = np.linspace(3.0, 1.0, T)

# Exponential decay (recent timesteps critical)
weights_exponential = np.exp(-0.1 * np.arange(T))
```

### 4. Scenario Coverage

**Ensure diverse scenarios**:
- Lane following (straight, curved)
- Intersections (protected/unprotected turns)
- Lane changes (left/right, single/multiple)
- Parking (parallel, perpendicular, angled)
- Traffic interactions (yielding, merging)
- Adverse conditions (rain, night, occlusion)

### 5. Statistical Significance

**Report with uncertainty**:
```python
# Multiple runs
scores = []
for seed in range(100):
    np.random.seed(seed)
    # ... run evaluation
    result = driving_score(pred, expert, obstacles, ref_path, t)
    scores.append(result['driving_score'])

print(f"Driving Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
print(f"95% CI: [{np.percentile(scores, 2.5):.1f}, {np.percentile(scores, 97.5):.1f}]")
```

### 6. Ablation Studies

**Component analysis**:
```python
# Baseline model
baseline_score = driving_score(baseline_pred, expert, obstacles, ref, t)

# Improved planning module
improved_planning_score = driving_score(improved_pred, expert, obstacles, ref, t)

# Compute improvement
improvement = improved_planning_score['driving_score'] - baseline_score['driving_score']
print(f"Planning improvement: +{improvement:.1f} points")

# Per-component improvement
for component in ['planning_accuracy', 'safety_score', 'progress_score', 'comfort_score']:
    delta = improved_planning_score[component] - baseline_score[component]
    print(f"  {component}: {delta:+.1f}")
```

### 7. Real-world vs. Simulation

**Sim-to-real considerations**:
- Simulation metrics are optimistic (perfect sensing, deterministic)
- Real-world: Add sensor noise, localization error, actuation delay
- Domain adaptation: Evaluate on both sim and real data
- Reality gap: Expect 10-20% performance drop

### 8. Computational Efficiency

**Optimization tips**:
```python
# Vectorized computation (fast)
dists = np.linalg.norm(pred_traj - expert_traj, axis=1)
l2 = np.mean(dists)

# Avoid loops (slow)
# for i in range(len(pred_traj)):
#     dist = np.linalg.norm(pred_traj[i] - expert_traj[i])

# Batch processing
results = []
for traj in trajectories:
    result = driving_score(traj, expert, obstacles, ref, t)
    results.append(result)
```

### 9. Failure Analysis

**Debugging poor performance**:
```python
# Identify failure modes
result = driving_score(pred, expert, obstacles, ref, t)

if result['safety_score'] < 100:
    print("SAFETY FAILURE: Collision detected")
    coll = collision_rate(pred, obstacles)
    print(f"  First collision at t={coll['first_collision']}")

if result['progress_score'] < 50:
    print("PROGRESS FAILURE: Route incomplete")
    prog = progress_score(pred, ref)
    print(f"  Only {prog['progress_ratio']:.1%} completed")

if result['comfort_score'] < 70:
    print("COMFORT FAILURE: Aggressive driving")
    comfort = comfort_metrics(pred, t)
    print(f"  {comfort['comfort_violations']} violations")
```

### 10. Reporting Standards

**Complete evaluation report should include**:
1. **Dataset**: Name, version, scenario types, number of samples
2. **Model**: Architecture, parameters, training details
3. **Metrics**: All relevant metrics with mean ± std
4. **Scenarios**: Performance breakdown by scenario type
5. **Failures**: Collision rate, failure cases, edge cases
6. **Comparison**: Baseline models, state-of-the-art benchmarks
7. **Visualizations**: Trajectory plots, error distributions
8. **Reproducibility**: Code, hyperparameters, random seeds

## Example: Complete Evaluation Pipeline

```python
import numpy as np
from admetrics.planning import (
    l2_distance, collision_rate, progress_score,
    lateral_deviation, comfort_metrics, driving_score
)

def evaluate_end_to_end_model(model, test_scenarios):
    """Complete evaluation pipeline."""
    
    results = {
        'l2_distances': [],
        'collision_rates': [],
        'progress_scores': [],
        'lateral_errors': [],
        'comfort_scores': [],
        'driving_scores': []
    }
    
    for scenario in test_scenarios:
        # Get model prediction
        pred_traj = model.predict(scenario['sensor_data'])
        
        # Extract ground truth
        expert_traj = scenario['expert_trajectory']
        obstacles = scenario['obstacles']
        ref_path = scenario['reference_path']
        timestamps = scenario['timestamps']
        
        # Compute metrics
        results['l2_distances'].append(
            l2_distance(pred_traj, expert_traj)
        )
        
        coll = collision_rate(pred_traj, obstacles)
        results['collision_rates'].append(coll['collision_rate'])
        
        prog = progress_score(pred_traj, ref_path)
        results['progress_scores'].append(prog['progress_ratio'])
        
        lat = lateral_deviation(pred_traj, ref_path)
        results['lateral_errors'].append(lat['mean_lateral_error'])
        
        comfort = comfort_metrics(pred_traj, timestamps)
        results['comfort_scores'].append(comfort['comfort_rate'])
        
        overall = driving_score(pred_traj, expert_traj, obstacles, ref_path, timestamps)
        results['driving_scores'].append(overall['driving_score'])
    
    # Aggregate results
    report = {
        'L2 Distance': f"{np.mean(results['l2_distances']):.3f} ± {np.std(results['l2_distances']):.3f}m",
        'Collision Rate': f"{np.mean(results['collision_rates']):.2%}",
        'Progress': f"{np.mean(results['progress_scores']):.1%}",
        'Lateral Error': f"{np.mean(results['lateral_errors']):.3f}m",
        'Comfort': f"{np.mean(results['comfort_scores']):.1%}",
        'Driving Score': f"{np.mean(results['driving_scores']):.1f}/100"
    }
    
    return report

# Usage
# report = evaluate_end_to_end_model(my_model, test_data)
# for metric, value in report.items():
#     print(f"{metric}: {value}")
```

## References

1. **nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles**
   - Caesar, H., Kabzan, J., Tan, K.S., Fong, W.K., Wolff, E., Lang, A., Fletcher, L., Beijbom, O., & Omari, S. (2021)
   - https://arxiv.org/abs/2106.11810
   - https://www.nuscenes.org/nuplan
   - 1200h driving data from 4 cities (Boston, Pittsburgh, Las Vegas, Singapore), closed-loop simulation framework

2. **CARLA: An open urban driving simulator**
   - Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017)
   - CoRL 2017
   - https://arxiv.org/abs/1711.03938
   - https://carla.org
   - Open-source simulator for autonomous driving research

3. **Scalability in perception for autonomous driving: Waymo open dataset**
   - Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo, J., Zhou, Y., Chai, Y., Caine, B., Vasudevan, V., Han, W., Ngiam, J., Zhao, H., Timofeev, A., Ettinger, S., Krivokon, M., Gao, A., Joshi, A., Zhang, Y., et al. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1912.04838
   - https://waymo.com/open
   - Large-scale dataset with 1,150 scenes, diverse urban/suburban environments, high-quality LiDAR and camera data

4. **Argoverse 2: Next generation datasets for self-driving perception and forecasting**
   - Wilson, B., Qi, W., Agarwal, T., Lambert, J., Singh, J., Khandelwal, S., Pan, B., Kumar, R., Hartnett, A., Khandelwal, S., Pan, B., et al. (2023)
   - NeurIPS 2021 Datasets and Benchmarks Track
   - https://arxiv.org/abs/2301.00493
   - https://argoverse.github.io/user-guide/
   - Three datasets: Sensor Dataset (1,000 sequences), Lidar Dataset (20,000 sequences), Motion Forecasting Dataset (250,000 scenarios)

5. **Planning-oriented autonomous driving (UniAD)**
   - Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., Lu, L., Jia, X., Liu, Q., Dai, J., Qiao, Y., & Li, H. (2023)
   - CVPR 2023 (Award Candidate - Best Paper)
   - https://arxiv.org/abs/2212.10156
   - https://github.com/OpenDriveLab/UniAD
   - Unified autonomous driving framework with hierarchical task planning (perception, prediction, planning)

6. **End-to-end driving via conditional imitation learning**
   - Codevilla, F., Müller, M., López, A., Koltun, V., & Dosovitskiy, A. (2018)
   - ICRA 2018
   - https://arxiv.org/abs/1710.02410
   - Imitation learning approach for autonomous driving

## See Also

- [DETECTION_METRICS.md](DETECTION_METRICS.md) - Object detection metrics
- [TRACKING_METRICS.md](TRACKING_METRICS.md) - Multi-object tracking metrics
- [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) - Motion prediction metrics
- [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) - 3D occupancy metrics
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Quick reference for all metrics
