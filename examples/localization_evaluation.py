"""
Example: Evaluating ego vehicle localization for autonomous driving.

This example demonstrates how to evaluate localization performance using various
metrics for HD map-based autonomous driving systems. We simulate:
1. GPS-based localization with varying accuracy
2. SLAM-based localization with drift
3. HD map alignment evaluation
4. Convergence rate analysis
"""

import numpy as np
from admetrics.localization import (
    calculate_ate,
    calculate_rte,
    calculate_are,
    calculate_lateral_error,
    calculate_longitudinal_error,
    calculate_convergence_rate,
    calculate_localization_metrics,
    calculate_map_alignment_score,
)


def generate_ground_truth_trajectory(num_poses=100):
    """Generate a ground truth trajectory (e.g., from RTK-GPS or ground truth SLAM)."""
    t = np.linspace(0, 10, num_poses)
    
    # Circular trajectory
    radius = 50.0
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros(num_poses)
    
    # Heading tangent to circle
    yaw = t + np.pi / 2
    
    poses_4d = np.column_stack([x, y, z, yaw])
    timestamps = t
    
    return poses_4d, timestamps


def simulate_gps_localization(gt_poses, horizontal_error=0.5, vertical_error=1.0):
    """Simulate GPS-based localization with typical accuracy."""
    predicted = gt_poses.copy()
    
    # Add Gaussian noise
    predicted[:, 0] += np.random.normal(0, horizontal_error, len(gt_poses))  # x
    predicted[:, 1] += np.random.normal(0, horizontal_error, len(gt_poses))  # y
    predicted[:, 2] += np.random.normal(0, vertical_error, len(gt_poses))    # z
    
    # Add small heading error (1-2 degrees)
    predicted[:, 3] += np.random.normal(0, np.deg2rad(1.5), len(gt_poses))
    
    return predicted


def simulate_slam_with_drift(gt_poses, drift_rate=0.001):
    """Simulate SLAM localization with accumulating drift."""
    predicted = gt_poses.copy()
    
    # Accumulating drift
    for i in range(1, len(predicted)):
        drift = drift_rate * i
        predicted[i, :2] += np.random.normal(0, drift, 2)
        predicted[i, 3] += np.random.normal(0, drift * 0.1)
    
    return predicted


def simulate_initialization_period(gt_poses, convergence_time=2.0, timestamps=None):
    """Simulate localization with initialization period."""
    predicted = gt_poses.copy()
    
    # Add large error at start, converging to small error
    for i, t in enumerate(timestamps):
        if t < convergence_time:
            error_scale = (convergence_time - t) / convergence_time
            predicted[i, :2] += np.random.normal(0, 5.0 * error_scale, 2)
        else:
            predicted[i, :2] += np.random.normal(0, 0.1, 2)
    
    return predicted


def generate_lane_centerlines():
    """Generate HD map lane centerlines (circular road)."""
    # Main lane (inner)
    t = np.linspace(0, 2 * np.pi, 100)
    radius = 50.0
    lane1 = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t)
    ])
    
    # Adjacent lane (outer)
    lane2 = np.column_stack([
        (radius + 3.5) * np.cos(t),
        (radius + 3.5) * np.sin(t)
    ])
    
    return [lane1, lane2]


def main():
    print("=" * 80)
    print("Ego Vehicle Localization Evaluation")
    print("=" * 80)
    
    # Generate ground truth trajectory
    gt_poses, timestamps = generate_ground_truth_trajectory(num_poses=100)
    print(f"\nGround truth trajectory: {len(gt_poses)} poses over {timestamps[-1]:.1f} seconds")
    print(f"Trajectory type: Circular path with radius 50m")
    
    # =========================================================================
    # Example 1: GPS-based localization
    # =========================================================================
    print("\n" + "-" * 80)
    print("Example 1: GPS-based Localization (Consumer-grade GPS)")
    print("-" * 80)
    
    gps_poses = simulate_gps_localization(gt_poses, horizontal_error=0.5, vertical_error=1.0)
    
    # Calculate comprehensive metrics
    gps_metrics = calculate_localization_metrics(
        gps_poses,
        gt_poses,
        timestamps=timestamps,
        lane_width=3.5,
        align=False  # Don't align for GPS (absolute positioning)
    )
    
    print(f"Absolute Trajectory Error (ATE):")
    print(f"  Mean: {gps_metrics['ate_mean']:.3f} m")
    print(f"  Std:  {gps_metrics['ate_std']:.3f} m")
    print(f"  Max:  {gps_metrics['ate_max']:.3f} m")
    
    print(f"\nLateral Error (cross-track):")
    print(f"  Mean: {gps_metrics['lateral_mean']:.3f} m")
    print(f"  Std:  {gps_metrics['lateral_std']:.3f} m")
    print(f"  Lane violation rate: {gps_metrics['lateral_lane_violation_rate']:.1%}")
    
    print(f"\nOrientation Error (ARE):")
    print(f"  Mean: {np.rad2deg(gps_metrics['are_mean']):.2f}°")
    print(f"  Max:  {np.rad2deg(gps_metrics['are_max']):.2f}°")
    
    # =========================================================================
    # Example 2: SLAM with drift
    # =========================================================================
    print("\n" + "-" * 80)
    print("Example 2: SLAM-based Localization (with drift)")
    print("-" * 80)
    
    slam_poses = simulate_slam_with_drift(gt_poses, drift_rate=0.001)
    
    # Calculate RTE to detect drift
    rte_result = calculate_rte(slam_poses[:, :3], gt_poses[:, :3], distances=[10, 50, 100])
    
    print("Relative Trajectory Error (drift analysis):")
    print(f"  RTE @ 10m:  {rte_result.get('rte_10', 0.0):.4f} (normalized)")
    print(f"  RTE @ 50m:  {rte_result.get('rte_50', 0.0):.4f} (normalized)")
    print(f"  RTE @ 100m: {rte_result.get('rte_100', 0.0):.4f} (normalized)")
    drift_detected = rte_result.get('rte_100', 0.0) > 0.05
    print(f"  Drift detected: {drift_detected}")
    
    # Calculate ATE with alignment (for SLAM evaluation)
    slam_metrics = calculate_localization_metrics(
        slam_poses,
        gt_poses,
        timestamps=timestamps,
        lane_width=3.5,
        align=True  # Align trajectories (Umeyama algorithm)
    )
    
    print(f"\nATE after alignment (Umeyama):")
    print(f"  Mean: {slam_metrics['ate_mean']:.3f} m")
    print(f"  Std:  {slam_metrics['ate_std']:.3f} m")
    
    # =========================================================================
    # Example 3: Initialization & convergence
    # =========================================================================
    print("\n" + "-" * 80)
    print("Example 3: Localization with Initialization Period")
    print("-" * 80)
    
    init_poses = simulate_initialization_period(gt_poses, convergence_time=2.0, timestamps=timestamps)
    
    # Calculate position errors
    position_errors = np.linalg.norm(init_poses[:, :3] - gt_poses[:, :3], axis=1)
    
    # Calculate convergence rate
    convergence = calculate_convergence_rate(
        position_errors,
        timestamps=timestamps,
        threshold=0.5
    )
    
    print(f"Convergence analysis:")
    print(f"  Converged: {convergence['converged']}")
    if convergence['converged']:
        print(f"  Time to convergence: {convergence['time_to_convergence']:.2f} s")
        print(f"  Initial error: {convergence['initial_error']:.2f} m")
        print(f"  Converged error: {convergence['converged_error']:.2f} m")
        print(f"  Convergence rate: {convergence['convergence_rate']:.2f} m/s")
    
    # =========================================================================
    # Example 4: HD map alignment
    # =========================================================================
    print("\n" + "-" * 80)
    print("Example 4: HD Map Alignment Evaluation")
    print("-" * 80)
    
    lane_centerlines = generate_lane_centerlines()
    
    # Evaluate alignment for GPS poses
    map_alignment = calculate_map_alignment_score(
        gps_poses[:, :2],  # 2D positions
        lane_centerlines
    )
    
    print(f"Map alignment metrics:")
    print(f"  Mean distance to lane: {map_alignment['mean_distance_to_lane']:.3f} m")
    print(f"  Median distance to lane: {map_alignment['median_distance_to_lane']:.3f} m")
    print(f"  Max distance to lane: {map_alignment['max_distance_to_lane']:.3f} m")
    print(f"  Alignment rate (within 2.0m): {map_alignment['alignment_rate']:.1%}")
    
    # =========================================================================
    # Example 5: Comparing different localization methods
    # =========================================================================
    print("\n" + "-" * 80)
    print("Example 5: Comparison of Localization Methods")
    print("-" * 80)
    
    # RTK-GPS simulation (high accuracy)
    rtk_poses = simulate_gps_localization(gt_poses, horizontal_error=0.02, vertical_error=0.05)
    rtk_metrics = calculate_localization_metrics(rtk_poses, gt_poses, timestamps=timestamps)
    
    methods = {
        'Consumer GPS': gps_metrics,
        'RTK-GPS': rtk_metrics,
        'SLAM (aligned)': slam_metrics,
    }
    
    print(f"\n{'Method':<20} {'ATE (m)':<12} {'Lateral (m)':<15} {'ARE (deg)':<12}")
    print("-" * 60)
    for method, metrics in methods.items():
        ate = metrics['ate_mean']
        lat = metrics['lateral_mean']
        are = np.rad2deg(metrics['are_mean'])
        print(f"{method:<20} {ate:<12.4f} {lat:<15.4f} {are:<12.3f}")
    
    # =========================================================================
    # Example 6: Using 7D poses (position + quaternion)
    # =========================================================================
    print("\n" + "-" * 80)
    print("Example 6: 7D Pose Evaluation (Position + Quaternion)")
    print("-" * 80)
    
    # Convert 4D (x, y, z, yaw) to 7D (x, y, z, qw, qx, qy, qz)
    def yaw_to_quaternion(yaw):
        """Convert yaw angle to quaternion [qw, qx, qy, qz]."""
        qw = np.cos(yaw / 2)
        qx = 0
        qy = 0
        qz = np.sin(yaw / 2)
        return np.array([qw, qx, qy, qz])
    
    gt_poses_7d = np.column_stack([
        gt_poses[:, :3],
        np.array([yaw_to_quaternion(yaw) for yaw in gt_poses[:, 3]])
    ])
    
    gps_poses_7d = np.column_stack([
        gps_poses[:, :3],
        np.array([yaw_to_quaternion(yaw) for yaw in gps_poses[:, 3]])
    ])
    
    # Calculate ARE with quaternions
    are_result = calculate_are(gps_poses_7d, gt_poses_7d)
    
    print(f"Absolute Rotation Error (quaternion-based):")
    print(f"  Mean: {np.rad2deg(are_result['mean']):.3f}°")
    print(f"  Std:  {np.rad2deg(are_result['std']):.3f}°")
    print(f"  Max:  {np.rad2deg(are_result['max']):.3f}°")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
