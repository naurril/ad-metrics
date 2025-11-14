"""
Example usage of end-to-end planning metrics for autonomous driving.

This script demonstrates how to evaluate end-to-end driving models that directly
output planned trajectories. Metrics cover planning accuracy, safety, comfort,
and overall driving performance.
"""

import numpy as np
from admetrics.planning import (
    l2_distance,
    collision_rate,
    progress_score,
    route_completion,
    average_displacement_error_planning,
    lateral_deviation,
    heading_error,
    velocity_error,
    comfort_metrics,
    driving_score,
    planning_kl_divergence,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def example_1_l2_distance():
    """Example 1: L2 Distance - Primary metric for trajectory matching."""
    print_section("Example 1: L2 Distance for Trajectory Matching")
    
    # Generate predicted and expert trajectories (3 seconds, 10 Hz = 30 points)
    np.random.seed(42)
    expert_traj = np.array([[i*0.5, 0] for i in range(30)])  # Straight line
    predicted_traj = expert_traj + np.random.normal(0, 0.1, expert_traj.shape)  # Slight noise
    
    # Uniform weights
    dist_uniform = l2_distance(predicted_traj, expert_traj)
    
    # Emphasize long-term planning (later timesteps weighted more)
    weights = np.linspace(1.0, 2.0, 30)
    dist_weighted = l2_distance(predicted_traj, expert_traj, weights=weights)
    
    print(f"Trajectory length: {len(expert_traj)} waypoints")
    print(f"L2 distance (uniform): {dist_uniform:.4f} m")
    print(f"L2 distance (weighted): {dist_weighted:.4f} m")
    print("\nLower is better. Typical good performance: < 0.5m")


def example_2_collision_safety():
    """Example 2: Collision Detection and Safety."""
    print_section("Example 2: Collision Detection and Safety")
    
    # Ego vehicle planned trajectory
    ego_traj = np.array([[0, 0], [2, 0], [4, 0], [6, 0], [8, 0], [10, 0]])
    
    # Static obstacle (parked car)
    static_obstacle = np.array([[5, 0.5]])
    
    # Dynamic obstacle (crossing pedestrian)
    dynamic_obstacle = np.array([[8, 5], [8, 3], [8, 1], [8, -1], [8, -3], [8, -5]])
    
    obstacles = [static_obstacle, dynamic_obstacle]
    
    result = collision_rate(
        ego_traj, 
        obstacles,
        vehicle_size=(4.5, 2.0),  # Length x Width in meters
        safety_margin=0.5  # Add 0.5m safety buffer
    )
    
    print(f"Total timesteps: {len(ego_traj)}")
    print(f"Collision rate: {result['collision_rate']:.2%}")
    print(f"Number of collisions: {result['num_collisions']}")
    print(f"Collision timesteps: {result['collision_timesteps']}")
    print(f"First collision at: {result['first_collision']}")
    print("\nSafety critical: collision_rate should be 0.0%")


def example_3_progress_and_completion():
    """Example 3: Progress and Route Completion."""
    print_section("Example 3: Progress Score and Route Completion")
    
    # Reference path (planned route)
    reference_path = np.array([[0, 0], [5, 0], [10, 0], [15, 5], [20, 10], [25, 10]])
    
    # Executed trajectory (slightly deviates but reaches goal)
    executed_traj = np.array([[0, 0], [5, 0.5], [10, 0.3], [15, 5.2], [20, 10.1]])
    
    # Progress metrics
    progress_result = progress_score(executed_traj, reference_path)
    
    print("Progress Metrics:")
    print(f"  Distance traveled: {progress_result['progress']:.2f} m")
    print(f"  Progress ratio: {progress_result['progress_ratio']:.2%}")
    print(f"  Goal reached: {progress_result['goal_reached']}")
    print(f"  Final distance to goal: {progress_result['final_distance_to_goal']:.2f} m")
    
    # Route completion with waypoints
    waypoints = np.array([[5, 0], [10, 0], [15, 5], [20, 10]])
    route_result = route_completion(executed_traj, waypoints, completion_radius=1.0)
    
    print(f"\nRoute Completion Metrics:")
    print(f"  Waypoints reached: {route_result['num_waypoints_reached']}/{route_result['total_waypoints']}")
    print(f"  Completion rate: {route_result['completion_rate']:.2%}")
    print(f"  Waypoint status: {route_result['waypoint_status']}")


def example_4_path_following_accuracy():
    """Example 4: Path Following - Lateral Deviation and Heading Error."""
    print_section("Example 4: Path Following Accuracy")
    
    # Reference lane centerline
    centerline = np.array([[i, 0] for i in range(50)])
    
    # Actual trajectory with some lateral deviations
    np.random.seed(42)
    actual_traj = np.array([[i, 0.3 * np.sin(i * 0.2) + np.random.normal(0, 0.1)] for i in range(40)])
    
    lateral_result = lateral_deviation(actual_traj, centerline)
    
    print("Lateral Deviation Metrics:")
    print(f"  Mean lateral error: {lateral_result['mean_lateral_error']:.3f} m")
    print(f"  Max lateral error: {lateral_result['max_lateral_error']:.3f} m")
    print(f"  Std lateral error: {lateral_result['std_lateral_error']:.3f} m")
    print(f"  Lane keeping rate: {lateral_result['lane_keeping_rate']:.2%}")
    print("\n  (Lane width assumed: 3.5m, keeping within 1.75m from center)")
    
    # Heading error
    expert_headings = np.linspace(0, np.pi/4, 30)  # Gradual turn
    predicted_headings = expert_headings + np.random.normal(0, 0.05, 30)  # Small errors
    
    heading_result = heading_error(predicted_headings, expert_headings)
    
    print(f"\nHeading Error Metrics:")
    print(f"  Mean heading error: {heading_result['mean_heading_error']:.4f} rad ({heading_result['mean_heading_error_deg']:.2f}°)")
    print(f"  Max heading error: {heading_result['max_heading_error']:.4f} rad")


def example_5_velocity_control():
    """Example 5: Velocity Control Accuracy."""
    print_section("Example 5: Velocity Control Accuracy")
    
    # Expert velocity profile (acceleration, cruise, deceleration)
    expert_vel = np.concatenate([
        np.linspace(0, 15, 10),  # Accelerate to 15 m/s
        np.ones(20) * 15,  # Cruise at 15 m/s
        np.linspace(15, 0, 10)  # Decelerate to stop
    ])
    
    # Predicted velocity with some tracking error
    np.random.seed(42)
    predicted_vel = expert_vel + np.random.normal(0, 0.5, len(expert_vel))
    
    vel_result = velocity_error(predicted_vel, expert_vel)
    
    print("Velocity Control Metrics:")
    print(f"  Mean velocity error: {vel_result['mean_velocity_error']:.3f} m/s")
    print(f"  RMSE velocity: {vel_result['rmse_velocity']:.3f} m/s")
    print(f"  Max velocity error: {vel_result['max_velocity_error']:.3f} m/s")
    print("\nGood performance: < 1.0 m/s mean error")


def example_6_comfort_evaluation():
    """Example 6: Comfort Metrics - Acceleration and Jerk."""
    print_section("Example 6: Comfort Metrics")
    
    # Smooth trajectory
    smooth_traj = np.array([[i**1.5 * 0.5, 0] for i in range(20)])
    smooth_t = np.linspace(0, 5, 20)
    
    smooth_comfort = comfort_metrics(smooth_traj, smooth_t, max_acceleration=2.0, max_jerk=2.0)
    
    print("Smooth Trajectory:")
    print(f"  Mean acceleration: {smooth_comfort['mean_acceleration']:.3f} m/s²")
    print(f"  Max acceleration: {smooth_comfort['max_acceleration']:.3f} m/s²")
    print(f"  Mean jerk: {smooth_comfort['mean_jerk']:.3f} m/s³")
    print(f"  Max jerk: {smooth_comfort['max_jerk']:.3f} m/s³")
    print(f"  Comfort rate: {smooth_comfort['comfort_rate']:.2%}")
    print(f"  Comfort violations: {smooth_comfort['comfort_violations']}")
    
    # Aggressive trajectory with hard braking
    aggressive_traj = np.array([[0, 0], [2, 0], [3, 0], [3.5, 0], [3.6, 0], [3.6, 0]])
    aggressive_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    aggressive_comfort = comfort_metrics(aggressive_traj, aggressive_t, max_acceleration=2.0, max_jerk=2.0)
    
    print(f"\nAggressive Trajectory (with hard braking):")
    print(f"  Mean acceleration: {aggressive_comfort['mean_acceleration']:.3f} m/s²")
    print(f"  Max acceleration: {aggressive_comfort['max_acceleration']:.3f} m/s²")
    print(f"  Comfort rate: {aggressive_comfort['comfort_rate']:.2%}")
    print(f"  Comfort violations: {aggressive_comfort['comfort_violations']}")


def example_7_comprehensive_driving_score():
    """Example 7: Comprehensive Driving Score."""
    print_section("Example 7: Comprehensive Driving Score")
    
    # Scenario: Urban driving with turns
    expert_traj = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0.5],
        [5, 1], [6, 1.5], [7, 2], [8, 2.5], [9, 3]
    ])
    
    # Predicted trajectory with slight errors
    np.random.seed(42)
    predicted_traj = expert_traj + np.random.normal(0, 0.15, expert_traj.shape)
    
    # Reference path for progress evaluation
    reference_path = np.array([
        [0, 0], [2, 0], [4, 0.5], [6, 1.5], [8, 2.5], [10, 3.5], [12, 4]
    ])
    
    # Obstacles in the scene
    obstacles = [
        np.array([[3, 2]]),  # Static obstacle
        np.array([[6, -1]])  # Another static obstacle
    ]
    
    # Timestamps
    timestamps = np.linspace(0, 3, 10)
    
    # Calculate comprehensive driving score
    result = driving_score(
        predicted_traj,
        expert_traj,
        obstacles,
        reference_path,
        timestamps
    )
    
    print("Comprehensive Driving Score:")
    print(f"  Overall Driving Score: {result['driving_score']:.2f}/100")
    print(f"\n  Component Scores:")
    print(f"    Planning Accuracy: {result['planning_accuracy']:.2f}/100")
    print(f"    Safety Score: {result['safety_score']:.2f}/100")
    print(f"    Progress Score: {result['progress_score']:.2f}/100")
    print(f"    Comfort Score: {result['comfort_score']:.2f}/100")
    
    # Custom weights emphasizing safety
    safety_focused_weights = {
        'planning': 0.2,
        'safety': 0.6,
        'progress': 0.1,
        'comfort': 0.1
    }
    
    result_safety_focused = driving_score(
        predicted_traj,
        expert_traj,
        obstacles,
        reference_path,
        timestamps,
        weights=safety_focused_weights
    )
    
    print(f"\n  With Safety-Focused Weights:")
    print(f"    Driving Score: {result_safety_focused['driving_score']:.2f}/100")


def example_8_imitation_learning_kl():
    """Example 8: KL Divergence for Imitation Learning."""
    print_section("Example 8: KL Divergence for Imitation Learning")
    
    # Expert action distribution (e.g., steering angle histogram)
    expert_actions = np.array([0.1, 0.3, 0.4, 0.2, 0.05, 0.05])
    
    # Learned policy distribution (close to expert)
    learned_actions_good = np.array([0.12, 0.28, 0.38, 0.18, 0.06, 0.08])
    
    # Learned policy distribution (poor match)
    learned_actions_poor = np.array([0.3, 0.1, 0.1, 0.3, 0.1, 0.1])
    
    kl_good = planning_kl_divergence(learned_actions_good, expert_actions)
    kl_poor = planning_kl_divergence(learned_actions_poor, expert_actions)
    
    print("KL Divergence (Imitation Learning):")
    print(f"  Expert distribution: {expert_actions}")
    print(f"\n  Good learned policy: {learned_actions_good}")
    print(f"  KL divergence: {kl_good:.4f}")
    print(f"\n  Poor learned policy: {learned_actions_poor}")
    print(f"  KL divergence: {kl_poor:.4f}")
    print(f"\nLower KL divergence indicates better policy matching.")
    print(f"Improvement: {((kl_poor - kl_good) / kl_poor * 100):.1f}% reduction")


def example_9_end_to_end_benchmark():
    """Example 9: Complete End-to-End Benchmark Scenario."""
    print_section("Example 9: Complete End-to-End Benchmark Evaluation")
    
    print("Scenario: Urban navigation with traffic")
    print("Model: Vision-based end-to-end planner")
    print("Duration: 10 seconds @ 5Hz = 50 waypoints\n")
    
    # Generate realistic scenario
    np.random.seed(42)
    t = np.linspace(0, 10, 50)
    
    # Expert trajectory (smooth S-curve)
    s = t / 10  # Normalize to [0, 1]
    expert_x = t * 3  # 3 m/s average speed
    expert_y = 10 * s**2 * (3 - 2*s)  # S-curve
    expert_traj = np.column_stack([expert_x, expert_y])
    
    # Predicted trajectory (with realistic errors)
    pred_traj = expert_traj + np.random.normal(0, 0.3, expert_traj.shape)
    
    # Reference path
    ref_path = np.column_stack([np.linspace(0, 35, 70), np.linspace(0, 12, 70)])
    
    # Obstacles
    obstacles = [
        np.array([[15, 3]]),  # Parked car
        np.array([[25, 8]]),  # Pedestrian
    ]
    
    # Waypoints to pass
    waypoints = np.array([[10, 3], [20, 7], [30, 10]])
    
    # Evaluate all metrics
    print("Results:")
    print("-" * 80)
    
    # 1. L2 distance
    l2 = l2_distance(pred_traj, expert_traj)
    print(f"1. L2 Distance: {l2:.3f} m")
    
    # 2. ADE/FDE
    ade_result = average_displacement_error_planning(pred_traj, expert_traj, timestep_weights='linear')
    print(f"2. ADE: {ade_result['ADE']:.3f} m, FDE: {ade_result['FDE']:.3f} m")
    
    # 3. Collision safety
    coll = collision_rate(pred_traj, obstacles, vehicle_size=(4.5, 2.0))
    print(f"3. Collision Rate: {coll['collision_rate']:.1%} ({coll['num_collisions']} collisions)")
    
    # 4. Progress
    prog = progress_score(pred_traj, ref_path)
    print(f"4. Progress: {prog['progress']:.1f}m ({prog['progress_ratio']:.1%}), Goal: {prog['goal_reached']}")
    
    # 5. Route completion
    route = route_completion(pred_traj, waypoints, completion_radius=2.0)
    print(f"5. Waypoints: {route['num_waypoints_reached']}/{route['total_waypoints']} ({route['completion_rate']:.1%})")
    
    # 6. Lateral deviation
    lateral = lateral_deviation(pred_traj, ref_path)
    print(f"6. Lateral Error: {lateral['mean_lateral_error']:.3f}m (max: {lateral['max_lateral_error']:.3f}m)")
    
    # 7. Comfort
    comfort = comfort_metrics(pred_traj, t, max_acceleration=3.0, max_jerk=3.0)
    print(f"7. Comfort: Rate={comfort['comfort_rate']:.1%}, Violations={comfort['comfort_violations']}")
    
    # 8. Overall score
    overall = driving_score(pred_traj, expert_traj, obstacles, ref_path, t)
    print(f"\n{'='*80}")
    print(f"OVERALL DRIVING SCORE: {overall['driving_score']:.1f}/100")
    print(f"{'='*80}")
    print(f"  Planning: {overall['planning_accuracy']:.1f}  |  Safety: {overall['safety_score']:.1f}  |  Progress: {overall['progress_score']:.1f}  |  Comfort: {overall['comfort_score']:.1f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("End-to-End Autonomous Driving Metrics - Usage Examples")
    print("=" * 80)
    
    example_1_l2_distance()
    example_2_collision_safety()
    example_3_progress_and_completion()
    example_4_path_following_accuracy()
    example_5_velocity_control()
    example_6_comfort_evaluation()
    example_7_comprehensive_driving_score()
    example_8_imitation_learning_kl()
    example_9_end_to_end_benchmark()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")
