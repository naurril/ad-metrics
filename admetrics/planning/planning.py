"""
End-to-End Planning Metrics for Autonomous Driving.

This module provides metrics for evaluating end-to-end autonomous driving models
that directly output planned trajectories or control commands from sensor inputs.
These metrics are commonly used in benchmarks like nuPlan, CARLA, and Waymo Open Dataset.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from scipy.stats import entropy


def l2_distance(
    predicted_trajectory: np.ndarray,
    expert_trajectory: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate L2 distance between predicted and expert trajectories.
    
    This is the primary metric for many end-to-end planning benchmarks,
    measuring the geometric distance between planned and expert paths.
    
    Args:
        predicted_trajectory: Predicted trajectory, shape (T, 2) or (T, 3) for [x, y] or [x, y, z]
        expert_trajectory: Expert/reference trajectory, same shape as predicted
        weights: Optional temporal weights, shape (T,). Later timesteps often weighted higher.
    
    Returns:
        Average L2 distance across all timesteps
    
    Example:
        >>> pred = np.array([[0, 0], [1, 0], [2, 0]])
        >>> expert = np.array([[0, 0], [1, 1], [2, 2]])
        >>> dist = l2_distance(pred, expert)
    """
    if predicted_trajectory.shape != expert_trajectory.shape:
        raise ValueError(
            f"Shape mismatch: pred {predicted_trajectory.shape} vs expert {expert_trajectory.shape}"
        )
    
    # Calculate Euclidean distance at each timestep
    distances = np.linalg.norm(predicted_trajectory - expert_trajectory, axis=1)
    
    if weights is not None:
        if len(weights) != len(distances):
            raise ValueError(f"Weights length {len(weights)} != trajectory length {len(distances)}")
        distances = distances * weights
        return float(np.sum(distances) / np.sum(weights))
    
    return float(np.mean(distances))


def collision_rate(
    trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    vehicle_size: Tuple[float, float] = (4.5, 2.0),
    obstacle_sizes: Optional[List[Tuple[float, float]]] = None,
    safety_margin: float = 0.0
) -> Dict[str, Union[float, int, List[int]]]:
    """
    Calculate collision rate for a planned trajectory.
    
    Checks if the ego vehicle's planned trajectory collides with static or
    dynamic obstacles. Critical safety metric for end-to-end driving.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y] positions
        obstacles: List of obstacle trajectories, each shape (T, 2) or (1, 2) for static
        vehicle_size: Ego vehicle (length, width) in meters
        obstacle_sizes: List of obstacle sizes, defaults to same as vehicle_size
        safety_margin: Additional safety buffer in meters
    
    Returns:
        Dictionary containing:
            - 'collision_rate': Fraction of timesteps with collision
            - 'num_collisions': Total number of collision timesteps
            - 'collision_timesteps': List of timestep indices with collisions
            - 'first_collision': Timestep of first collision (None if no collision)
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> obs = [np.array([[1.5, 0], [1.5, 0], [1.5, 0]])]
        >>> result = collision_rate(traj, obs)
    """
    T = len(trajectory)
    collision_timesteps = []
    
    # Default obstacle sizes
    if obstacle_sizes is None:
        obstacle_sizes = [vehicle_size] * len(obstacles)
    
    ego_length, ego_width = vehicle_size
    ego_radius = np.sqrt((ego_length/2)**2 + (ego_width/2)**2) + safety_margin
    
    for t in range(T):
        ego_pos = trajectory[t]
        
        for obs_traj, (obs_length, obs_width) in zip(obstacles, obstacle_sizes):
            # Handle static obstacles (single position)
            if len(obs_traj.shape) == 1 or len(obs_traj) == 1:
                obs_pos = obs_traj if len(obs_traj.shape) == 1 else obs_traj[0]
            else:
                # Dynamic obstacle - get position at time t
                if t < len(obs_traj):
                    obs_pos = obs_traj[t]
                else:
                    continue  # Obstacle trajectory ended
            
            obs_radius = np.sqrt((obs_length/2)**2 + (obs_width/2)**2) + safety_margin
            
            # Check circular collision
            distance = np.linalg.norm(ego_pos - obs_pos)
            if distance < (ego_radius + obs_radius):
                collision_timesteps.append(t)
                break  # Count each timestep only once
    
    num_collisions = len(collision_timesteps)
    collision_rate_val = num_collisions / T if T > 0 else 0.0
    first_collision = collision_timesteps[0] if collision_timesteps else None
    
    return {
        'collision_rate': float(collision_rate_val),
        'num_collisions': num_collisions,
        'collision_timesteps': collision_timesteps,
        'first_collision': first_collision
    }


def progress_score(
    trajectory: np.ndarray,
    reference_path: np.ndarray,
    goal_position: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate progress along the reference path.
    
    Measures how far the ego vehicle advances along the intended route,
    important for evaluating task completion in end-to-end driving.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        reference_path: Reference centerline/path, shape (N, 2)
        goal_position: Optional goal position, shape (2,). If None, uses last point of reference
    
    Returns:
        Dictionary containing:
            - 'progress': Distance traveled along reference path
            - 'progress_ratio': Fraction of total path completed [0, 1]
            - 'goal_reached': Whether goal was reached (within 2m)
            - 'final_distance_to_goal': Distance from final position to goal
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0]])
        >>> ref = np.array([[0, 0], [5, 0], [10, 0]])
        >>> score = progress_score(traj, ref)
    """
    if goal_position is None:
        goal_position = reference_path[-1]
    
    # Calculate total reference path length
    path_segments = np.diff(reference_path, axis=0)
    path_lengths = np.linalg.norm(path_segments, axis=1)
    total_path_length = np.sum(path_lengths)
    
    # Find closest point on reference path to final trajectory position
    final_pos = trajectory[-1]
    distances_to_path = np.linalg.norm(reference_path - final_pos, axis=1)
    closest_idx = np.argmin(distances_to_path)
    
    # Calculate progress as distance along path to closest point
    if closest_idx == 0:
        progress = 0.0
    else:
        progress = np.sum(path_lengths[:closest_idx])
        # Add partial distance to closest point
        if closest_idx < len(reference_path) - 1:
            prev_point = reference_path[closest_idx - 1]
            next_point = reference_path[closest_idx]
            segment_vec = next_point - prev_point
            to_final = final_pos - prev_point
            # Project onto segment
            projection = np.dot(to_final, segment_vec) / (np.linalg.norm(segment_vec) + 1e-8)
            projection = np.clip(projection, 0, np.linalg.norm(segment_vec))
            progress += projection
    
    progress_ratio = progress / total_path_length if total_path_length > 0 else 0.0
    
    # Check if goal reached (within 2 meters)
    final_distance_to_goal = np.linalg.norm(final_pos - goal_position)
    goal_reached = final_distance_to_goal < 2.0
    
    return {
        'progress': float(progress),
        'progress_ratio': float(np.clip(progress_ratio, 0.0, 1.0)),
        'goal_reached': bool(goal_reached),
        'final_distance_to_goal': float(final_distance_to_goal)
    }


def route_completion(
    trajectory: np.ndarray,
    waypoints: np.ndarray,
    completion_radius: float = 2.0
) -> Dict[str, Union[float, int, List[bool]]]:
    """
    Calculate route completion based on waypoint passing.
    
    Evaluates whether the vehicle successfully passes through required waypoints,
    used in navigation tasks and long-range planning evaluation.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        waypoints: Required waypoints to pass, shape (N, 2)
        completion_radius: Distance threshold to consider waypoint reached (meters)
    
    Returns:
        Dictionary containing:
            - 'completion_rate': Fraction of waypoints reached [0, 1]
            - 'num_waypoints_reached': Number of waypoints successfully reached
            - 'waypoint_status': Boolean list indicating which waypoints were reached
            - 'total_waypoints': Total number of waypoints
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        >>> waypoints = np.array([[1, 0], [2, 0], [5, 0]])
        >>> result = route_completion(traj, waypoints, completion_radius=0.5)
    """
    num_waypoints = len(waypoints)
    waypoint_status = []
    
    for waypoint in waypoints:
        # Check if any point in trajectory is within completion_radius of waypoint
        distances = np.linalg.norm(trajectory - waypoint, axis=1)
        reached = np.any(distances <= completion_radius)
        waypoint_status.append(bool(reached))
    
    num_reached = sum(waypoint_status)
    completion_rate_val = num_reached / num_waypoints if num_waypoints > 0 else 0.0
    
    return {
        'completion_rate': float(completion_rate_val),
        'num_waypoints_reached': num_reached,
        'waypoint_status': waypoint_status,
        'total_waypoints': num_waypoints
    }


def average_displacement_error_planning(
    predicted_trajectory: np.ndarray,
    expert_trajectory: np.ndarray,
    timestep_weights: Optional[str] = 'linear'
) -> Dict[str, float]:
    """
    Calculate Average Displacement Error (ADE) for planning.
    
    Similar to trajectory prediction ADE but applied to planning/control output.
    Often used with temporal weighting to emphasize long-term planning accuracy.
    
    Args:
        predicted_trajectory: Planned trajectory, shape (T, 2) or (T, 3)
        expert_trajectory: Expert trajectory, same shape
        timestep_weights: Weighting scheme - 'uniform', 'linear', 'exponential', or None
    
    Returns:
        Dictionary containing:
            - 'ADE': Average displacement error
            - 'FDE': Final displacement error
            - 'weighted_ADE': ADE with temporal weighting (if weights specified)
    
    Example:
        >>> pred = np.random.rand(10, 2)
        >>> expert = np.random.rand(10, 2)
        >>> errors = average_displacement_error_planning(pred, expert, timestep_weights='linear')
    """
    if predicted_trajectory.shape != expert_trajectory.shape:
        raise ValueError("Trajectory shapes must match")
    
    T = len(predicted_trajectory)
    
    # Calculate per-timestep errors
    displacements = np.linalg.norm(predicted_trajectory - expert_trajectory, axis=1)
    
    ade = float(np.mean(displacements))
    fde = float(displacements[-1])
    
    # Calculate weighted ADE
    if timestep_weights == 'linear':
        weights = np.linspace(1.0, 2.0, T)  # Later timesteps weighted more
    elif timestep_weights == 'exponential':
        weights = np.exp(np.linspace(0, 1, T))
    elif timestep_weights == 'uniform' or timestep_weights is None:
        weights = np.ones(T)
    else:
        weights = np.ones(T)
    
    weighted_ade = float(np.sum(displacements * weights) / np.sum(weights))
    
    return {
        'ADE': ade,
        'FDE': fde,
        'weighted_ADE': weighted_ade
    }


def lateral_deviation(
    trajectory: np.ndarray,
    reference_path: np.ndarray
) -> Dict[str, float]:
    """
    Calculate lateral deviation from reference path.
    
    Measures cross-track error, important for lane-keeping and path-following
    performance in end-to-end driving systems.
    
    Args:
        trajectory: Ego trajectory, shape (T, 2) for [x, y]
        reference_path: Reference centerline, shape (N, 2)
    
    Returns:
        Dictionary containing:
            - 'mean_lateral_error': Average lateral deviation (meters)
            - 'max_lateral_error': Maximum lateral deviation (meters)
            - 'std_lateral_error': Standard deviation of lateral deviation
            - 'lane_keeping_rate': Fraction of time within lane (assuming 1.75m from center)
    
    Example:
        >>> traj = np.array([[0, 0.5], [1, 0.3], [2, -0.2]])
        >>> ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        >>> deviation = lateral_deviation(traj, ref)
    """
    lateral_errors = []
    
    for point in trajectory:
        # Find closest point on reference path
        distances = np.linalg.norm(reference_path - point, axis=1)
        closest_idx = np.argmin(distances)
        lateral_error = distances[closest_idx]
        lateral_errors.append(lateral_error)
    
    lateral_errors = np.array(lateral_errors)
    
    # Lane keeping rate (assuming 3.5m lane width, so 1.75m from center)
    lane_width_half = 1.75
    within_lane = np.sum(lateral_errors <= lane_width_half)
    lane_keeping_rate = within_lane / len(lateral_errors) if len(lateral_errors) > 0 else 0.0
    
    return {
        'mean_lateral_error': float(np.mean(lateral_errors)),
        'max_lateral_error': float(np.max(lateral_errors)),
        'std_lateral_error': float(np.std(lateral_errors)),
        'lane_keeping_rate': float(lane_keeping_rate)
    }


def heading_error(
    predicted_headings: np.ndarray,
    expert_headings: np.ndarray
) -> Dict[str, float]:
    """
    Calculate heading/yaw angle error.
    
    Measures orientation accuracy, important for vehicle control and
    trajectory smoothness in end-to-end systems.
    
    Args:
        predicted_headings: Predicted heading angles in radians, shape (T,)
        expert_headings: Expert heading angles in radians, shape (T,)
    
    Returns:
        Dictionary containing:
            - 'mean_heading_error': Average heading error (radians)
            - 'mean_heading_error_deg': Average heading error (degrees)
            - 'max_heading_error': Maximum heading error (radians)
            - 'std_heading_error': Standard deviation (radians)
    
    Example:
        >>> pred_headings = np.array([0.0, 0.1, 0.2])
        >>> expert_headings = np.array([0.0, 0.15, 0.25])
        >>> error = heading_error(pred_headings, expert_headings)
    """
    if len(predicted_headings) != len(expert_headings):
        raise ValueError("Heading arrays must have same length")
    
    # Calculate angular difference (handling wrap-around)
    diff = predicted_headings - expert_headings
    diff = np.arctan2(np.sin(diff), np.cos(diff))  # Wrap to [-π, π]
    
    errors = np.abs(diff)
    
    return {
        'mean_heading_error': float(np.mean(errors)),
        'mean_heading_error_deg': float(np.degrees(np.mean(errors))),
        'max_heading_error': float(np.max(errors)),
        'std_heading_error': float(np.std(errors))
    }


def velocity_error(
    predicted_velocities: np.ndarray,
    expert_velocities: np.ndarray
) -> Dict[str, float]:
    """
    Calculate velocity/speed error.
    
    Measures how well the planner matches desired speeds,
    important for comfort and traffic flow.
    
    Args:
        predicted_velocities: Predicted velocities in m/s, shape (T,)
        expert_velocities: Expert velocities in m/s, shape (T,)
    
    Returns:
        Dictionary containing:
            - 'mean_velocity_error': Average velocity error (m/s)
            - 'rmse_velocity': Root mean squared velocity error
            - 'max_velocity_error': Maximum velocity error
    
    Example:
        >>> pred_vel = np.array([10.0, 12.0, 15.0])
        >>> expert_vel = np.array([10.0, 13.0, 14.0])
        >>> error = velocity_error(pred_vel, expert_vel)
    """
    if len(predicted_velocities) != len(expert_velocities):
        raise ValueError("Velocity arrays must have same length")
    
    errors = np.abs(predicted_velocities - expert_velocities)
    rmse = np.sqrt(np.mean((predicted_velocities - expert_velocities)**2))
    
    return {
        'mean_velocity_error': float(np.mean(errors)),
        'rmse_velocity': float(rmse),
        'max_velocity_error': float(np.max(errors))
    }


def comfort_metrics(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    max_acceleration: float = 3.0,
    max_jerk: float = 3.0
) -> Dict[str, Union[float, int]]:
    """
    Calculate comfort metrics for driving quality.
    
    Evaluates smoothness and passenger comfort through acceleration,
    jerk, and other dynamic measures. Critical for end-to-end driving quality.
    
    Args:
        trajectory: Trajectory positions, shape (T, 2) for [x, y]
        timestamps: Time at each position, shape (T,)
        max_acceleration: Comfort threshold for acceleration (m/s²)
        max_jerk: Comfort threshold for jerk (m/s³)
    
    Returns:
        Dictionary containing:
            - 'mean_acceleration': Average acceleration magnitude
            - 'max_acceleration': Maximum acceleration
            - 'mean_jerk': Average jerk magnitude
            - 'max_jerk': Maximum jerk
            - 'comfort_violations': Number of timesteps exceeding thresholds
            - 'comfort_rate': Fraction of time within comfort limits
    
    Example:
        >>> traj = np.array([[0, 0], [1, 0], [2.5, 0], [4.5, 0]])
        >>> t = np.array([0.0, 1.0, 2.0, 3.0])
        >>> metrics = comfort_metrics(traj, t)
    """
    if len(trajectory) != len(timestamps):
        raise ValueError("Trajectory and timestamps must have same length")
    
    if len(trajectory) < 3:
        return {
            'mean_acceleration': 0.0,
            'max_acceleration': 0.0,
            'mean_jerk': 0.0,
            'max_jerk': 0.0,
            'comfort_violations': 0,
            'comfort_rate': 1.0
        }
    
    # Calculate velocities
    dt = np.diff(timestamps)
    positions_diff = np.diff(trajectory, axis=0)
    velocities = np.linalg.norm(positions_diff, axis=1) / dt
    
    # Calculate accelerations
    dt_acc = timestamps[2:] - timestamps[:-2]
    accelerations = np.diff(velocities) / dt[:-1]
    
    # Calculate jerk
    if len(accelerations) > 1:
        dt_jerk = timestamps[3:] - timestamps[:-3]
        jerk = np.diff(accelerations) / dt[1:-1]
    else:
        jerk = np.array([0.0])
    
    # Count comfort violations
    violations = 0
    if len(accelerations) > 0:
        violations += np.sum(np.abs(accelerations) > max_acceleration)
    if len(jerk) > 0:
        violations += np.sum(np.abs(jerk) > max_jerk)
    
    total_samples = len(accelerations) + len(jerk)
    comfort_rate = 1.0 - (violations / total_samples) if total_samples > 0 else 1.0
    
    return {
        'mean_acceleration': float(np.mean(np.abs(accelerations))) if len(accelerations) > 0 else 0.0,
        'max_acceleration': float(np.max(np.abs(accelerations))) if len(accelerations) > 0 else 0.0,
        'mean_jerk': float(np.mean(np.abs(jerk))) if len(jerk) > 0 else 0.0,
        'max_jerk': float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0.0,
        'comfort_violations': int(violations),
        'comfort_rate': float(comfort_rate)
    }


def driving_score(
    predicted_trajectory: np.ndarray,
    expert_trajectory: np.ndarray,
    obstacles: List[np.ndarray],
    reference_path: np.ndarray,
    timestamps: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive driving score combining multiple metrics.
    
    Composite metric similar to nuPlan's driving score or CARLA's benchmark score,
    combining planning accuracy, safety, progress, and comfort.
    
    Args:
        predicted_trajectory: Planned trajectory, shape (T, 2)
        expert_trajectory: Expert trajectory, shape (T, 2)
        obstacles: List of obstacle trajectories
        reference_path: Reference path, shape (N, 2)
        timestamps: Time array, shape (T,)
        weights: Optional weights for each component
    
    Returns:
        Dictionary containing:
            - 'driving_score': Overall score [0, 100]
            - 'planning_accuracy': Planning accuracy score [0, 100]
            - 'safety_score': Safety score [0, 100]
            - 'progress_score': Progress score [0, 100]
            - 'comfort_score': Comfort score [0, 100]
    
    Example:
        >>> pred = np.random.rand(10, 2)
        >>> expert = np.random.rand(10, 2)
        >>> obs = [np.random.rand(10, 2)]
        >>> ref = np.random.rand(20, 2)
        >>> t = np.linspace(0, 5, 10)
        >>> score = driving_score(pred, expert, obs, ref, t)
    """
    default_weights = {
        'planning': 0.3,
        'safety': 0.4,
        'progress': 0.2,
        'comfort': 0.1
    }
    
    if weights is None:
        weights = default_weights
    
    # Planning accuracy (lower L2 distance is better)
    l2_dist = l2_distance(predicted_trajectory, expert_trajectory)
    # Convert to score (0-100), assuming 5m is very poor, 0m is perfect
    planning_accuracy = max(0, 100 * (1 - min(l2_dist / 5.0, 1.0)))
    
    # Safety (no collisions is best)
    collision_result = collision_rate(predicted_trajectory, obstacles)
    safety_score_val = 100 * (1 - collision_result['collision_rate'])
    
    # Progress
    progress_result = progress_score(predicted_trajectory, reference_path)
    progress_score_val = 100 * progress_result['progress_ratio']
    
    # Comfort
    comfort_result = comfort_metrics(predicted_trajectory, timestamps)
    comfort_score_val = 100 * comfort_result['comfort_rate']
    
    # Weighted combination
    overall_score = (
        weights['planning'] * planning_accuracy +
        weights['safety'] * safety_score_val +
        weights['progress'] * progress_score_val +
        weights['comfort'] * comfort_score_val
    )
    
    return {
        'driving_score': float(overall_score),
        'planning_accuracy': float(planning_accuracy),
        'safety_score': float(safety_score_val),
        'progress_score': float(progress_score_val),
        'comfort_score': float(comfort_score_val)
    }


def planning_kl_divergence(
    predicted_distribution: np.ndarray,
    expert_distribution: np.ndarray,
    bins: int = 50
) -> float:
    """
    Calculate KL divergence between predicted and expert action distributions.
    
    For imitation learning, measures how well the learned policy matches
    the expert policy in terms of action distributions.
    
    Args:
        predicted_distribution: Predicted action probabilities, shape (N,)
        expert_distribution: Expert action probabilities, shape (N,)
        bins: Number of bins for discretization (if inputs are continuous)
    
    Returns:
        KL divergence value (lower is better)
    
    Example:
        >>> pred_dist = np.array([0.2, 0.5, 0.3])
        >>> expert_dist = np.array([0.1, 0.6, 0.3])
        >>> kl = planning_kl_divergence(pred_dist, expert_dist)
    """
    # Ensure distributions are normalized
    pred_dist = predicted_distribution / (np.sum(predicted_distribution) + 1e-10)
    expert_dist = expert_distribution / (np.sum(expert_distribution) + 1e-10)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    pred_dist = pred_dist + epsilon
    expert_dist = expert_dist + epsilon
    
    # Renormalize after adding epsilon
    pred_dist = pred_dist / np.sum(pred_dist)
    expert_dist = expert_dist / np.sum(expert_dist)
    
    # Calculate KL divergence
    kl_div = entropy(expert_dist, pred_dist)
    
    return float(kl_div)
