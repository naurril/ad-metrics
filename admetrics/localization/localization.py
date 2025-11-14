"""
Localization metrics for evaluating ego vehicle pose estimation.

These metrics assess the accuracy of localization modules that estimate
the ego vehicle's position and orientation using HD maps, GPS, IMU, LiDAR,
and other sensors in autonomous driving systems.
"""

import numpy as np
from typing import List, Dict, Union, Tuple, Optional


def calculate_ate(
    predicted_poses: Union[np.ndarray, List],
    ground_truth_poses: Union[np.ndarray, List],
    align: bool = False
) -> Dict[str, float]:
    """
    Calculate Absolute Trajectory Error (ATE).
    
    ATE measures the absolute distance between predicted and ground truth
    positions over the entire trajectory. This is the primary metric for
    localization accuracy.
    
    Args:
        predicted_poses: Predicted poses (N, 3) for [x, y, z] or (N, 7) for [x, y, z, qw, qx, qy, qz]
        ground_truth_poses: Ground truth poses (N, 3) or (N, 7)
        align: Whether to align trajectories using Umeyama algorithm before comparison
    
    Returns:
        Dictionary with ATE metrics (mean, std, min, max, rmse)
        
    Example:
        >>> pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        >>> gt = np.array([[0, 0, 0], [1.1, 0, 0], [2.2, 0, 0]])
        >>> result = calculate_ate(pred, gt)
        >>> print(f"ATE: {result['mean']:.3f}m")
    """
    predicted_poses = np.array(predicted_poses, dtype=np.float64)
    ground_truth_poses = np.array(ground_truth_poses, dtype=np.float64)
    
    if predicted_poses.shape[0] != ground_truth_poses.shape[0]:
        raise ValueError(
            f"Number of poses must match: {predicted_poses.shape[0]} vs {ground_truth_poses.shape[0]}"
        )
    
    # Extract positions (first 3 columns)
    pred_positions = predicted_poses[:, :3]
    gt_positions = ground_truth_poses[:, :3]
    
    # Optionally align trajectories (useful for SLAM evaluation)
    if align:
        pred_positions = _align_trajectories(pred_positions, gt_positions)
    
    # Calculate position errors
    position_errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    
    return {
        'mean': float(np.mean(position_errors)),
        'std': float(np.std(position_errors)),
        'min': float(np.min(position_errors)),
        'max': float(np.max(position_errors)),
        'rmse': float(np.sqrt(np.mean(position_errors**2))),
        'median': float(np.median(position_errors))
    }


def calculate_rte(
    predicted_poses: Union[np.ndarray, List],
    ground_truth_poses: Union[np.ndarray, List],
    distances: List[float] = [100, 200, 300, 400, 500, 600, 700, 800]
) -> Dict[str, float]:
    """
    Calculate Relative Trajectory Error (RTE).
    
    RTE measures the drift over fixed distances, useful for understanding
    how localization error accumulates over distance traveled.
    
    Args:
        predicted_poses: Predicted poses (N, 3) or (N, 7)
        ground_truth_poses: Ground truth poses (N, 3) or (N, 7)
        distances: List of distances (in meters) to evaluate drift
    
    Returns:
        Dictionary with RTE for each distance
        
    Example:
        >>> result = calculate_rte(pred_poses, gt_poses, distances=[100, 200])
        >>> print(f"RTE@100m: {result['rte_100']:.2%}")
    """
    predicted_poses = np.array(predicted_poses, dtype=np.float64)
    ground_truth_poses = np.array(ground_truth_poses, dtype=np.float64)
    
    pred_positions = predicted_poses[:, :3]
    gt_positions = ground_truth_poses[:, :3]
    
    # Calculate cumulative distances
    pred_dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)])
    gt_dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)])
    
    results = {}
    
    for dist in distances:
        errors = []
        
        # Find pose pairs separated by approximately 'dist' meters
        for i in range(len(gt_positions)):
            # Find index j where distance from i is approximately dist
            target_dist = gt_dists[i] + dist
            j = np.argmin(np.abs(gt_dists - target_dist))
            
            if j >= len(gt_positions) or gt_dists[j] - gt_dists[i] < dist * 0.9:
                continue
            
            # Relative transformation error
            gt_rel = gt_positions[j] - gt_positions[i]
            pred_rel = pred_positions[j] - pred_positions[i]
            
            error = np.linalg.norm(gt_rel - pred_rel)
            traveled = gt_dists[j] - gt_dists[i]
            
            # Normalize by distance to get percentage error
            if traveled > 0:
                errors.append(error / traveled)
        
        if errors:
            results[f'rte_{int(dist)}'] = float(np.mean(errors))
            results[f'rte_{int(dist)}_std'] = float(np.std(errors))
    
    return results


def calculate_are(
    predicted_poses: Union[np.ndarray, List],
    ground_truth_poses: Union[np.ndarray, List]
) -> Dict[str, float]:
    """
    Calculate Absolute Rotation Error (ARE).
    
    ARE measures the angular difference between predicted and ground truth
    orientations (heading/yaw).
    
    Args:
        predicted_poses: Predicted poses with orientation
                        (N, 4) for [x, y, z, yaw] or
                        (N, 7) for [x, y, z, qw, qx, qy, qz]
        ground_truth_poses: Ground truth poses (N, 4) or (N, 7)
    
    Returns:
        Dictionary with ARE metrics in degrees
        
    Example:
        >>> pred = np.array([[0, 0, 0, 0], [1, 0, 0, 0.1]])
        >>> gt = np.array([[0, 0, 0, 0], [1, 0, 0, 0.05]])
        >>> result = calculate_are(pred, gt)
        >>> print(f"ARE: {result['mean']:.2f} degrees")
    """
    predicted_poses = np.array(predicted_poses, dtype=np.float64)
    ground_truth_poses = np.array(ground_truth_poses, dtype=np.float64)
    
    # Extract orientations
    if predicted_poses.shape[1] == 4:
        # [x, y, z, yaw] format
        pred_yaw = predicted_poses[:, 3]
        gt_yaw = ground_truth_poses[:, 3]
    elif predicted_poses.shape[1] == 7:
        # [x, y, z, qw, qx, qy, qz] format - extract yaw from quaternion
        pred_yaw = _quaternion_to_yaw(predicted_poses[:, 3:7])
        gt_yaw = _quaternion_to_yaw(ground_truth_poses[:, 3:7])
    else:
        raise ValueError(f"Unsupported pose format: {predicted_poses.shape}")
    
    # Calculate angular errors
    angular_errors = np.abs(_normalize_angles(pred_yaw - gt_yaw))
    angular_errors_deg = np.rad2deg(angular_errors)
    
    return {
        'mean': float(np.mean(angular_errors_deg)),
        'std': float(np.std(angular_errors_deg)),
        'min': float(np.min(angular_errors_deg)),
        'max': float(np.max(angular_errors_deg)),
        'rmse': float(np.sqrt(np.mean(angular_errors_deg**2))),
        'median': float(np.median(angular_errors_deg))
    }


def calculate_lateral_error(
    predicted_poses: Union[np.ndarray, List],
    ground_truth_poses: Union[np.ndarray, List],
    lane_width: float = 3.5
) -> Dict[str, float]:
    """
    Calculate Lateral (Cross-Track) Error.
    
    Measures perpendicular distance from the predicted path to the ground
    truth path. Critical for lane-keeping and HD map alignment.
    
    Args:
        predicted_poses: Predicted poses (N, 3) for [x, y, z]
        ground_truth_poses: Ground truth poses (N, 3)
        lane_width: Lane width in meters for violation rate calculation
    
    Returns:
        Dictionary with lateral error metrics
        
    Example:
        >>> result = calculate_lateral_error(pred, gt, lane_width=3.5)
        >>> print(f"Lateral error: {result['mean']:.3f}m")
    """
    predicted_poses = np.array(predicted_poses, dtype=np.float64)
    ground_truth_poses = np.array(ground_truth_poses, dtype=np.float64)
    
    pred_positions = predicted_poses[:, :3]
    gt_positions = ground_truth_poses[:, :3]
    
    lateral_errors = []
    
    for i in range(len(pred_positions)):
        # Find closest point on GT trajectory
        if i == 0:
            # Use next point direction
            direction = gt_positions[i+1] - gt_positions[i]
        elif i == len(pred_positions) - 1:
            # Use previous point direction
            direction = gt_positions[i] - gt_positions[i-1]
        else:
            # Use average of neighboring directions
            direction = gt_positions[i+1] - gt_positions[i-1]
        
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Vector from GT to prediction
        offset = pred_positions[i] - gt_positions[i]
        
        # Project offset onto direction to get longitudinal component
        longitudinal = np.dot(offset, direction) * direction
        
        # Lateral component is perpendicular to direction
        lateral = offset - longitudinal
        lateral_error = np.linalg.norm(lateral[:2])  # Use only x-y for lateral
        
        lateral_errors.append(lateral_error)
    
    lateral_errors = np.array(lateral_errors)
    
    # Calculate lane violation rate
    lane_violations = np.sum(lateral_errors > lane_width / 2)
    violation_rate = lane_violations / len(lateral_errors)
    
    return {
        'mean': float(np.mean(lateral_errors)),
        'std': float(np.std(lateral_errors)),
        'max': float(np.max(lateral_errors)),
        'rmse': float(np.sqrt(np.mean(lateral_errors**2))),
        'median': float(np.median(lateral_errors)),
        'lane_violation_rate': float(violation_rate),
        'lane_violations': int(lane_violations)
    }


def calculate_longitudinal_error(
    predicted_poses: Union[np.ndarray, List],
    ground_truth_poses: Union[np.ndarray, List]
) -> Dict[str, float]:
    """
    Calculate Longitudinal (Along-Track) Error.
    
    Measures distance along the path direction. Indicates how far ahead
    or behind the prediction is compared to ground truth.
    
    Args:
        predicted_poses: Predicted poses (N, 3) for [x, y, z]
        ground_truth_poses: Ground truth poses (N, 3)
    
    Returns:
        Dictionary with longitudinal error metrics
    """
    predicted_poses = np.array(predicted_poses, dtype=np.float64)
    ground_truth_poses = np.array(ground_truth_poses, dtype=np.float64)
    
    pred_positions = predicted_poses[:, :3]
    gt_positions = ground_truth_poses[:, :3]
    
    longitudinal_errors = []
    
    for i in range(len(pred_positions)):
        if i == 0:
            direction = gt_positions[i+1] - gt_positions[i]
        elif i == len(pred_positions) - 1:
            direction = gt_positions[i] - gt_positions[i-1]
        else:
            direction = gt_positions[i+1] - gt_positions[i-1]
        
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Project offset onto direction
        offset = pred_positions[i] - gt_positions[i]
        longitudinal = np.dot(offset, direction)
        
        longitudinal_errors.append(longitudinal)
    
    longitudinal_errors = np.array(longitudinal_errors)
    
    return {
        'mean': float(np.mean(longitudinal_errors)),
        'std': float(np.std(longitudinal_errors)),
        'mean_abs': float(np.mean(np.abs(longitudinal_errors))),
        'rmse': float(np.sqrt(np.mean(longitudinal_errors**2))),
        'median': float(np.median(longitudinal_errors))
    }


def calculate_convergence_rate(
    position_errors: Union[np.ndarray, List],
    timestamps: Optional[Union[np.ndarray, List]] = None,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Calculate convergence rate and time to convergence.
    
    Measures how quickly the localization system converges to accurate
    estimates, important for initialization and recovery.
    
    Args:
        position_errors: Position errors over time (N,)
        timestamps: Timestamps in seconds (N,). If None, uses indices
        threshold: Convergence threshold in meters (e.g., 0.1m)
    
    Returns:
        Dictionary with convergence metrics
    """
    position_errors = np.array(position_errors, dtype=np.float64)
    
    if timestamps is None:
        timestamps = np.arange(len(position_errors))
    else:
        timestamps = np.array(timestamps, dtype=np.float64)
    
    # Find first time error drops below threshold
    converged_indices = np.where(position_errors < threshold)[0]
    
    if len(converged_indices) == 0:
        return {
            'converged': False,
            'time_to_convergence': None,
            'convergence_rate': 0.0
        }
    
    first_convergence = converged_indices[0]
    time_to_convergence = timestamps[first_convergence] - timestamps[0]
    
    # Calculate convergence rate (error reduction per second)
    if first_convergence > 0:
        initial_error = position_errors[0]
        converged_error = position_errors[first_convergence]
        convergence_rate = (initial_error - converged_error) / time_to_convergence
    else:
        convergence_rate = float('inf')
    
    return {
        'converged': True,
        'time_to_convergence': float(time_to_convergence),
        'convergence_rate': float(convergence_rate),
        'initial_error': float(position_errors[0]),
        'converged_error': float(position_errors[first_convergence])
    }


def calculate_localization_metrics(
    predicted_poses: Union[np.ndarray, List],
    ground_truth_poses: Union[np.ndarray, List],
    timestamps: Optional[Union[np.ndarray, List]] = None,
    lane_width: float = 3.5,
    align: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive localization metrics.
    
    Args:
        predicted_poses: Predicted poses (N, 3), (N, 4), or (N, 7)
        ground_truth_poses: Ground truth poses (N, 3), (N, 4), or (N, 7)
        timestamps: Optional timestamps for convergence analysis
        lane_width: Lane width for lateral error violation rate
        align: Whether to align trajectories for ATE
    
    Returns:
        Dictionary with all localization metrics
    """
    predicted_poses = np.array(predicted_poses, dtype=np.float64)
    ground_truth_poses = np.array(ground_truth_poses, dtype=np.float64)
    
    results = {}
    
    # Absolute Trajectory Error
    ate = calculate_ate(predicted_poses, ground_truth_poses, align=align)
    results.update({f'ate_{k}': v for k, v in ate.items()})
    
    # Lateral Error
    lateral = calculate_lateral_error(predicted_poses, ground_truth_poses, lane_width)
    results.update({f'lateral_{k}': v for k, v in lateral.items()})
    
    # Longitudinal Error
    longitudinal = calculate_longitudinal_error(predicted_poses, ground_truth_poses)
    results.update({f'longitudinal_{k}': v for k, v in longitudinal.items()})
    
    # Rotation Error (if orientation available)
    if predicted_poses.shape[1] >= 4:
        are = calculate_are(predicted_poses, ground_truth_poses)
        results.update({f'are_{k}': v for k, v in are.items()})
    
    # Convergence (if timestamps available)
    if timestamps is not None:
        position_errors = np.linalg.norm(
            predicted_poses[:, :3] - ground_truth_poses[:, :3], axis=1
        )
        convergence = calculate_convergence_rate(position_errors, timestamps)
        results.update({f'convergence_{k}': v for k, v in convergence.items()})
    
    return results


def calculate_map_alignment_score(
    predicted_poses: Union[np.ndarray, List],
    hd_map_lanes: List[np.ndarray],
    max_distance: float = 2.0
) -> Dict[str, float]:
    """
    Calculate HD map alignment score.
    
    Measures how well the localization aligns with HD map features,
    particularly lane centerlines.
    
    Args:
        predicted_poses: Predicted poses (N, 3) for [x, y, z]
        hd_map_lanes: List of lane centerlines, each (M, 3)
        max_distance: Maximum distance for considering alignment (meters)
    
    Returns:
        Dictionary with map alignment metrics
    """
    predicted_poses = np.array(predicted_poses, dtype=np.float64)
    pred_positions = predicted_poses[:, :3]
    
    alignment_errors = []
    
    for pos in pred_positions:
        # Find minimum distance to any lane centerline
        min_dist = float('inf')
        
        for lane in hd_map_lanes:
            lane = np.array(lane, dtype=np.float64)
            # Distance to each point on lane
            dists = np.linalg.norm(lane - pos, axis=1)
            lane_min_dist = np.min(dists)
            
            if lane_min_dist < min_dist:
                min_dist = lane_min_dist
        
        alignment_errors.append(min_dist)
    
    alignment_errors = np.array(alignment_errors)
    
    # Calculate alignment rate (within max_distance)
    aligned_count = np.sum(alignment_errors <= max_distance)
    alignment_rate = aligned_count / len(alignment_errors)
    
    return {
        'mean_distance_to_lane': float(np.mean(alignment_errors)),
        'median_distance_to_lane': float(np.median(alignment_errors)),
        'max_distance_to_lane': float(np.max(alignment_errors)),
        'alignment_rate': float(alignment_rate),
        'aligned_poses': int(aligned_count),
        'total_poses': len(alignment_errors)
    }


# Helper functions

def _align_trajectories(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> np.ndarray:
    """
    Align predicted trajectory to ground truth using Umeyama algorithm.
    
    This is useful for SLAM evaluation where absolute position doesn't matter.
    """
    # Center the point clouds
    pred_mean = np.mean(predicted, axis=0)
    gt_mean = np.mean(ground_truth, axis=0)
    
    pred_centered = predicted - pred_mean
    gt_centered = ground_truth - gt_mean
    
    # Compute covariance matrix
    H = pred_centered.T @ gt_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    scale = S.sum() / np.sum(pred_centered**2)
    
    # Apply transformation
    aligned = scale * (predicted @ R.T) + gt_mean - scale * (pred_mean @ R.T)
    
    return aligned


def _quaternion_to_yaw(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to yaw angle.
    
    Args:
        quaternions: (N, 4) array of [qw, qx, qy, qz]
    
    Returns:
        (N,) array of yaw angles in radians
    """
    qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Yaw from quaternion
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    
    return yaw


def _normalize_angles(angles: np.ndarray) -> np.ndarray:
    """Normalize angles to [-pi, pi]."""
    return np.arctan2(np.sin(angles), np.cos(angles))
