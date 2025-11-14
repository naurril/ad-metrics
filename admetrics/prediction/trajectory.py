"""
Trajectory prediction metrics for evaluating future path forecasting.

These metrics assess the quality of predicted future trajectories compared to
ground truth trajectories. Common in autonomous driving and motion forecasting.
"""

import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from collections import defaultdict


def calculate_ade(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List]
) -> float:
    """
    Calculate Average Displacement Error (ADE).
    
    ADE is the mean Euclidean distance between predicted and ground truth
    positions across all time steps.
    
    Args:
        predictions: Predicted trajectory (T, 2) or (T, 3) for x,y or x,y,z
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
    
    Returns:
        Average displacement error in meters
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1], [2, 2]])
        >>> gt = np.array([[0, 0], [1.1, 0.9], [2.2, 1.8]])
        >>> ade = calculate_ade(pred, gt)
    """
    predictions = np.array(predictions, dtype=np.float64)
    ground_truth = np.array(ground_truth, dtype=np.float64)
    
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )
    
    # Calculate Euclidean distance at each timestep
    displacements = np.linalg.norm(predictions - ground_truth, axis=1)
    
    return float(np.mean(displacements))


def calculate_fde(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List]
) -> float:
    """
    Calculate Final Displacement Error (FDE).
    
    FDE is the Euclidean distance between the final predicted position
    and the final ground truth position.
    
    Args:
        predictions: Predicted trajectory (T, 2) or (T, 3)
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
    
    Returns:
        Final displacement error in meters
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1], [2, 2]])
        >>> gt = np.array([[0, 0], [1.1, 0.9], [2.2, 1.8]])
        >>> fde = calculate_fde(pred, gt)
    """
    predictions = np.array(predictions, dtype=np.float64)
    ground_truth = np.array(ground_truth, dtype=np.float64)
    
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )
    
    # Distance at final timestep
    final_displacement = np.linalg.norm(predictions[-1] - ground_truth[-1])
    
    return float(final_displacement)


def calculate_miss_rate(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List],
    threshold: float = 2.0
) -> Dict[str, float]:
    """
    Calculate Miss Rate (MR).
    
    Miss Rate is the fraction of predictions where the final displacement
    exceeds a threshold (typically 2 meters).
    
    Args:
        predictions: Predicted trajectory (T, 2) or (T, 3)
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
        threshold: Distance threshold for considering a prediction as "miss"
    
    Returns:
        Dictionary with miss_rate and is_miss flag
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1], [5, 5]])
        >>> gt = np.array([[0, 0], [1.1, 0.9], [2.2, 1.8]])
        >>> result = calculate_miss_rate(pred, gt, threshold=2.0)
    """
    fde = calculate_fde(predictions, ground_truth)
    is_miss = fde > threshold
    
    return {
        'miss_rate': float(is_miss),
        'fde': fde,
        'threshold': threshold,
        'is_miss': bool(is_miss)
    }


def calculate_multimodal_ade(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List],
    mode: str = "min"
) -> Dict[str, float]:
    """
    Calculate ADE for multi-modal predictions.
    
    Multi-modal predictors output K possible future trajectories.
    Common evaluation uses the best (minimum) ADE among all modes.
    
    Args:
        predictions: Predicted trajectories (K, T, 2) or (K, T, 3)
                    K modes, T timesteps, 2 or 3 dimensions
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
        mode: "min" (best), "mean" (average), or "all" (return all)
    
    Returns:
        Dictionary with ADE metrics
        
    Example:
        >>> # 3 possible futures, 5 timesteps, 2D
        >>> preds = np.random.randn(3, 5, 2)
        >>> gt = np.random.randn(5, 2)
        >>> result = calculate_multimodal_ade(preds, gt, mode="min")
    """
    predictions = np.array(predictions, dtype=np.float64)
    ground_truth = np.array(ground_truth, dtype=np.float64)
    
    if predictions.ndim != 3:
        raise ValueError(
            f"Predictions must be 3D (K, T, D), got shape {predictions.shape}"
        )
    
    # Calculate ADE for each mode
    num_modes = predictions.shape[0]
    ades = []
    
    for k in range(num_modes):
        ade = calculate_ade(predictions[k], ground_truth)
        ades.append(ade)
    
    ades = np.array(ades)
    
    result = {
        'min_ade': float(np.min(ades)),
        'mean_ade': float(np.mean(ades)),
        'best_mode': int(np.argmin(ades)),
        'num_modes': num_modes
    }
    
    if mode == "all":
        result['all_ades'] = ades.tolist()
    
    return result


def calculate_multimodal_fde(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List],
    mode: str = "min"
) -> Dict[str, float]:
    """
    Calculate FDE for multi-modal predictions.
    
    Args:
        predictions: Predicted trajectories (K, T, 2) or (K, T, 3)
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
        mode: "min" (best), "mean" (average), or "all" (return all)
    
    Returns:
        Dictionary with FDE metrics
    """
    predictions = np.array(predictions, dtype=np.float64)
    ground_truth = np.array(ground_truth, dtype=np.float64)
    
    if predictions.ndim != 3:
        raise ValueError(
            f"Predictions must be 3D (K, T, D), got shape {predictions.shape}"
        )
    
    # Calculate FDE for each mode
    num_modes = predictions.shape[0]
    fdes = []
    
    for k in range(num_modes):
        fde = calculate_fde(predictions[k], ground_truth)
        fdes.append(fde)
    
    fdes = np.array(fdes)
    
    result = {
        'min_fde': float(np.min(fdes)),
        'mean_fde': float(np.mean(fdes)),
        'best_mode': int(np.argmin(fdes)),
        'num_modes': num_modes
    }
    
    if mode == "all":
        result['all_fdes'] = fdes.tolist()
    
    return result


def calculate_brier_fde(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List],
    probabilities: Optional[Union[np.ndarray, List]] = None
) -> Dict[str, float]:
    """
    Calculate Brier-FDE (probability-weighted FDE).
    
    For probabilistic multi-modal predictions, Brier-FDE weighs each mode's
    FDE by its predicted probability.
    
    Args:
        predictions: Predicted trajectories (K, T, 2) or (K, T, 3)
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
        probabilities: Probability for each mode (K,). If None, uniform weights
    
    Returns:
        Dictionary with Brier-FDE and related metrics
        
    Example:
        >>> preds = np.random.randn(3, 5, 2)
        >>> gt = np.random.randn(5, 2)
        >>> probs = np.array([0.5, 0.3, 0.2])  # Sum to 1
        >>> result = calculate_brier_fde(preds, gt, probs)
    """
    predictions = np.array(predictions, dtype=np.float64)
    ground_truth = np.array(ground_truth, dtype=np.float64)
    
    if predictions.ndim != 3:
        raise ValueError(
            f"Predictions must be 3D (K, T, D), got shape {predictions.shape}"
        )
    
    num_modes = predictions.shape[0]
    
    # Use uniform probabilities if not provided
    if probabilities is None:
        probabilities = np.ones(num_modes) / num_modes
    else:
        probabilities = np.array(probabilities, dtype=np.float64)
        # Normalize to sum to 1
        probabilities = probabilities / probabilities.sum()
    
    # Calculate FDE for each mode
    fdes = []
    for k in range(num_modes):
        fde = calculate_fde(predictions[k], ground_truth)
        fdes.append(fde)
    
    fdes = np.array(fdes)
    
    # Brier-FDE is probability-weighted average of FDEs
    brier_fde = float(np.sum(probabilities * fdes))
    
    return {
        'brier_fde': brier_fde,
        'min_fde': float(np.min(fdes)),
        'probabilities': probabilities.tolist()
    }


def calculate_nll(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List],
    covariances: Union[np.ndarray, List],
    probabilities: Optional[Union[np.ndarray, List]] = None
) -> Dict[str, float]:
    """
    Calculate Negative Log-Likelihood (NLL) for Gaussian mixture predictions.
    
    Evaluates the likelihood of ground truth under predicted Gaussian mixture.
    Lower NLL indicates better calibrated predictions.
    
    Args:
        predictions: Predicted trajectory means (K, T, 2) or (K, T, 3)
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
        covariances: Covariance matrices (K, T, D, D) or (K, T) for diagonal
        probabilities: Mixture weights (K,). If None, uniform
    
    Returns:
        Dictionary with NLL and related metrics
        
    Example:
        >>> preds = np.random.randn(3, 5, 2)
        >>> gt = np.random.randn(5, 2)
        >>> covs = np.tile(np.eye(2), (3, 5, 1, 1)) * 0.1  # (3, 5, 2, 2)
        >>> result = calculate_nll(preds, gt, covs)
    """
    predictions = np.array(predictions, dtype=np.float64)
    ground_truth = np.array(ground_truth, dtype=np.float64)
    covariances = np.array(covariances, dtype=np.float64)
    
    num_modes, num_timesteps, dims = predictions.shape
    
    # Use uniform probabilities if not provided
    if probabilities is None:
        probabilities = np.ones(num_modes) / num_modes
    else:
        probabilities = np.array(probabilities, dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
    
    # Handle different covariance formats
    if covariances.ndim == 2:
        # Diagonal covariances (K, T) - expand to full matrices
        covs_full = np.zeros((num_modes, num_timesteps, dims, dims))
        for k in range(num_modes):
            for t in range(num_timesteps):
                covs_full[k, t] = np.eye(dims) * covariances[k, t]
        covariances = covs_full
    
    # Calculate Gaussian log-likelihood for each mode
    log_likelihoods = []
    
    for k in range(num_modes):
        mode_log_likelihood = 0.0
        
        for t in range(num_timesteps):
            mean = predictions[k, t]
            cov = covariances[k, t]
            gt_point = ground_truth[t]
            
            # Multivariate Gaussian log-likelihood
            diff = gt_point - mean
            
            # Add small epsilon for numerical stability
            cov_stable = cov + np.eye(dims) * 1e-6
            
            try:
                # log|Σ|
                sign, logdet = np.linalg.slogdet(cov_stable)
                
                # (x-μ)ᵀ Σ⁻¹ (x-μ)
                cov_inv = np.linalg.inv(cov_stable)
                mahalanobis = diff @ cov_inv @ diff
                
                # -0.5 * (k*log(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ))
                log_likelihood = -0.5 * (dims * np.log(2 * np.pi) + logdet + mahalanobis)
                mode_log_likelihood += log_likelihood
                
            except np.linalg.LinAlgError:
                # Singular covariance matrix
                mode_log_likelihood += -1e10
        
        log_likelihoods.append(mode_log_likelihood)
    
    log_likelihoods = np.array(log_likelihoods)
    
    # Mixture log-likelihood: log(Σ π_k * p_k(x))
    # Use log-sum-exp trick for numerical stability
    log_probs = np.log(probabilities + 1e-10)
    log_mixture_components = log_probs + log_likelihoods
    max_log = np.max(log_mixture_components)
    
    log_likelihood = max_log + np.log(
        np.sum(np.exp(log_mixture_components - max_log))
    )
    
    nll = -log_likelihood
    
    return {
        'nll': float(nll),
        'log_likelihood': float(log_likelihood),
        'best_mode': int(np.argmax(log_likelihoods)),
        'num_modes': num_modes
    }


def calculate_trajectory_metrics(
    predictions: Union[np.ndarray, List],
    ground_truth: Union[np.ndarray, List],
    miss_threshold: float = 2.0,
    multimodal: bool = False
) -> Dict[str, float]:
    """
    Calculate comprehensive trajectory prediction metrics.
    
    Args:
        predictions: Predicted trajectory/trajectories
                    Single-modal: (T, 2) or (T, 3)
                    Multi-modal: (K, T, 2) or (K, T, 3)
        ground_truth: Ground truth trajectory (T, 2) or (T, 3)
        miss_threshold: Threshold for miss rate calculation
        multimodal: Whether predictions are multi-modal
    
    Returns:
        Dictionary with all relevant metrics
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1], [2, 2]])
        >>> gt = np.array([[0, 0], [1.1, 0.9], [2.1, 1.9]])
        >>> metrics = calculate_trajectory_metrics(pred, gt)
    """
    predictions = np.array(predictions, dtype=np.float64)
    ground_truth = np.array(ground_truth, dtype=np.float64)
    
    if multimodal:
        # Multi-modal predictions
        ade_result = calculate_multimodal_ade(predictions, ground_truth)
        fde_result = calculate_multimodal_fde(predictions, ground_truth)
        
        # Calculate miss rate for best mode
        best_mode = ade_result['best_mode']
        miss_result = calculate_miss_rate(
            predictions[best_mode],
            ground_truth,
            miss_threshold
        )
        
        return {
            'ade': ade_result['min_ade'],
            'fde': fde_result['min_fde'],
            'miss_rate': miss_result['miss_rate'],
            'mean_ade': ade_result['mean_ade'],
            'mean_fde': fde_result['mean_fde'],
            'num_modes': ade_result['num_modes'],
            'best_mode': best_mode
        }
    else:
        # Single-modal predictions
        ade = calculate_ade(predictions, ground_truth)
        fde = calculate_fde(predictions, ground_truth)
        miss_result = calculate_miss_rate(predictions, ground_truth, miss_threshold)
        
        return {
            'ade': ade,
            'fde': fde,
            'miss_rate': miss_result['miss_rate'],
            'is_miss': miss_result['is_miss']
        }


def calculate_collision_rate(
    predictions: Union[np.ndarray, List],
    obstacles: List[Dict],
    safety_margin: float = 0.5
) -> Dict[str, float]:
    """
    Calculate collision rate with static obstacles.
    
    Args:
        predictions: Predicted trajectory (T, 2) or (T, 3)
        obstacles: List of obstacles, each with 'center' and 'radius'
        safety_margin: Additional safety buffer in meters
    
    Returns:
        Dictionary with collision metrics
        
    Example:
        >>> pred = np.array([[0, 0], [1, 0], [2, 0]])
        >>> obstacles = [{'center': [1.5, 0], 'radius': 0.3}]
        >>> result = calculate_collision_rate(pred, obstacles, safety_margin=0.2)
    """
    predictions = np.array(predictions, dtype=np.float64)
    
    num_timesteps = len(predictions)
    num_collisions = 0
    collision_timesteps = []
    
    for t, point in enumerate(predictions):
        for obs in obstacles:
            obs_center = np.array(obs['center'][:len(point)])
            obs_radius = obs['radius'] + safety_margin
            
            distance = np.linalg.norm(point - obs_center)
            
            if distance < obs_radius:
                num_collisions += 1
                collision_timesteps.append(t)
                break  # Count each timestep once
    
    collision_rate = num_collisions / num_timesteps if num_timesteps > 0 else 0.0
    
    return {
        'collision_rate': float(collision_rate),
        'num_collisions': num_collisions,
        'collision_timesteps': collision_timesteps,
        'total_timesteps': num_timesteps
    }


def calculate_drivable_area_compliance(
    predictions: Union[np.ndarray, List],
    drivable_area: Dict
) -> Dict[str, float]:
    """
    Calculate compliance with drivable area constraints.
    
    Args:
        predictions: Predicted trajectory (T, 2) for x, y positions
        drivable_area: Dict with 'type' and parameters
                      'rectangle': {'x_min', 'x_max', 'y_min', 'y_max'}
                      'polygon': {'vertices': List of (x, y)}
    
    Returns:
        Dictionary with compliance metrics
        
    Example:
        >>> pred = np.array([[0, 0], [1, 1], [2, 2]])
        >>> area = {'type': 'rectangle', 'x_min': -1, 'x_max': 3,
        ...         'y_min': -1, 'y_max': 3}
        >>> result = calculate_drivable_area_compliance(pred, area)
    """
    predictions = np.array(predictions, dtype=np.float64)
    
    num_timesteps = len(predictions)
    num_violations = 0
    violation_timesteps = []
    
    area_type = drivable_area.get('type', 'rectangle')
    
    for t, point in enumerate(predictions):
        in_bounds = False
        
        if area_type == 'rectangle':
            x_min = drivable_area['x_min']
            x_max = drivable_area['x_max']
            y_min = drivable_area['y_min']
            y_max = drivable_area['y_max']
            
            in_bounds = (x_min <= point[0] <= x_max and
                        y_min <= point[1] <= y_max)
        
        elif area_type == 'polygon':
            # Point-in-polygon test using ray casting
            vertices = np.array(drivable_area['vertices'])
            in_bounds = _point_in_polygon(point[:2], vertices)
        
        if not in_bounds:
            num_violations += 1
            violation_timesteps.append(t)
    
    compliance_rate = 1.0 - (num_violations / num_timesteps) if num_timesteps > 0 else 1.0
    
    return {
        'compliance_rate': float(compliance_rate),
        'num_violations': num_violations,
        'violation_timesteps': violation_timesteps,
        'total_timesteps': num_timesteps
    }


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: Point coordinates (2,)
        polygon: Polygon vertices (N, 2)
    
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside
