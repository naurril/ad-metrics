"""
Distance-based metrics for 3D object detection.
"""

import numpy as np
from typing import List, Dict, Union, Tuple


def calculate_center_distance(
    box1: Union[np.ndarray, List[float]],
    box2: Union[np.ndarray, List[float]],
    distance_type: str = "euclidean"
) -> float:
    """
    Calculate distance between centers of two 3D bounding boxes.
    
    Args:
        box1: First box [x, y, z, w, h, l, r]
        box2: Second box [x, y, z, w, h, l, r]
        distance_type: 'euclidean' (3D), 'bev' (2D x-y), or 'vertical' (z-axis only)
    
    Returns:
        Distance value in meters
        
    Example:
        >>> box1 = [0, 0, 0, 4, 2, 1.5, 0]
        >>> box2 = [3, 4, 0, 4, 2, 1.5, 0]
        >>> dist = calculate_center_distance(box1, box2)
        >>> print(f"Distance: {dist:.2f}m")
    """
    box1 = np.array(box1, dtype=np.float64)
    box2 = np.array(box2, dtype=np.float64)
    
    center1 = box1[:3]
    center2 = box2[:3]
    
    if distance_type == "euclidean":
        # 3D Euclidean distance
        dist = np.linalg.norm(center1 - center2)
    elif distance_type == "bev":
        # Bird's eye view distance (x-y plane only)
        dist = np.linalg.norm(center1[:2] - center2[:2])
    elif distance_type == "vertical":
        # Vertical distance (z-axis only)
        dist = abs(center1[2] - center2[2])
    else:
        raise ValueError(f"Unknown distance_type: {distance_type}")
    
    return float(dist)


def calculate_orientation_error(
    box1: Union[np.ndarray, List[float]],
    box2: Union[np.ndarray, List[float]],
    error_type: str = "absolute"
) -> float:
    """
    Calculate orientation/heading error between two boxes.
    
    Args:
        box1: First box [x, y, z, w, h, l, r]
        box2: Second box [x, y, z, w, h, l, r]
        error_type: 'absolute' for absolute error in radians, 'degrees' for degrees
    
    Returns:
        Orientation error
        
    Example:
        >>> box1 = [0, 0, 0, 4, 2, 1.5, 0]
        >>> box2 = [0, 0, 0, 4, 2, 1.5, np.pi/4]
        >>> error = calculate_orientation_error(box1, box2)
        >>> print(f"Orientation error: {error:.3f} rad")
    """
    box1 = np.array(box1, dtype=np.float64)
    box2 = np.array(box2, dtype=np.float64)
    
    yaw1 = box1[6]
    yaw2 = box2[6]
    
    # Normalize angle difference to [-pi, pi]
    delta = yaw1 - yaw2
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    
    if error_type == "absolute":
        error = abs(delta)
    elif error_type == "degrees":
        error = abs(np.degrees(delta))
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
    
    return float(error)


def calculate_size_error(
    box1: Union[np.ndarray, List[float]],
    box2: Union[np.ndarray, List[float]],
    box_format: str = "xyzwhlr",
    error_type: str = "absolute"
) -> Dict[str, float]:
    """
    Calculate size/dimension errors between two boxes.
    
    Args:
        box1: First box
        box2: Second box
        box_format: Box format
        error_type: 'absolute', 'relative', or 'percentage'
    
    Returns:
        Dictionary with errors for width, height, length, and volume
    """
    box1 = np.array(box1, dtype=np.float64)
    box2 = np.array(box2, dtype=np.float64)
    
    if box_format == "xyzwhlr":
        w1, h1, l1 = box1[3], box1[4], box1[5]
        w2, h2, l2 = box2[3], box2[4], box2[5]
    else:  # xyzhwlr
        h1, w1, l1 = box1[3], box1[4], box1[5]
        h2, w2, l2 = box2[3], box2[4], box2[5]
    
    if error_type == "absolute":
        w_error = abs(w1 - w2)
        h_error = abs(h1 - h2)
        l_error = abs(l1 - l2)
    elif error_type == "relative":
        w_error = (w1 - w2) / max(w2, 1e-6)
        h_error = (h1 - h2) / max(h2, 1e-6)
        l_error = (l1 - l2) / max(l2, 1e-6)
    elif error_type == "percentage":
        w_error = abs((w1 - w2) / max(w2, 1e-6)) * 100
        h_error = abs((h1 - h2) / max(h2, 1e-6)) * 100
        l_error = abs((l1 - l2) / max(l2, 1e-6)) * 100
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
    
    vol1 = w1 * h1 * l1
    vol2 = w2 * h2 * l2
    
    if error_type == "absolute":
        vol_error = abs(vol1 - vol2)
    elif error_type in ["relative", "percentage"]:
        vol_error = abs((vol1 - vol2) / max(vol2, 1e-6))
        if error_type == "percentage":
            vol_error *= 100
    
    return {
        'width_error': float(w_error),
        'height_error': float(h_error),
        'length_error': float(l_error),
        'volume_error': float(vol_error)
    }


def calculate_velocity_error(
    pred_velocity: Union[np.ndarray, List[float]],
    gt_velocity: Union[np.ndarray, List[float]],
    error_type: str = "euclidean"
) -> float:
    """
    Calculate velocity error.
    
    Args:
        pred_velocity: Predicted velocity [vx, vy] or [vx, vy, vz]
        gt_velocity: Ground truth velocity
        error_type: 'euclidean', 'manhattan', or 'angular'
    
    Returns:
        Velocity error in m/s
    """
    pred_vel = np.array(pred_velocity, dtype=np.float64)
    gt_vel = np.array(gt_velocity, dtype=np.float64)
    
    if error_type == "euclidean":
        error = np.linalg.norm(pred_vel - gt_vel)
    elif error_type == "manhattan":
        error = np.sum(np.abs(pred_vel - gt_vel))
    elif error_type == "angular":
        # Angular difference between velocity vectors
        dot_product = np.dot(pred_vel, gt_vel)
        norm_product = np.linalg.norm(pred_vel) * np.linalg.norm(gt_vel)
        if norm_product > 1e-6:
            cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
            error = np.arccos(cos_angle)
        else:
            error = 0.0
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
    
    return float(error)


def calculate_average_distance_error(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    distance_type: str = "euclidean"
) -> Dict[str, float]:
    """
    Calculate average distance error for matched detections.
    
    Args:
        predictions: List of predictions with 'box' and 'score'
        ground_truth: List of ground truth with 'box'
        iou_threshold: IoU threshold for matching
        distance_type: Type of distance metric
    
    Returns:
        Dictionary with:
            - 'mean_distance_error': Average distance error
            - 'median_distance_error': Median distance error
            - 'std_distance_error': Standard deviation
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
    
    # Match predictions to ground truth
    gt_matched = [False] * len(ground_truth)
    distance_errors = []
    
    for pred in predictions:
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            
            if pred.get('class') != gt.get('class'):
                continue
            
            iou = calculate_iou_3d(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            gt_matched[max_gt_idx] = True
            
            # Calculate distance error
            dist = calculate_center_distance(
                pred['box'],
                ground_truth[max_gt_idx]['box'],
                distance_type=distance_type
            )
            distance_errors.append(dist)
    
    if len(distance_errors) == 0:
        return {
            'mean_distance_error': float('nan'),
            'median_distance_error': float('nan'),
            'std_distance_error': float('nan'),
            'num_matched': 0
        }
    
    distance_errors = np.array(distance_errors)
    
    return {
        'mean_distance_error': float(np.mean(distance_errors)),
        'median_distance_error': float(np.median(distance_errors)),
        'std_distance_error': float(np.std(distance_errors)),
        'num_matched': len(distance_errors)
    }


def calculate_translation_error_bins(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    bins: List[float] = None
) -> Dict[str, Dict[str, int]]:
    """
    Calculate translation errors binned by distance ranges.
    
    Useful for analyzing performance at different distances.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
        bins: Distance bins in meters (e.g., [0, 10, 30, 50, 100])
    
    Returns:
        Dictionary with TP/FP counts per distance bin
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    if bins is None:
        bins = [0, 10, 30, 50, 100]
    
    # Initialize bins
    bin_results = {}
    for i in range(len(bins) - 1):
        bin_name = f"{bins[i]}-{bins[i+1]}m"
        bin_results[bin_name] = {'tp': 0, 'fp': 0, 'fn': 0}
    bin_results[f">{bins[-1]}m"] = {'tp': 0, 'fp': 0, 'fn': 0}
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
    
    # Track matched ground truth
    gt_matched = [False] * len(ground_truth)
    
    # Match predictions
    for pred in predictions:
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            
            if pred.get('class') != gt.get('class'):
                continue
            
            iou = calculate_iou_3d(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Determine distance bin based on prediction
        pred_dist = np.linalg.norm(pred['box'][:2])  # BEV distance from origin
        
        bin_name = None
        for i in range(len(bins) - 1):
            if bins[i] <= pred_dist < bins[i + 1]:
                bin_name = f"{bins[i]}-{bins[i+1]}m"
                break
        if bin_name is None and pred_dist >= bins[-1]:
            bin_name = f">{bins[-1]}m"

        if bin_name is None:
            continue
        
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            bin_results[bin_name]['tp'] += 1
            gt_matched[max_gt_idx] = True
        else:
            bin_results[bin_name]['fp'] += 1
    
    # Count false negatives
    for gt_idx, gt in enumerate(ground_truth):
        if not gt_matched[gt_idx]:
            gt_dist = np.linalg.norm(gt['box'][:2])
            
            bin_name = None
            for i in range(len(bins) - 1):
                if bins[i] <= gt_dist < bins[i + 1]:
                    bin_name = f"{bins[i]}-{bins[i+1]}m"
                    break
            if bin_name is None and gt_dist >= bins[-1]:
                bin_name = f">{bins[-1]}m"

            if bin_name:
                bin_results[bin_name]['fn'] += 1
    
    return bin_results
