"""
Average Orientation Similarity (AOS) metric for KITTI benchmark.

AOS accounts for both detection accuracy and orientation estimation.
"""

import numpy as np
from typing import List, Dict, Tuple


def calculate_aos(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.7,
    num_recall_points: int = 40
) -> Dict[str, float]:
    """
    Calculate Average Orientation Similarity (AOS) for KITTI.
    
    AOS = 1/N * sum(s_i * δ(o_i, ô_i))
    
    where:
    - s_i is the orientation similarity
    - δ is the detection indicator (1 if IoU >= threshold)
    - o_i, ô_i are ground truth and predicted orientations
    
    Args:
        predictions: List of prediction dicts with keys:
            - 'box': [x, y, z, w, h, l, yaw]
            - 'score': confidence score
            - 'class': class name
        ground_truth: List of ground truth dicts
        iou_threshold: IoU threshold for detection
        num_recall_points: Number of recall points
    
    Returns:
        Dictionary with:
            - 'aos': Average Orientation Similarity
            - 'ap': Average Precision
            - 'orientation_similarity': Mean orientation similarity for TPs
            
    Example:
        >>> result = calculate_aos(predictions, ground_truth)
        >>> print(f"AOS: {result['aos']:.4f}")
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Track matching and orientation similarity
    gt_matched = [False] * len(ground_truth)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    orientation_similarity = np.zeros(len(predictions))
    scores = np.array([p['score'] for p in predictions])
    
    # Match predictions to ground truth
    for pred_idx, pred in enumerate(predictions):
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if pred.get('class') != gt.get('class'):
                continue
            
            if gt_matched[gt_idx]:
                continue
            
            iou = calculate_iou_3d(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Calculate orientation similarity
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[max_gt_idx] = True
            
            # Calculate orientation similarity
            pred_yaw = pred['box'][6]
            gt_yaw = ground_truth[max_gt_idx]['box'][6]
            
            sim = calculate_orientation_similarity(pred_yaw, gt_yaw)
            orientation_similarity[pred_idx] = sim
        else:
            fp[pred_idx] = 1
            orientation_similarity[pred_idx] = 0
    
    # Compute cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute recall and precision weighted by orientation similarity
    num_gt = len(ground_truth)
    
    # Standard recall and precision
    recalls = tp_cumsum / max(num_gt, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Orientation-weighted precision
    orientation_cumsum = np.cumsum(orientation_similarity)
    orientation_precisions = orientation_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Compute AOS using interpolation
    aos = _compute_aos_interp(recalls, orientation_precisions, num_recall_points)
    
    # Compute standard AP for comparison
    ap = _compute_aos_interp(recalls, precisions, num_recall_points)
    
    # Calculate mean orientation similarity for true positives
    mean_orientation_sim = np.mean(orientation_similarity[tp == 1]) if np.sum(tp) > 0 else 0.0
    
    return {
        'aos': float(aos),
        'ap': float(ap),
        'orientation_similarity': float(mean_orientation_sim),
        'num_tp': int(np.sum(tp)),
        'num_fp': int(np.sum(fp)),
        'num_gt': num_gt
    }


def calculate_orientation_similarity(
    pred_yaw: float,
    gt_yaw: float
) -> float:
    """
    Calculate orientation similarity between predicted and ground truth yaw.
    
    Similarity = (1 + cos(Δθ)) / 2
    
    Args:
        pred_yaw: Predicted yaw angle in radians
        gt_yaw: Ground truth yaw angle in radians
    
    Returns:
        Similarity value between 0 and 1
    """
    # Normalize angle difference to [-pi, pi]
    delta = pred_yaw - gt_yaw
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    
    # Calculate similarity
    similarity = (1 + np.cos(delta)) / 2
    
    return float(similarity)


def _compute_aos_interp(
    recalls: np.ndarray,
    precisions: np.ndarray,
    num_points: int = 40
) -> float:
    """
    Compute AOS using interpolation (same as AP calculation).
    """
    if len(recalls) == 0:
        return 0.0
    
    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Interpolate
    recall_points = np.linspace(0, 1, num_points)
    aos = 0.0
    
    for r in recall_points:
        idx = np.where(recalls >= r)[0]
        if len(idx) > 0:
            aos += precisions[idx[0]]
    
    aos = aos / num_points
    
    return float(aos)


def calculate_aos_per_difficulty(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.7
) -> Dict[str, Dict[str, float]]:
    """
    Calculate AOS for different difficulty levels (KITTI Easy/Moderate/Hard).
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth with 'difficulty' field
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with AOS per difficulty level:
            - 'easy': AOS for easy samples
            - 'moderate': AOS for moderate samples
            - 'hard': AOS for hard samples
    """
    difficulties = ['easy', 'moderate', 'hard']
    results = {}
    
    for difficulty in difficulties:
        # Filter ground truth by difficulty
        filtered_gt = [gt for gt in ground_truth if gt.get('difficulty') == difficulty]
        
        if len(filtered_gt) == 0:
            results[difficulty] = {
                'aos': 0.0,
                'ap': 0.0,
                'orientation_similarity': 0.0
            }
            continue
        
        # Calculate AOS for this difficulty
        aos_result = calculate_aos(
            predictions=predictions,
            ground_truth=filtered_gt,
            iou_threshold=iou_threshold
        )
        
        results[difficulty] = aos_result
    
    return results


def calculate_orientation_error(
    pred_yaw: float,
    gt_yaw: float,
    error_type: str = "absolute"
) -> float:
    """
    Calculate orientation error.
    
    Args:
        pred_yaw: Predicted yaw in radians
        gt_yaw: Ground truth yaw in radians
        error_type: 'absolute' for absolute error, 'similarity' for similarity score
    
    Returns:
        Orientation error or similarity
    """
    # Normalize angle difference to [-pi, pi]
    delta = pred_yaw - gt_yaw
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    
    if error_type == "absolute":
        return float(abs(delta))
    elif error_type == "similarity":
        return calculate_orientation_similarity(pred_yaw, gt_yaw)
    else:
        raise ValueError(f"Unknown error_type: {error_type}")
