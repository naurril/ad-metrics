"""
Detection matching algorithms for assigning predictions to ground truth.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment


def match_detections(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    method: str = "greedy",
    metric_type: str = "3d"
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predictions to ground truth boxes.
    
    Args:
        predictions: List of prediction dicts
        ground_truth: List of ground truth dicts
        iou_threshold: Minimum IoU for a match
        method: Matching method ('greedy' or 'hungarian')
        metric_type: '3d' or 'bev' for IoU calculation
    
    Returns:
        Tuple of:
            - matches: List of (pred_idx, gt_idx) tuples
            - unmatched_preds: List of unmatched prediction indices
            - unmatched_gts: List of unmatched ground truth indices
    """
    if method == "greedy":
        return greedy_matching(predictions, ground_truth, iou_threshold, metric_type)
    elif method == "hungarian":
        return hungarian_matching(predictions, ground_truth, iou_threshold, metric_type)
    else:
        raise ValueError(f"Unknown matching method: {method}")


def greedy_matching(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    metric_type: str = "3d"
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy matching algorithm (highest score first).
    
    Matches predictions to ground truth in order of confidence score,
    assigning each prediction to the highest IoU ground truth box.
    
    Args:
        predictions: List of predictions sorted by score (descending)
        ground_truth: List of ground truth
        iou_threshold: Minimum IoU threshold
        metric_type: '3d' or 'bev'
    
    Returns:
        Tuple of (matches, unmatched_preds, unmatched_gts)
    """
    from admetrics.detection.iou import calculate_iou_3d, calculate_iou_bev
    
    iou_func = calculate_iou_3d if metric_type == "3d" else calculate_iou_bev
    
    # Sort predictions by score
    pred_scores = np.array([p.get('score', 0) for p in predictions])
    sorted_indices = np.argsort(-pred_scores)
    
    gt_matched = [False] * len(ground_truth)
    matches = []
    unmatched_preds = []
    
    for pred_idx in sorted_indices:
        pred = predictions[pred_idx]
        
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            # Skip if already matched
            if gt_matched[gt_idx]:
                continue
            
            # Check class match
            if pred.get('class') != gt.get('class'):
                continue
            
            # Calculate IoU
            iou = iou_func(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Create match if above threshold
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            matches.append((pred_idx, max_gt_idx))
            gt_matched[max_gt_idx] = True
        else:
            unmatched_preds.append(pred_idx)
    
    # Find unmatched ground truth
    unmatched_gts = [i for i, matched in enumerate(gt_matched) if not matched]
    
    return matches, unmatched_preds, unmatched_gts


def hungarian_matching(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    metric_type: str = "3d"
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian algorithm (optimal assignment) for matching.
    
    Finds the optimal assignment that maximizes total IoU.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: Minimum IoU threshold
        metric_type: '3d' or 'bev'
    
    Returns:
        Tuple of (matches, unmatched_preds, unmatched_gts)
    """
    from admetrics.detection.iou import calculate_iou_batch
    
    if len(predictions) == 0 or len(ground_truth) == 0:
        return [], list(range(len(predictions))), list(range(len(ground_truth)))
    
    # Build IoU matrix
    pred_boxes = np.array([p['box'] for p in predictions])
    gt_boxes = np.array([g['box'] for g in ground_truth])
    
    mode = "3d" if metric_type == "3d" else "bev"
    iou_matrix = calculate_iou_batch(pred_boxes, gt_boxes, mode=mode)
    
    # Apply class filtering
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            if pred.get('class') != gt.get('class'):
                iou_matrix[i, j] = 0
    
    # Convert to cost matrix (maximize IoU = minimize negative IoU)
    cost_matrix = -iou_matrix
    
    # Solve assignment problem
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Filter by threshold
    matches = []
    unmatched_preds = []
    unmatched_gts = []
    
    matched_preds = set()
    matched_gts = set()
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            matches.append((int(pred_idx), int(gt_idx)))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)
    
    # Find unmatched
    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(ground_truth)) if i not in matched_gts]
    
    return matches, unmatched_preds, unmatched_gts


def match_by_center_distance(
    predictions: List[Dict],
    ground_truth: List[Dict],
    distance_threshold: float = 2.0,
    method: str = "greedy"
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match detections by center distance instead of IoU.
    
    Useful for very small objects where IoU might be unreliable.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        distance_threshold: Maximum distance for a match (meters)
        method: 'greedy' or 'hungarian'
    
    Returns:
        Tuple of (matches, unmatched_preds, unmatched_gts)
    """
    from admetrics.detection.distance import calculate_center_distance
    
    if method == "greedy":
        # Sort by score
        pred_scores = np.array([p.get('score', 0) for p in predictions])
        sorted_indices = np.argsort(-pred_scores)
        
        gt_matched = [False] * len(ground_truth)
        matches = []
        unmatched_preds = []
        
        for pred_idx in sorted_indices:
            pred = predictions[pred_idx]
            
            min_dist = float('inf')
            min_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                if pred.get('class') != gt.get('class'):
                    continue
                
                dist = calculate_center_distance(pred['box'], gt['box'], 'euclidean')
                
                if dist < min_dist:
                    min_dist = dist
                    min_gt_idx = gt_idx
            
            if min_dist <= distance_threshold and min_gt_idx >= 0:
                matches.append((pred_idx, min_gt_idx))
                gt_matched[min_gt_idx] = True
            else:
                unmatched_preds.append(pred_idx)
        
        unmatched_gts = [i for i, matched in enumerate(gt_matched) if not matched]
        
        return matches, unmatched_preds, unmatched_gts
    
    elif method == "hungarian":
        # Build distance matrix
        n_pred = len(predictions)
        n_gt = len(ground_truth)
        
        if n_pred == 0 or n_gt == 0:
            return [], list(range(n_pred)), list(range(n_gt))
        
        dist_matrix = np.full((n_pred, n_gt), float('inf'))
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                if pred.get('class') != gt.get('class'):
                    continue
                
                dist = calculate_center_distance(pred['box'], gt['box'], 'euclidean')
                dist_matrix[i, j] = dist
        
        # Solve assignment
        pred_indices, gt_indices = linear_sum_assignment(dist_matrix)
        
        matches = []
        matched_preds = set()
        matched_gts = set()
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            dist = dist_matrix[pred_idx, gt_idx]
            if dist <= distance_threshold:
                matches.append((int(pred_idx), int(gt_idx)))
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)
        
        unmatched_preds = [i for i in range(n_pred) if i not in matched_preds]
        unmatched_gts = [i for i in range(n_gt) if i not in matched_gts]
        
        return matches, unmatched_preds, unmatched_gts
    
    else:
        raise ValueError(f"Unknown method: {method}")


def filter_matches_by_class(
    matches: List[Tuple[int, int]],
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> List[Tuple[int, int]]:
    """
    Filter matches to only include same-class matches.
    
    Args:
        matches: List of (pred_idx, gt_idx) tuples
        predictions: List of predictions
        ground_truth: List of ground truth
    
    Returns:
        Filtered matches
    """
    filtered = []
    for pred_idx, gt_idx in matches:
        if predictions[pred_idx].get('class') == ground_truth[gt_idx].get('class'):
            filtered.append((pred_idx, gt_idx))
    return filtered
