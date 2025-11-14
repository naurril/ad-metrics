"""
Vector Map Detection and Evaluation Metrics.

Metrics for evaluating HD map vector extraction from sensor data, including:
- Lane line detection (centerlines, boundaries)
- Road boundary detection
- Crosswalk/pedestrian crossing detection
- Topology estimation (connectivity, lane relationships)

Used in: nuScenes Map, Argoverse 2 Map, OpenLane-V2, MapTR, HDMapNet, VectorMapNet
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import directed_hausdorff, cdist


def chamfer_distance_polyline(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    max_distance: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate Chamfer Distance between two polylines.
    
    Primary metric for polyline/curve matching in vector map evaluation.
    
    Args:
        pred_polyline: Predicted polyline points, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline points, shape (M, 2) or (M, 3)
        max_distance: Optional maximum distance threshold for filtering outliers
        
    Returns:
        Dictionary with:
            - chamfer_distance: Average bidirectional chamfer distance
            - chamfer_pred_to_gt: Pred -> GT distance
            - chamfer_gt_to_pred: GT -> Pred distance
            - precision: Fraction of pred points within threshold
            - recall: Fraction of GT points within threshold
            
    Example:
        >>> pred = np.array([[0, 0], [1, 0], [2, 0.1]])
        >>> gt = np.array([[0, 0], [1, 0], [2, 0]])
        >>> result = chamfer_distance_polyline(pred, gt)
        >>> print(f"Chamfer: {result['chamfer_distance']:.3f}m")
    """
    if len(pred_polyline) == 0 or len(gt_polyline) == 0:
        return {
            'chamfer_distance': float('inf'),
            'chamfer_pred_to_gt': float('inf'),
            'chamfer_gt_to_pred': float('inf'),
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Compute pairwise distances
    dist_matrix = cdist(pred_polyline, gt_polyline)
    
    # Pred to GT: for each predicted point, find nearest GT point
    min_dist_pred_to_gt = np.min(dist_matrix, axis=1)
    chamfer_pred_to_gt = np.mean(min_dist_pred_to_gt)
    
    # GT to Pred: for each GT point, find nearest predicted point
    min_dist_gt_to_pred = np.min(dist_matrix, axis=0)
    chamfer_gt_to_pred = np.mean(min_dist_gt_to_pred)
    
    # Symmetric Chamfer distance
    chamfer_distance = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2
    
    # Precision and recall (if threshold provided)
    precision = recall = None
    if max_distance is not None:
        precision = np.mean(min_dist_pred_to_gt <= max_distance)
        recall = np.mean(min_dist_gt_to_pred <= max_distance)
    
    return {
        'chamfer_distance': float(chamfer_distance),
        'chamfer_pred_to_gt': float(chamfer_pred_to_gt),
        'chamfer_gt_to_pred': float(chamfer_gt_to_pred),
        'precision': float(precision) if precision is not None else None,
        'recall': float(recall) if recall is not None else None
    }


def frechet_distance(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray
) -> float:
    """
    Calculate Fréchet Distance between two polylines.
    
    Measures similarity of curves considering ordering and continuity.
    Also known as "dog-leash distance" - better than Chamfer for curves.
    
    Args:
        pred_polyline: Predicted polyline points, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline points, shape (M, 2) or (M, 3)
        
    Returns:
        Fréchet distance (meters)
        
    Example:
        >>> pred = np.array([[0, 0], [1, 0], [2, 0]])
        >>> gt = np.array([[0, 0.1], [1, 0.1], [2, 0.1]])
        >>> dist = frechet_distance(pred, gt)
    """
    n = len(pred_polyline)
    m = len(gt_polyline)
    
    if n == 0 or m == 0:
        return float('inf')
    
    # Dynamic programming matrix
    ca = np.full((n, m), -1.0)
    
    def _compute_frechet(i: int, j: int) -> float:
        """Recursive computation with memoization."""
        if ca[i, j] > -1:
            return ca[i, j]
        
        # Distance between current points
        d = np.linalg.norm(pred_polyline[i] - gt_polyline[j])
        
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_compute_frechet(i-1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_compute_frechet(0, j-1), d)
        else:
            ca[i, j] = max(
                min(
                    _compute_frechet(i-1, j),
                    _compute_frechet(i-1, j-1),
                    _compute_frechet(i, j-1)
                ),
                d
            )
        
        return ca[i, j]
    
    return float(_compute_frechet(n-1, m-1))


def polyline_iou(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    width: float = 1.0,
    num_samples: int = 100
) -> float:
    """
    Calculate IoU between two polylines by treating them as thin rectangles.
    
    Useful for lane line matching where width represents lane width tolerance.
    
    Args:
        pred_polyline: Predicted polyline, shape (N, 2)
        gt_polyline: Ground truth polyline, shape (M, 2)
        width: Width of the polyline corridor (meters)
        num_samples: Number of points to sample along polylines for IoU calculation
        
    Returns:
        IoU score [0, 1]
        
    Example:
        >>> pred = np.array([[0, 0], [10, 0]])
        >>> gt = np.array([[0, 0.2], [10, 0.2]])
        >>> iou = polyline_iou(pred, gt, width=1.0)
    """
    if len(pred_polyline) < 2 or len(gt_polyline) < 2:
        return 0.0
    
    # Sample points uniformly along both polylines
    def sample_polyline(polyline, n_samples):
        """Sample n points uniformly along polyline."""
        # Calculate cumulative distances
        segments = np.diff(polyline, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]
        
        if total_length == 0:
            return polyline[:1]
        
        # Sample at uniform intervals
        sample_distances = np.linspace(0, total_length, n_samples)
        sampled_points = []
        
        for dist in sample_distances:
            # Find which segment this distance falls on
            seg_idx = np.searchsorted(cumulative_lengths, dist) - 1
            seg_idx = max(0, min(seg_idx, len(polyline) - 2))
            
            # Interpolate within segment
            seg_start_dist = cumulative_lengths[seg_idx]
            seg_length = segment_lengths[seg_idx]
            
            if seg_length > 0:
                t = (dist - seg_start_dist) / seg_length
                t = np.clip(t, 0, 1)
                point = polyline[seg_idx] + t * segments[seg_idx]
            else:
                point = polyline[seg_idx]
            
            sampled_points.append(point)
        
        return np.array(sampled_points)
    
    # Sample both polylines
    pred_samples = sample_polyline(pred_polyline, num_samples)
    gt_samples = sample_polyline(gt_polyline, num_samples)
    
    # Compute distances between all sample pairs
    dist_matrix = cdist(pred_samples, gt_samples)
    
    # Count intersection: points within width threshold
    pred_in_gt = np.any(dist_matrix <= width, axis=1)
    gt_in_pred = np.any(dist_matrix <= width, axis=0)
    
    intersection = np.sum(pred_in_gt) + np.sum(gt_in_pred)
    union = 2 * num_samples  # Both polylines sampled with num_samples points
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(np.clip(iou, 0, 1))


def lane_detection_metrics(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    distance_threshold: float = 1.5,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive lane detection evaluation metrics.
    
    Used in: OpenLane, OpenLane-V2, Argoverse 2 HD Map Challenge.
    
    Args:
        pred_lanes: List of predicted lane polylines, each shape (N, 2) or (N, 3)
        gt_lanes: List of ground truth lane polylines, each shape (M, 2) or (M, 3)
        distance_threshold: Maximum chamfer distance for matching (meters)
        iou_threshold: Minimum IoU for matching
        
    Returns:
        Dictionary with:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1_score: Harmonic mean of precision and recall
            - chamfer_distance: Mean chamfer distance for matched lanes
            - tp: Number of true positives
            - fp: Number of false positives
            - fn: Number of false negatives
            
    Example:
        >>> pred = [np.array([[0, 0], [10, 0]]), np.array([[0, 3], [10, 3]])]
        >>> gt = [np.array([[0, 0.1], [10, 0.1]]), np.array([[0, 3.1], [10, 3.1]])]
        >>> metrics = lane_detection_metrics(pred, gt)
    """
    if len(pred_lanes) == 0 and len(gt_lanes) == 0:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'chamfer_distance': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
    
    if len(gt_lanes) == 0:
        return {
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0,
            'chamfer_distance': float('inf'),
            'tp': 0,
            'fp': len(pred_lanes),
            'fn': 0
        }
    
    if len(pred_lanes) == 0:
        return {
            'precision': 1.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'chamfer_distance': float('inf'),
            'tp': 0,
            'fp': 0,
            'fn': len(gt_lanes)
        }
    
    # Compute pairwise Chamfer distances between all pred-GT lane pairs
    chamfer_matrix = np.zeros((len(pred_lanes), len(gt_lanes)))
    
    for i, pred_lane in enumerate(pred_lanes):
        for j, gt_lane in enumerate(gt_lanes):
            result = chamfer_distance_polyline(pred_lane, gt_lane)
            chamfer_matrix[i, j] = result['chamfer_distance']
    
    # Greedy matching: match pred to GT based on minimum Chamfer distance
    matched_pred = set()
    matched_gt = set()
    matched_distances = []
    
    # Sort all pairs by chamfer distance
    pairs = []
    for i in range(len(pred_lanes)):
        for j in range(len(gt_lanes)):
            pairs.append((chamfer_matrix[i, j], i, j))
    pairs.sort()
    
    # Greedily match
    for dist, pred_idx, gt_idx in pairs:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            if dist <= distance_threshold:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                matched_distances.append(dist)
    
    # Calculate metrics
    tp = len(matched_pred)
    fp = len(pred_lanes) - tp
    fn = len(gt_lanes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mean_chamfer = np.mean(matched_distances) if len(matched_distances) > 0 else float('inf')
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'chamfer_distance': float(mean_chamfer),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def topology_metrics(
    pred_topology: Dict[str, List[int]],
    gt_topology: Dict[str, List[int]],
    lane_matches: Dict[int, int]
) -> Dict[str, float]:
    """
    Evaluate lane topology/connectivity prediction.
    
    Measures how well the model predicts lane-to-lane relationships:
    - Lane successors (which lanes follow current lane)
    - Lane predecessors (which lanes lead to current lane)
    - Left/right neighbors
    
    Args:
        pred_topology: Predicted topology, e.g., {'successors': [lane_ids], 'left_neighbor': [id]}
        gt_topology: Ground truth topology, same format
        lane_matches: Mapping from predicted lane ID to GT lane ID
        
    Returns:
        Dictionary with:
            - topology_precision: Correct connections / predicted connections
            - topology_recall: Correct connections / GT connections
            - topology_f1: F1 score
            
    Example:
        >>> pred_topo = {'successors': [1, 2], 'left_neighbor': [3]}
        >>> gt_topo = {'successors': [1, 2], 'left_neighbor': [3, 4]}
        >>> matches = {0: 0, 1: 1, 2: 2, 3: 3}
        >>> metrics = topology_metrics(pred_topo, gt_topo, matches)
    """
    # Count correct and total topology connections
    correct_connections = 0
    pred_connections = 0
    gt_connections = 0
    
    # Check each topology type (successors, predecessors, left_neighbor, right_neighbor)
    topology_types = set(list(pred_topology.keys()) + list(gt_topology.keys()))
    
    for topo_type in topology_types:
        pred_ids = set(pred_topology.get(topo_type, []))
        gt_ids = set(gt_topology.get(topo_type, []))
        
        # Map predicted IDs to GT space
        mapped_pred_ids = set()
        for pred_id in pred_ids:
            if pred_id in lane_matches:
                mapped_pred_ids.add(lane_matches[pred_id])
        
        # Count matches
        correct = len(mapped_pred_ids & gt_ids)
        correct_connections += correct
        pred_connections += len(pred_ids)
        gt_connections += len(gt_ids)
    
    # Calculate metrics
    precision = correct_connections / pred_connections if pred_connections > 0 else 0.0
    recall = correct_connections / gt_connections if gt_connections > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'topology_precision': float(precision),
        'topology_recall': float(recall),
        'topology_f1': float(f1),
        'correct_connections': correct_connections,
        'pred_connections': pred_connections,
        'gt_connections': gt_connections
    }


def endpoint_error(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray
) -> Dict[str, float]:
    """
    Calculate error at polyline endpoints.
    
    Important for lane merging/splitting scenarios and topology.
    
    Args:
        pred_polyline: Predicted polyline, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline, shape (M, 2) or (M, 3)
        
    Returns:
        Dictionary with:
            - start_error: Distance between start points
            - end_error: Distance between end points
            - mean_endpoint_error: Average of start and end errors
            
    Example:
        >>> pred = np.array([[0, 0], [10, 0]])
        >>> gt = np.array([[0.1, 0], [10.1, 0]])
        >>> errors = endpoint_error(pred, gt)
    """
    if len(pred_polyline) == 0 or len(gt_polyline) == 0:
        return {
            'start_error': float('inf'),
            'end_error': float('inf'),
            'mean_endpoint_error': float('inf')
        }
    
    start_error = np.linalg.norm(pred_polyline[0] - gt_polyline[0])
    end_error = np.linalg.norm(pred_polyline[-1] - gt_polyline[-1])
    mean_error = (start_error + end_error) / 2
    
    return {
        'start_error': float(start_error),
        'end_error': float(end_error),
        'mean_endpoint_error': float(mean_error)
    }


def direction_accuracy(
    pred_polyline: np.ndarray,
    gt_polyline: np.ndarray,
    num_samples: int = 10
) -> Dict[str, float]:
    """
    Evaluate directional accuracy of polyline (tangent vector alignment).
    
    Measures whether the predicted lane follows the correct direction.
    
    Args:
        pred_polyline: Predicted polyline, shape (N, 2) or (N, 3)
        gt_polyline: Ground truth polyline, shape (M, 2) or (M, 3)
        num_samples: Number of points to sample for direction comparison
        
    Returns:
        Dictionary with:
            - mean_direction_error: Mean angle error in radians
            - mean_direction_error_deg: Mean angle error in degrees
            - direction_accuracy: Fraction of samples with error < 15°
            
    Example:
        >>> pred = np.array([[0, 0], [10, 1]])
        >>> gt = np.array([[0, 0], [10, 0]])
        >>> acc = direction_accuracy(pred, gt)
    """
    if len(pred_polyline) < 2 or len(gt_polyline) < 2:
        return {
            'mean_direction_error': float('inf'),
            'mean_direction_error_deg': float('inf'),
            'direction_accuracy': 0.0
        }
    
    # Sample points along pred polyline
    pred_length = np.sum(np.linalg.norm(np.diff(pred_polyline, axis=0), axis=1))
    gt_length = np.sum(np.linalg.norm(np.diff(gt_polyline, axis=0), axis=1))
    
    if pred_length == 0 or gt_length == 0:
        return {
            'mean_direction_error': float('inf'),
            'mean_direction_error_deg': float('inf'),
            'direction_accuracy': 0.0
        }
    
    # Get tangent vectors at sampled points
    direction_errors = []
    
    for i in range(num_samples):
        # Sample parameter t
        t = i / (num_samples - 1) if num_samples > 1 else 0.5
        
        # Get tangent at this point for both polylines
        pred_idx = int(t * (len(pred_polyline) - 1))
        gt_idx = int(t * (len(gt_polyline) - 1))
        
        pred_idx = min(pred_idx, len(pred_polyline) - 2)
        gt_idx = min(gt_idx, len(gt_polyline) - 2)
        
        # Tangent vectors
        pred_tangent = pred_polyline[pred_idx + 1] - pred_polyline[pred_idx]
        gt_tangent = gt_polyline[gt_idx + 1] - gt_polyline[gt_idx]
        
        # Normalize
        pred_tangent_norm = pred_tangent / (np.linalg.norm(pred_tangent) + 1e-8)
        gt_tangent_norm = gt_tangent / (np.linalg.norm(gt_tangent) + 1e-8)
        
        # Angle between vectors (handle 2D and 3D)
        dot_product = np.dot(pred_tangent_norm, gt_tangent_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_error = np.arccos(dot_product)
        
        direction_errors.append(angle_error)
    
    mean_error_rad = np.mean(direction_errors)
    mean_error_deg = np.degrees(mean_error_rad)
    
    # Accuracy: fraction within 15 degrees
    accuracy = np.mean([err < np.radians(15) for err in direction_errors])
    
    return {
        'mean_direction_error': float(mean_error_rad),
        'mean_direction_error_deg': float(mean_error_deg),
        'direction_accuracy': float(accuracy)
    }


def vectormap_ap(
    pred_lanes: List[Dict],
    gt_lanes: List[Dict],
    distance_thresholds: List[float] = [0.5, 1.0, 1.5],
    confidence_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate Average Precision for vector map detection.
    
    Similar to object detection AP but using Chamfer distance for matching.
    
    Args:
        pred_lanes: List of predicted lane dicts with 'polyline' (np.ndarray) and optional 'score'
        gt_lanes: List of ground truth lane dicts with 'polyline' (np.ndarray)
        distance_thresholds: Chamfer distance thresholds for matching (meters)
        confidence_scores: Optional confidence scores for predictions (if not in pred_lanes)
        
    Returns:
        Dictionary with AP at different thresholds:
            - ap_0.5: AP at 0.5m threshold
            - ap_1.0: AP at 1.0m threshold
            - ap_1.5: AP at 1.5m threshold
            - map: Mean AP across all thresholds
            
    Example:
        >>> pred = [{'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9}]
        >>> gt = [{'polyline': np.array([[0, 0.1], [10, 0.1]])}]
        >>> ap = vectormap_ap(pred, gt)
    """
    results = {}
    
    # Extract confidence scores
    if confidence_scores is None:
        scores = [lane.get('score', 1.0) for lane in pred_lanes]
    else:
        scores = confidence_scores
    
    # Sort predictions by confidence (descending)
    if len(scores) > 0:
        sorted_indices = np.argsort(scores)[::-1]
        pred_lanes = [pred_lanes[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
    
    # Calculate AP for each threshold
    aps = []
    for threshold in distance_thresholds:
        # Track which GT lanes have been matched
        matched_gt = set()
        tp = []
        fp = []
        
        for pred_idx, pred_lane in enumerate(pred_lanes):
            pred_polyline = pred_lane['polyline']
            
            # Find best matching GT lane
            best_chamfer = float('inf')
            best_gt_idx = -1
            
            for gt_idx, gt_lane in enumerate(gt_lanes):
                if gt_idx in matched_gt:
                    continue
                
                gt_polyline = gt_lane['polyline']
                result = chamfer_distance_polyline(pred_polyline, gt_polyline)
                chamfer = result['chamfer_distance']
                
                if chamfer < best_chamfer:
                    best_chamfer = chamfer
                    best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_chamfer <= threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gt_lanes) if len(gt_lanes) > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Calculate AP (area under PR curve)
        # Use 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        aps.append(ap)
        results[f'ap_{threshold:.1f}'] = float(ap)
    
    # Mean AP
    results['map'] = float(np.mean(aps))
    
    return results
