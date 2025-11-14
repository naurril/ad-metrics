"""
Multi-Object Tracking (MOT) metrics for 3D object tracking evaluation.

Implements CLEAR MOT metrics (MOTA, MOTP) and other tracking-specific metrics
including ID switches, fragmentations, and HOTA.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


def calculate_mota(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Multiple Object Tracking Accuracy (MOTA).
    
    MOTA = 1 - (FN + FP + ID_SW) / GT
    
    Note: For pure detection (single frame), ID switches are 0.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with MOTA and components
    """
    from admetrics.detection.confusion import calculate_tp_fp_fn
    
    counts = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold)
    
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    num_gt = len(ground_truth)
    
    if num_gt == 0:
        return {
            'mota': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'num_gt': 0
        }
    
    # For detection only (no tracking), ID switches = 0
    id_switches = 0
    
    mota = 1 - (fn + fp + id_switches) / num_gt
    
    return {
        'mota': float(mota),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'id_switches': id_switches,
        'num_gt': num_gt
    }


def calculate_motp(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    distance_type: str = "euclidean"
) -> Dict[str, float]:
    """
    Calculate Multiple Object Tracking Precision (MOTP).
    
    MOTP = sum(distance_i) / num_TP
    
    Average distance error for all true positive detections.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold for matching
        distance_type: Type of distance metric to use
    
    Returns:
        Dictionary with MOTP and related metrics
    """
    from admetrics.detection.iou import calculate_iou_3d
    from admetrics.detection.distance import calculate_center_distance
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
    
    # Match predictions to ground truth
    gt_matched = [False] * len(ground_truth)
    distances = []
    
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
            
            # Calculate distance
            dist = calculate_center_distance(
                pred['box'],
                ground_truth[max_gt_idx]['box'],
                distance_type=distance_type
            )
            distances.append(dist)
    
    if len(distances) == 0:
        return {
            'motp': 0.0,
            'mean_distance': 0.0,
            'num_tp': 0
        }
    
    motp = np.mean(distances)
    
    return {
        'motp': float(motp),
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'num_tp': len(distances)
    }


def calculate_clearmot_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate CLEAR MOT metrics (combines MOTA and MOTP).
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with both MOTA and MOTP metrics
    """
    mota_result = calculate_mota(predictions, ground_truth, iou_threshold)
    motp_result = calculate_motp(predictions, ground_truth, iou_threshold)
    
    return {
        **mota_result,
        **motp_result
    }


def calculate_multi_frame_mota(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate MOTA across multiple frames with ID switch tracking.
    
    MOTA = 1 - (FN + FP + IDSW) / total_GT
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> list of predictions
                          Each prediction should have 'box', 'track_id', 'class'
        frame_ground_truth: Dictionary mapping frame_id -> list of ground truth
                           Each GT should have 'box', 'track_id', 'class'
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with tracking metrics including:
            - mota: Multiple Object Tracking Accuracy
            - motp: Multiple Object Tracking Precision
            - num_matches: Total true positives
            - num_false_positives: Total false positives
            - num_misses: Total false negatives (missed detections)
            - num_switches: Total ID switches
            - num_fragmentations: Total track fragmentations
            - mostly_tracked: Number of mostly tracked GT trajectories
            - mostly_lost: Number of mostly lost GT trajectories
            - partially_tracked: Number of partially tracked GT trajectories
    
    Example:
        >>> frame_preds = {
        ...     0: [{'box': [0,0,0,4,2,1.5,0], 'track_id': 1, 'class': 'car'}],
        ...     1: [{'box': [1,0,0,4,2,1.5,0], 'track_id': 1, 'class': 'car'}]
        ... }
        >>> frame_gt = {
        ...     0: [{'box': [0,0,0,4,2,1.5,0], 'track_id': 100, 'class': 'car'}],
        ...     1: [{'box': [1,0,0,4,2,1.5,0], 'track_id': 100, 'class': 'car'}]
        ... }
        >>> metrics = calculate_multi_frame_mota(frame_preds, frame_gt)
    """
    from admetrics.detection.iou import calculate_iou_3d
    from admetrics.detection.distance import calculate_center_distance
    
    # Tracking state
    num_matches = 0
    num_false_positives = 0
    num_misses = 0
    num_switches = 0
    total_gt = 0
    total_distance = 0.0
    
    # Track GT -> Pred mapping for ID switch detection
    gt_to_pred_mapping = {}  # {gt_track_id: pred_track_id}
    
    # Track coverage for each GT trajectory
    gt_track_frames = defaultdict(int)  # {gt_track_id: num_frames_detected}
    gt_track_total_frames = defaultdict(int)  # {gt_track_id: total_frames}
    
    # Process frames in order
    for frame_id in sorted(frame_predictions.keys()):
        if frame_id not in frame_ground_truth:
            # All predictions are FP if no GT in this frame
            num_false_positives += len(frame_predictions[frame_id])
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        total_gt += len(gts)
        
        # Count total frames for each GT track
        for gt in gts:
            gt_id = gt.get('track_id')
            if gt_id is not None:
                gt_track_total_frames[gt_id] += 1
        
        # Match predictions to ground truth using Hungarian assignment
        matches, unmatched_preds, unmatched_gts = _match_frame(
            preds, gts, iou_threshold
        )
        
        # Process matches
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            num_matches += 1
            
            # Calculate distance for MOTP
            dist = calculate_center_distance(pred['box'], gt['box'])
            total_distance += dist
            
            # Check for ID switch
            gt_id = gt.get('track_id')
            pred_id = pred.get('track_id')
            
            if gt_id is not None and pred_id is not None:
                gt_track_frames[gt_id] += 1
                
                if gt_id in gt_to_pred_mapping:
                    # Check if predicted ID matches previous assignment
                    if gt_to_pred_mapping[gt_id] != pred_id:
                        num_switches += 1
                        gt_to_pred_mapping[gt_id] = pred_id
                else:
                    # First time seeing this GT track
                    gt_to_pred_mapping[gt_id] = pred_id
        
        num_false_positives += len(unmatched_preds)
        num_misses += len(unmatched_gts)
    
    # Calculate MOTA
    if total_gt == 0:
        mota = 0.0
        motp = 0.0
    else:
        mota = 1.0 - (num_misses + num_false_positives + num_switches) / total_gt
        motp = total_distance / num_matches if num_matches > 0 else 0.0
    
    # Calculate trajectory-level metrics
    mostly_tracked = 0
    partially_tracked = 0
    mostly_lost = 0
    
    for gt_id, frames_detected in gt_track_frames.items():
        total_frames = gt_track_total_frames[gt_id]
        ratio = frames_detected / total_frames if total_frames > 0 else 0
        
        if ratio >= 0.8:
            mostly_tracked += 1
        elif ratio >= 0.2:
            partially_tracked += 1
        else:
            mostly_lost += 1
    
    # Count fragmentations (GT tracks that were lost and then re-found)
    num_fragmentations = _count_fragmentations(frame_predictions, frame_ground_truth, iou_threshold)
    
    return {
        'mota': float(mota),
        'motp': float(motp),
        'num_matches': num_matches,
        'num_false_positives': num_false_positives,
        'num_misses': num_misses,
        'num_switches': num_switches,
        'num_fragmentations': num_fragmentations,
        'total_gt': total_gt,
        'precision': num_matches / (num_matches + num_false_positives) if (num_matches + num_false_positives) > 0 else 0.0,
        'recall': num_matches / (num_matches + num_misses) if (num_matches + num_misses) > 0 else 0.0,
        'mostly_tracked': mostly_tracked,
        'partially_tracked': partially_tracked,
        'mostly_lost': mostly_lost,
        'num_gt_trajectories': len(gt_track_total_frames)
    }


def _match_frame(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predictions to ground truth in a single frame using Hungarian algorithm.
    
    Args:
        predictions: List of predictions with 'box' and 'class'
        ground_truth: List of ground truth with 'box' and 'class'
        iou_threshold: Minimum IoU for valid match
    
    Returns:
        matches: List of (pred_idx, gt_idx) tuples
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    from admetrics.detection.iou import calculate_iou_3d
    from scipy.optimize import linear_sum_assignment
    
    if len(predictions) == 0:
        return [], [], list(range(len(ground_truth)))
    
    if len(ground_truth) == 0:
        return [], list(range(len(predictions))), []
    
    # Compute IoU cost matrix
    cost_matrix = np.zeros((len(predictions), len(ground_truth)))
    
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            # Only match same class
            if pred.get('class') != gt.get('class'):
                cost_matrix[i, j] = 0
            else:
                iou = calculate_iou_3d(pred['box'], gt['box'])
                cost_matrix[i, j] = iou
    
    # Hungarian algorithm maximizes, so use negative cost
    cost_matrix_neg = -cost_matrix
    
    # Solve assignment problem
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix_neg)
    
    # Filter by IoU threshold
    matches = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] >= iou_threshold:
            matches.append((pred_idx, gt_idx))
    
    # Find unmatched
    matched_pred_indices = {m[0] for m in matches}
    matched_gt_indices = {m[1] for m in matches}
    
    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_pred_indices]
    unmatched_gts = [i for i in range(len(ground_truth)) if i not in matched_gt_indices]
    
    return matches, unmatched_preds, unmatched_gts


def _count_fragmentations(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float
) -> int:
    """
    Count the number of times a ground truth track is fragmented.
    
    A fragmentation occurs when a GT track is matched, then unmatched,
    then matched again.
    
    Args:
        frame_predictions: Frame-indexed predictions
        frame_ground_truth: Frame-indexed ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Number of fragmentations
    """
    # Track the state of each GT trajectory
    gt_track_states = defaultdict(list)  # {gt_track_id: [matched, matched, unmatched, matched, ...]}
    
    for frame_id in sorted(frame_ground_truth.keys()):
        if frame_id not in frame_predictions:
            # All GTs are unmatched
            for gt in frame_ground_truth[frame_id]:
                gt_id = gt.get('track_id')
                if gt_id is not None:
                    gt_track_states[gt_id].append(False)
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, _, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        # Mark matched GTs
        matched_gt_indices = {m[1] for m in matches}
        
        for gt_idx, gt in enumerate(gts):
            gt_id = gt.get('track_id')
            if gt_id is not None:
                is_matched = gt_idx in matched_gt_indices
                gt_track_states[gt_id].append(is_matched)
    
    # Count fragmentations
    num_fragmentations = 0
    
    for gt_id, states in gt_track_states.items():
        # Look for pattern: matched -> unmatched -> matched
        was_matched = False
        was_broken = False
        
        for is_matched in states:
            if was_matched and not is_matched:
                was_broken = True
            elif was_broken and is_matched:
                num_fragmentations += 1
                was_broken = False
            
            if is_matched:
                was_matched = True
    
    return num_fragmentations


def calculate_hota(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Higher Order Tracking Accuracy (HOTA).
    
    HOTA = sqrt(DetA * AssA)
    
    Where:
    - DetA: Detection Accuracy
    - AssA: Association Accuracy
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for detection matching
    
    Returns:
        Dictionary with HOTA and sub-metrics
    
    Reference:
        "HOTA: A Higher Order Metric for Evaluating Multi-object Tracking"
        Luiten et al., IJCV 2021
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    # Calculate detection accuracy
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Calculate association accuracy
    # Track correspondences across frames
    pred_track_to_gt = defaultdict(set)  # {pred_track_id: set of gt_track_ids}
    gt_track_to_pred = defaultdict(set)  # {gt_track_id: set of pred_track_ids}
    
    for frame_id in sorted(frame_ground_truth.keys()):
        if frame_id not in frame_predictions:
            gts = frame_ground_truth[frame_id]
            total_fn += len(gts)
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        total_tp += len(matches)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)
        
        # Track associations
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            pred_id = pred.get('track_id')
            gt_id = gt.get('track_id')
            
            if pred_id is not None and gt_id is not None:
                pred_track_to_gt[pred_id].add(gt_id)
                gt_track_to_pred[gt_id].add(pred_id)
    
    # Calculate DetA (Detection Accuracy)
    if total_tp + total_fp + total_fn == 0:
        det_a = 0.0
    else:
        det_a = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn)
    
    # Calculate AssA (Association Accuracy)
    # For each GT trajectory, find the best matching predicted trajectory
    total_tpa = 0  # True Positive Associations
    total_gt_trajectories = len(gt_track_to_pred)
    
    for gt_id, pred_ids in gt_track_to_pred.items():
        if len(pred_ids) == 0:
            continue
        
        # Find pred_id with maximum overlap
        max_overlap = 0
        for pred_id in pred_ids:
            # Count frames where both tracks are matched to each other
            # This is simplified; full HOTA uses Jaccard similarity
            overlap = 1 / (len(pred_ids) + len(pred_track_to_gt.get(pred_id, set())) - 1)
            max_overlap = max(max_overlap, overlap)
        
        total_tpa += max_overlap
    
    if total_gt_trajectories == 0:
        ass_a = 0.0
    else:
        ass_a = total_tpa / total_gt_trajectories
    
    # HOTA is geometric mean of DetA and AssA
    hota = np.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0.0
    
    return {
        'hota': float(hota),
        'det_a': float(det_a),
        'ass_a': float(ass_a),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def calculate_id_f1(
    frame_predictions: Dict[int, List[Dict]],
    frame_ground_truth: Dict[int, List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate ID-based precision, recall, and F1 score.
    
    Evaluates how well track identities are preserved across frames.
    
    Args:
        frame_predictions: Dictionary mapping frame_id -> predictions
        frame_ground_truth: Dictionary mapping frame_id -> ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with IDP, IDR, IDF1 metrics
    """
    # Count correct ID assignments
    idtp = 0  # ID true positives
    idfp = 0  # ID false positives  
    idfn = 0  # ID false negatives
    
    # Track GT -> Pred mapping
    gt_to_pred_mapping = {}
    
    for frame_id in sorted(frame_ground_truth.keys()):
        if frame_id not in frame_predictions:
            continue
        
        preds = frame_predictions[frame_id]
        gts = frame_ground_truth[frame_id]
        
        matches, unmatched_preds, unmatched_gts = _match_frame(preds, gts, iou_threshold)
        
        for pred_idx, gt_idx in matches:
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            
            pred_id = pred.get('track_id')
            gt_id = gt.get('track_id')
            
            if pred_id is None or gt_id is None:
                continue
            
            # Check if this is the correct ID assignment
            if gt_id in gt_to_pred_mapping:
                if gt_to_pred_mapping[gt_id] == pred_id:
                    idtp += 1
                else:
                    idfp += 1
                    idfn += 1
            else:
                gt_to_pred_mapping[gt_id] = pred_id
                idtp += 1
        
        # Unmatched predictions with IDs are false positives
        for pred_idx in unmatched_preds:
            if preds[pred_idx].get('track_id') is not None:
                idfp += 1
        
        # Unmatched GTs with IDs are false negatives
        for gt_idx in unmatched_gts:
            if gts[gt_idx].get('track_id') is not None:
                idfn += 1
    
    # Calculate metrics
    idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0.0  # ID Precision
    idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0.0  # ID Recall
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0  # ID F1
    
    return {
        'idp': float(idp),
        'idr': float(idr),
        'idf1': float(idf1),
        'idtp': idtp,
        'idfp': idfp,
        'idfn': idfn
    }
