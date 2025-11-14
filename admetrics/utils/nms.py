"""
Non-Maximum Suppression (NMS) for 3D object detection.
"""

import numpy as np
from typing import List, Dict, Union, Tuple


def nms_3d(
    boxes: Union[List[Dict], np.ndarray],
    scores: np.ndarray = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0
) -> Union[List[int], np.ndarray]:
    """
    3D Non-Maximum Suppression.
    
    Args:
        boxes: Either list of dicts with 'box' and 'score', or (N, 7) array
        scores: Score array (required if boxes is ndarray)
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score threshold
    
    Returns:
        Indices of boxes to keep
        
    Example:
        >>> boxes = [
        ...     {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
        ...     {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8}
        ... ]
        >>> keep_indices = nms_3d(boxes, iou_threshold=0.5)
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    # Parse input
    if isinstance(boxes, list):
        box_array = np.array([b['box'] for b in boxes])
        score_array = np.array([b.get('score', 1.0) for b in boxes])
    else:
        box_array = boxes
        score_array = scores if scores is not None else np.ones(len(boxes))
    
    # Filter by score threshold
    score_mask = score_array >= score_threshold
    valid_indices = np.where(score_mask)[0]
    
    if len(valid_indices) == 0:
        return []
    
    box_array = box_array[valid_indices]
    score_array = score_array[valid_indices]
    
    # Sort by score (descending)
    sorted_indices = np.argsort(-score_array)
    
    keep = []
    
    while len(sorted_indices) > 0:
        # Keep the highest score box
        current_idx = sorted_indices[0]
        keep.append(valid_indices[current_idx])
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = box_array[current_idx]
        remaining_indices = sorted_indices[1:]
        
        ious = np.array([
            calculate_iou_3d(current_box, box_array[idx])
            for idx in remaining_indices
        ])
        
        # Keep boxes with IoU below threshold
        keep_mask = ious < iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    return keep


def nms_bev(
    boxes: Union[List[Dict], np.ndarray],
    scores: np.ndarray = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0
) -> Union[List[int], np.ndarray]:
    """
    Bird's Eye View NMS (2D).
    
    Args:
        boxes: Boxes with 'box' or (N, 7) array
        scores: Scores (if boxes is array)
        iou_threshold: IoU threshold
        score_threshold: Score threshold
    
    Returns:
        Indices of boxes to keep
    """
    from admetrics.detection.iou import calculate_iou_bev
    
    # Parse input
    if isinstance(boxes, list):
        box_array = np.array([b['box'] for b in boxes])
        score_array = np.array([b.get('score', 1.0) for b in boxes])
    else:
        box_array = boxes
        score_array = scores if scores is not None else np.ones(len(boxes))
    
    # Filter by score
    score_mask = score_array >= score_threshold
    valid_indices = np.where(score_mask)[0]
    
    if len(valid_indices) == 0:
        return []
    
    box_array = box_array[valid_indices]
    score_array = score_array[valid_indices]
    
    # Sort by score
    sorted_indices = np.argsort(-score_array)
    
    keep = []
    
    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        keep.append(valid_indices[current_idx])
        
        if len(sorted_indices) == 1:
            break
        
        current_box = box_array[current_idx]
        remaining_indices = sorted_indices[1:]
        
        ious = np.array([
            calculate_iou_bev(current_box, box_array[idx])
            for idx in remaining_indices
        ])
        
        keep_mask = ious < iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    return keep


def nms_per_class(
    boxes: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    mode: str = "3d"
) -> List[int]:
    """
    Apply NMS separately for each class.
    
    Args:
        boxes: List of box dicts with 'box', 'score', 'class'
        iou_threshold: IoU threshold
        score_threshold: Score threshold
        mode: '3d' or 'bev'
    
    Returns:
        Indices of boxes to keep
    """
    # Group boxes by class
    class_boxes = {}
    for idx, box in enumerate(boxes):
        cls = box.get('class', 'default')
        if cls not in class_boxes:
            class_boxes[cls] = []
        class_boxes[cls].append((idx, box))
    
    # Apply NMS per class
    nms_func = nms_3d if mode == "3d" else nms_bev
    keep_indices = []
    
    for cls, cls_boxes in class_boxes.items():
        indices, boxes_only = zip(*cls_boxes)
        
        # Apply NMS
        keep_local = nms_func(
            list(boxes_only),
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )
        
        # Map back to global indices
        keep_global = [indices[i] for i in keep_local]
        keep_indices.extend(keep_global)
    
    return sorted(keep_indices)


def soft_nms_3d(
    boxes: Union[List[Dict], np.ndarray],
    scores: np.ndarray = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    sigma: float = 0.5,
    method: str = "gaussian"
) -> Tuple[List[int], np.ndarray]:
    """
    Soft NMS for 3D boxes.
    
    Instead of removing overlapping boxes, reduces their scores.
    
    Args:
        boxes: Boxes
        scores: Scores
        iou_threshold: IoU threshold
        score_threshold: Score threshold
        sigma: Sigma parameter for Gaussian decay
        method: 'gaussian' or 'linear'
    
    Returns:
        Tuple of (keep_indices, updated_scores)
    """
    from admetrics.detection.iou import calculate_iou_3d
    
    # Parse input
    if isinstance(boxes, list):
        box_array = np.array([b['box'] for b in boxes])
        score_array = np.array([b.get('score', 1.0) for b in boxes])
    else:
        box_array = boxes
        score_array = scores if scores is not None else np.ones(len(boxes))
    
    score_array = score_array.copy()
    
    N = len(box_array)
    keep = []
    
    for i in range(N):
        # Find box with highest score
        max_idx = np.argmax(score_array)
        max_score = score_array[max_idx]
        
        if max_score < score_threshold:
            break
        
        keep.append(max_idx)
        
        # Update scores of remaining boxes based on IoU
        current_box = box_array[max_idx]
        
        for j in range(N):
            if j == max_idx or score_array[j] < score_threshold:
                continue
            
            iou = calculate_iou_3d(current_box, box_array[j])
            
            if method == "gaussian":
                # Gaussian decay
                weight = np.exp(-(iou ** 2) / sigma)
            elif method == "linear":
                # Linear decay
                if iou > iou_threshold:
                    weight = 1 - iou
                else:
                    weight = 1.0
            else:
                raise ValueError(f"Unknown method: {method}")
            
            score_array[j] *= weight
        
        # Mark current box as processed
        score_array[max_idx] = -1
    
    return keep, score_array


def distance_based_nms(
    boxes: List[Dict],
    distance_threshold: float = 2.0,
    score_threshold: float = 0.0
) -> List[int]:
    """
    NMS based on center distance instead of IoU.
    
    Useful for clustering nearby detections.
    
    Args:
        boxes: List of box dicts
        distance_threshold: Distance threshold in meters
        score_threshold: Score threshold
    
    Returns:
        Indices of boxes to keep
    """
    from admetrics.detection.distance import calculate_center_distance
    
    # Filter by score
    valid_boxes = [(i, b) for i, b in enumerate(boxes) if b.get('score', 1.0) >= score_threshold]
    
    if len(valid_boxes) == 0:
        return []
    
    indices, boxes_only = zip(*valid_boxes)
    scores = np.array([b.get('score', 1.0) for b in boxes_only])
    
    # Sort by score
    sorted_idx = np.argsort(-scores)
    
    keep = []
    
    while len(sorted_idx) > 0:
        current_local_idx = sorted_idx[0]
        current_global_idx = indices[current_local_idx]
        keep.append(current_global_idx)
        
        if len(sorted_idx) == 1:
            break
        
        current_box = boxes_only[current_local_idx]
        remaining_idx = sorted_idx[1:]
        
        # Calculate distances
        distances = np.array([
            calculate_center_distance(current_box['box'], boxes_only[idx]['box'], 'euclidean')
            for idx in remaining_idx
        ])
        
        # Keep boxes beyond distance threshold
        keep_mask = distances >= distance_threshold
        sorted_idx = remaining_idx[keep_mask]
    
    return sorted(keep)
