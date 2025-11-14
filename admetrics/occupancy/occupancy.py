"""
3D Occupancy Prediction Metrics.

This module provides metrics for evaluating 3D occupancy prediction models
used in autonomous driving. Occupancy prediction predicts which voxels in
a 3D grid are occupied by different semantic classes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist


def occupancy_iou(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    class_id: Optional[int] = None,
    ignore_index: int = 255
) -> float:
    """
    Calculate Intersection over Union (IoU) for 3D occupancy predictions.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        class_id: Specific class ID to compute IoU for. If None, treats as binary occupancy
        ignore_index: Label value to ignore in calculation (e.g., unknown/invalid voxels)
    
    Returns:
        IoU score between 0 and 1
    
    Example:
        >>> pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        >>> gt = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        >>> iou = occupancy_iou(pred, gt, class_id=1)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    # Create mask for valid voxels (not ignored)
    valid_mask = gt_occupancy != ignore_index
    
    if class_id is not None:
        # Calculate IoU for specific class
        pred_mask = (pred_occupancy == class_id) & valid_mask
        gt_mask = (gt_occupancy == class_id) & valid_mask
    else:
        # Binary occupancy (occupied vs free)
        pred_mask = (pred_occupancy > 0) & valid_mask
        gt_mask = (gt_occupancy > 0) & valid_mask
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def mean_iou(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    ignore_classes: Optional[List[int]] = None
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Calculate mean Intersection over Union (mIoU) across all classes.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        num_classes: Total number of semantic classes (including background/free space)
        ignore_index: Label value to ignore in calculation
        ignore_classes: List of class IDs to exclude from mIoU calculation
    
    Returns:
        Dictionary containing:
            - 'mIoU': Mean IoU across all valid classes
            - 'class_iou': Dictionary mapping class_id to IoU score
            - 'valid_classes': Number of classes with non-zero union
    
    Example:
        >>> pred = np.random.randint(0, 3, size=(10, 10, 10))
        >>> gt = np.random.randint(0, 3, size=(10, 10, 10))
        >>> result = mean_iou(pred, gt, num_classes=3)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    ignore_classes = ignore_classes or []
    valid_mask = gt_occupancy != ignore_index
    
    class_ious = {}
    valid_class_count = 0
    
    for class_id in range(num_classes):
        if class_id in ignore_classes:
            continue
        
        pred_mask = (pred_occupancy == class_id) & valid_mask
        gt_mask = (gt_occupancy == class_id) & valid_mask
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union > 0:
            class_ious[class_id] = float(intersection / union)
            valid_class_count += 1
        else:
            # Class not present in ground truth or prediction
            class_ious[class_id] = np.nan
    
    # Calculate mIoU only over classes that appear in the data
    valid_ious = [iou for iou in class_ious.values() if not np.isnan(iou)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    
    return {
        'mIoU': miou,
        'class_iou': class_ious,
        'valid_classes': valid_class_count
    }


def occupancy_precision_recall(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    class_id: Optional[int] = None,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1-score for occupancy predictions.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        class_id: Specific class ID to evaluate. If None, treats as binary occupancy
        ignore_index: Label value to ignore in calculation
    
    Returns:
        Dictionary containing:
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1-score
            - 'true_positives': Number of true positives
            - 'false_positives': Number of false positives
            - 'false_negatives': Number of false negatives
    
    Example:
        >>> pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        >>> gt = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        >>> metrics = occupancy_precision_recall(pred, gt, class_id=1)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    valid_mask = gt_occupancy != ignore_index
    
    if class_id is not None:
        pred_mask = (pred_occupancy == class_id) & valid_mask
        gt_mask = (gt_occupancy == class_id) & valid_mask
    else:
        pred_mask = (pred_occupancy > 0) & valid_mask
        gt_mask = (gt_occupancy > 0) & valid_mask
    
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def scene_completion(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    free_class: int = 0,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Calculate Scene Completion (SC) metrics for 3D occupancy.
    
    Scene Completion evaluates how well the model completes the scene by
    predicting both occupied and free space correctly.
    
    Args:
        pred_occupancy: Predicted occupancy grid of shape (X, Y, Z) with class labels
        gt_occupancy: Ground truth occupancy grid of shape (X, Y, Z) with class labels
        free_class: Class ID representing free/empty space (default: 0)
        ignore_index: Label value to ignore in calculation
    
    Returns:
        Dictionary containing:
            - 'SC_IoU': IoU for scene completion (occupied space)
            - 'SC_Precision': Precision for occupied voxels
            - 'SC_Recall': Recall for occupied voxels
            - 'SSC_mIoU': Semantic Scene Completion mIoU (excludes free space)
            - 'completion_ratio': Ratio of predicted occupied to GT occupied voxels
    
    Example:
        >>> pred = np.random.randint(0, 3, size=(10, 10, 10))
        >>> gt = np.random.randint(0, 3, size=(10, 10, 10))
        >>> sc_metrics = scene_completion(pred, gt, free_class=0)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    valid_mask = gt_occupancy != ignore_index
    
    # Binary occupancy: occupied (any non-free class) vs free
    pred_occupied = (pred_occupancy != free_class) & valid_mask
    gt_occupied = (gt_occupancy != free_class) & valid_mask
    
    # Scene Completion IoU (binary: occupied vs free)
    intersection = np.logical_and(pred_occupied, gt_occupied).sum()
    union = np.logical_or(pred_occupied, gt_occupied).sum()
    sc_iou = float(intersection / union) if union > 0 else 0.0
    
    # Precision and Recall for occupied voxels
    tp = intersection
    fp = np.logical_and(pred_occupied, ~gt_occupied).sum()
    fn = np.logical_and(~pred_occupied, gt_occupied).sum()
    
    sc_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    sc_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Completion ratio
    pred_occupied_count = pred_occupied.sum()
    gt_occupied_count = gt_occupied.sum()
    completion_ratio = float(pred_occupied_count / gt_occupied_count) if gt_occupied_count > 0 else 0.0
    
    # Semantic Scene Completion mIoU (for occupied voxels only)
    # Get unique classes excluding free space
    occupied_classes = np.unique(gt_occupancy[gt_occupied])
    occupied_classes = occupied_classes[occupied_classes != free_class]
    
    if len(occupied_classes) > 0:
        class_ious = []
        for class_id in occupied_classes:
            pred_class = (pred_occupancy == class_id) & valid_mask
            gt_class = (gt_occupancy == class_id) & valid_mask
            
            class_intersection = np.logical_and(pred_class, gt_class).sum()
            class_union = np.logical_or(pred_class, gt_class).sum()
            
            if class_union > 0:
                class_ious.append(class_intersection / class_union)
        
        ssc_miou = float(np.mean(class_ious)) if class_ious else 0.0
    else:
        ssc_miou = 0.0
    
    return {
        'SC_IoU': sc_iou,
        'SC_Precision': sc_precision,
        'SC_Recall': sc_recall,
        'SSC_mIoU': ssc_miou,
        'completion_ratio': completion_ratio
    }


def chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    bidirectional: bool = True
) -> Dict[str, float]:
    """
    Calculate Chamfer Distance between predicted and ground truth point clouds.
    
    Chamfer Distance measures the average distance from each point in one set
    to its nearest neighbor in the other set. Useful for evaluating occupancy
    prediction quality at surface boundaries.
    
    Args:
        pred_points: Predicted occupied voxel centers, shape (N, 3)
        gt_points: Ground truth occupied voxel centers, shape (M, 3)
        bidirectional: If True, compute symmetric Chamfer Distance (both directions)
    
    Returns:
        Dictionary containing:
            - 'chamfer_distance': Average Chamfer Distance
            - 'pred_to_gt': Average distance from predicted to GT points
            - 'gt_to_pred': Average distance from GT to predicted points (if bidirectional)
    
    Example:
        >>> pred_pts = np.random.rand(100, 3)
        >>> gt_pts = np.random.rand(120, 3)
        >>> cd = chamfer_distance(pred_pts, gt_pts)
    """
    if pred_points.shape[1] != 3 or gt_points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return {
            'chamfer_distance': np.inf,
            'pred_to_gt': np.inf,
            'gt_to_pred': np.inf if bidirectional else None
        }
    
    # Compute pairwise distances
    distances = cdist(pred_points, gt_points, metric='euclidean')
    
    # Pred to GT: for each predicted point, find nearest GT point
    pred_to_gt = np.mean(np.min(distances, axis=1))
    
    result = {
        'pred_to_gt': float(pred_to_gt)
    }
    
    if bidirectional:
        # GT to Pred: for each GT point, find nearest predicted point
        gt_to_pred = np.mean(np.min(distances, axis=0))
        result['gt_to_pred'] = float(gt_to_pred)
        result['chamfer_distance'] = float((pred_to_gt + gt_to_pred) / 2)
    else:
        result['chamfer_distance'] = float(pred_to_gt)
        result['gt_to_pred'] = None
    
    return result


def surface_distance(
    pred_occupancy: np.ndarray,
    gt_occupancy: np.ndarray,
    voxel_size: float = 1.0,
    percentile: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate surface distance metrics between predicted and GT occupancy.
    
    Measures how well the predicted occupancy boundaries match the ground truth
    boundaries by computing distances between surface voxels.
    
    Args:
        pred_occupancy: Predicted binary occupancy grid, shape (X, Y, Z)
        gt_occupancy: Ground truth binary occupancy grid, shape (X, Y, Z)
        voxel_size: Physical size of each voxel in meters
        percentile: If specified, compute percentile distance (e.g., 95 for 95th percentile)
    
    Returns:
        Dictionary containing:
            - 'mean_surface_distance': Average distance between surfaces
            - 'median_surface_distance': Median distance
            - 'std_surface_distance': Standard deviation of distances
            - 'percentile_distance': Percentile distance if percentile is specified
            - 'max_surface_distance': Maximum distance
    
    Example:
        >>> pred = np.random.randint(0, 2, size=(20, 20, 20))
        >>> gt = np.random.randint(0, 2, size=(20, 20, 20))
        >>> sd = surface_distance(pred, gt, voxel_size=0.2)
    """
    if pred_occupancy.shape != gt_occupancy.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_occupancy.shape} vs gt {gt_occupancy.shape}"
        )
    
    # Convert to binary if not already
    pred_binary = (pred_occupancy > 0).astype(bool)
    gt_binary = (gt_occupancy > 0).astype(bool)
    
    # Find surface voxels (voxels with at least one free neighbor)
    def get_surface_voxels(occupancy: np.ndarray) -> np.ndarray:
        """Extract coordinates of surface voxels."""
        from scipy.ndimage import binary_erosion
        
        # Surface = occupied voxels that are not completely surrounded
        eroded = binary_erosion(occupancy)
        surface = occupancy & ~eroded
        
        return np.argwhere(surface)
    
    pred_surface = get_surface_voxels(pred_binary)
    gt_surface = get_surface_voxels(gt_binary)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return {
            'mean_surface_distance': np.inf,
            'median_surface_distance': np.inf,
            'std_surface_distance': 0.0,
            'max_surface_distance': np.inf,
            'percentile_distance': np.inf if percentile else None
        }
    
    # Convert voxel coordinates to physical coordinates
    pred_surface_coords = pred_surface * voxel_size
    gt_surface_coords = gt_surface * voxel_size
    
    # Compute distances
    distances = cdist(pred_surface_coords, gt_surface_coords, metric='euclidean')
    
    # For each predicted surface voxel, find distance to nearest GT surface voxel
    min_distances = np.min(distances, axis=1)
    
    result = {
        'mean_surface_distance': float(np.mean(min_distances)),
        'median_surface_distance': float(np.median(min_distances)),
        'std_surface_distance': float(np.std(min_distances)),
        'max_surface_distance': float(np.max(min_distances))
    }
    
    if percentile is not None:
        result['percentile_distance'] = float(np.percentile(min_distances, percentile))
    else:
        result['percentile_distance'] = None
    
    return result
