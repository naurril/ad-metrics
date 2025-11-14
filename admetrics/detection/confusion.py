"""
Confusion matrix metrics: TP, FP, FN, Precision, Recall, F1-Score.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def calculate_tp_fp_fn(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    metric_type: str = "3d"
) -> Dict[str, int]:
    """
    Calculate True Positives, False Positives, and False Negatives.
    
    Args:
        predictions: List of prediction dicts
        ground_truth: List of ground truth dicts
        iou_threshold: IoU threshold for matching
        metric_type: '3d' or 'bev'
    
    Returns:
        Dictionary with:
            - 'tp': Number of true positives
            - 'fp': Number of false positives
            - 'fn': Number of false negatives
            
    Example:
        >>> result = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold=0.5)
        >>> print(f"TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
    """
    from admetrics.detection.iou import calculate_iou_3d, calculate_iou_bev
    
    iou_func = calculate_iou_3d if metric_type == "3d" else calculate_iou_bev
    
    # Track matched ground truth boxes
    gt_matched = [False] * len(ground_truth)
    
    tp = 0
    fp = 0
    
    # Match predictions to ground truth
    for pred in predictions:
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            # Check class match
            if pred.get('class') != gt.get('class'):
                continue
            
            # Skip if already matched
            if gt_matched[gt_idx]:
                continue
            
            iou = iou_func(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Assign TP or FP
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            tp += 1
            gt_matched[max_gt_idx] = True
        else:
            fp += 1
    
    # False negatives are unmatched ground truth boxes
    fn = len(ground_truth) - sum(gt_matched)
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_confusion_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    metric_type: str = "3d"
) -> Dict[str, float]:
    """
    Calculate comprehensive confusion matrix metrics.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
        metric_type: '3d' or 'bev'
    
    Returns:
        Dictionary with:
            - 'precision': TP / (TP + FP)
            - 'recall': TP / (TP + FN)
            - 'f1_score': 2 * (precision * recall) / (precision + recall)
            - 'tp', 'fp', 'fn': Raw counts
            
    Example:
        >>> metrics = calculate_confusion_metrics(predictions, ground_truth)
        >>> print(f"Precision: {metrics['precision']:.4f}")
        >>> print(f"Recall: {metrics['recall']:.4f}")
        >>> print(f"F1: {metrics['f1_score']:.4f}")
    """
    counts = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold, metric_type)
    
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    
    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_confusion_matrix_multiclass(
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5,
    metric_type: str = "3d"
) -> Dict[str, np.ndarray]:
    """
    Calculate confusion matrix for multi-class detection.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        class_names: List of all class names
        iou_threshold: IoU threshold
        metric_type: '3d' or 'bev'
    
    Returns:
        Dictionary with:
            - 'confusion_matrix': (N, N) numpy array where N is number of classes
            - 'class_names': List of class names corresponding to matrix indices
            - 'per_class_metrics': Metrics for each class
    """
    from admetrics.detection.iou import calculate_iou_3d, calculate_iou_bev
    
    iou_func = calculate_iou_3d if metric_type == "3d" else calculate_iou_bev
    
    n_classes = len(class_names)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Initialize confusion matrix
    # Rows: ground truth classes, Columns: predicted classes
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    # Track matched ground truth
    gt_matched = [False] * len(ground_truth)
    
    # Match predictions to ground truth
    for pred in predictions:
        pred_class = pred.get('class')
        if pred_class not in class_to_idx:
            continue
        
        max_iou = 0
        max_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            
            iou = iou_func(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Update confusion matrix
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            gt_class = ground_truth[max_gt_idx].get('class')
            if gt_class in class_to_idx:
                gt_idx = class_to_idx[gt_class]
                pred_idx = class_to_idx[pred_class]
                confusion_matrix[gt_idx, pred_idx] += 1
                gt_matched[max_gt_idx] = True
        else:
            # False positive - no matching ground truth
            # Can be represented as background class if desired
            pass
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for cls_name in class_names:
        cls_idx = class_to_idx[cls_name]
        
        # TP: diagonal element
        tp = confusion_matrix[cls_idx, cls_idx]
        
        # FP: sum of column excluding diagonal
        fp = np.sum(confusion_matrix[:, cls_idx]) - tp
        
        # FN: sum of row excluding diagonal
        fn = np.sum(confusion_matrix[cls_idx, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[cls_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    return {
        'confusion_matrix': confusion_matrix,
        'class_names': class_names,
        'per_class_metrics': per_class_metrics
    }


def calculate_specificity(
    predictions: List[Dict],
    ground_truth: List[Dict],
    total_negatives: int,
    iou_threshold: float = 0.5,
    metric_type: str = "3d"
) -> float:
    """
    Calculate specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    
    Note: In object detection, true negatives are typically not well-defined
    since there are infinite possible negative boxes. This requires specifying
    the total number of negative samples explicitly.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        total_negatives: Total number of negative samples
        iou_threshold: IoU threshold
        metric_type: '3d' or 'bev'
    
    Returns:
        Specificity value
    """
    counts = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold, metric_type)
    
    fp = counts['fp']
    tn = total_negatives - fp
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return float(specificity)
