"""
Average Precision (AP) and Mean Average Precision (mAP) calculations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from scipy.interpolate import interp1d


def calculate_ap(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    num_recall_points: int = 40,
    metric_type: str = "3d"
) -> Dict[str, float]:
    """
    Calculate Average Precision (AP) for 3D object detection.
    
    Args:
        predictions: List of prediction dicts with keys:
            - 'box': 3D bounding box [x, y, z, w, h, l, r]
            - 'score': confidence score
            - 'class': class name
        ground_truth: List of ground truth dicts with keys:
            - 'box': 3D bounding box
            - 'class': class name
            - 'difficulty': (optional) difficulty level
        iou_threshold: IoU threshold for considering a match
        num_recall_points: Number of recall points for interpolation
        metric_type: '3d' or 'bev'
    
    Returns:
        Dictionary containing:
            - 'ap': Average Precision value
            - 'precision': Precision values at recall points
            - 'recall': Recall values
            - 'scores': Confidence scores
            
    Example:
        >>> predictions = [
        ...     {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
        ...     {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}
        ... ]
        >>> ground_truth = [
        ...     {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ... ]
        >>> result = calculate_ap(predictions, ground_truth)
        >>> print(f"AP: {result['ap']:.4f}")
    """
    from admetrics.detection.iou import calculate_iou_3d, calculate_iou_bev
    
    iou_func = calculate_iou_3d if metric_type == "3d" else calculate_iou_bev
    
    # Sort predictions by confidence score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Track which ground truth boxes have been matched
    gt_matched = [False] * len(ground_truth)
    
    # Arrays to track TP and FP
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    scores = np.array([p['score'] for p in predictions])
    
    # Match predictions to ground truth
    for pred_idx, pred in enumerate(predictions):
        max_iou = 0
        max_gt_idx = -1
        
        # Find best matching ground truth box
        for gt_idx, gt in enumerate(ground_truth):
            # Check if same class
            if pred.get('class') != gt.get('class'):
                continue
            
            # Skip if already matched
            if gt_matched[gt_idx]:
                continue
            
            # Calculate IoU
            iou = iou_func(pred['box'], gt['box'])
            
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx
        
        # Assign TP or FP
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[max_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute precision and recall
    num_gt = len(ground_truth)
    recalls = tp_cumsum / max(num_gt, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Compute AP using interpolated precision-recall curve
    ap = _compute_ap_interp(recalls, precisions, num_recall_points)
    
    return {
        'ap': float(ap),
        'precision': precisions,
        'recall': recalls,
        'scores': scores,
        'num_tp': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
        'num_fp': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
        'num_gt': num_gt
    }


def _compute_ap_interp(
    recalls: np.ndarray,
    precisions: np.ndarray,
    num_points: int = 40
) -> float:
    """
    Compute AP using N-point interpolation method.
    
    This is the standard PASCAL VOC / COCO style AP calculation.
    """
    if len(recalls) == 0:
        return 0.0
    
    # Append sentinel values at the beginning and end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute the precision envelope (maximum precision at each recall level)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP by numerical integration
    recall_points = np.linspace(0, 1, num_points)
    ap = 0.0
    
    for r in recall_points:
        # Find all precisions at recalls >= r
        idx = np.where(recalls >= r)[0]
        if len(idx) > 0:
            ap += precisions[idx[0]]
    
    ap = ap / num_points
    
    return float(ap)


def calculate_map(
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_names: List[str],
    iou_thresholds: Union[List[float], float] = 0.5,
    num_recall_points: int = 40,
    metric_type: str = "3d"
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate Mean Average Precision (mAP) across multiple classes and IoU thresholds.
    
    Args:
        predictions: List of all predictions
        ground_truth: List of all ground truth annotations
        class_names: List of class names to evaluate
        iou_thresholds: Single threshold or list of thresholds
        num_recall_points: Number of recall points
        metric_type: '3d' or 'bev'
    
    Returns:
        Dictionary containing:
            - 'mAP': Overall mean AP
            - 'AP_per_class': AP for each class
            - 'AP_per_threshold': AP for each IoU threshold
            
    Example:
        >>> results = calculate_map(
        ...     predictions=all_preds,
        ...     ground_truth=all_gt,
        ...     class_names=['car', 'pedestrian', 'cyclist'],
        ...     iou_thresholds=[0.5, 0.7]
        ... )
        >>> print(f"mAP: {results['mAP']:.4f}")
    """
    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = [iou_thresholds]
    
    # Store results
    ap_per_class = {cls: [] for cls in class_names}
    ap_per_threshold = {thr: [] for thr in iou_thresholds}
    all_aps = []
    
    # Calculate AP for each class and threshold combination
    for cls in class_names:
        # Filter predictions and ground truth for this class
        cls_preds = [p for p in predictions if p.get('class') == cls]
        cls_gt = [g for g in ground_truth if g.get('class') == cls]
        
        if len(cls_gt) == 0:
            continue
        
        for iou_thr in iou_thresholds:
            result = calculate_ap(
                predictions=cls_preds,
                ground_truth=cls_gt,
                iou_threshold=iou_thr,
                num_recall_points=num_recall_points,
                metric_type=metric_type
            )
            
            ap = result['ap']
            ap_per_class[cls].append(ap)
            ap_per_threshold[iou_thr].append(ap)
            all_aps.append(ap)
    
    # Compute mean AP
    mAP = np.mean(all_aps) if len(all_aps) > 0 else 0.0
    
    # Average across thresholds for each class
    ap_per_class_avg = {
        cls: np.mean(aps) if len(aps) > 0 else 0.0
        for cls, aps in ap_per_class.items()
    }
    
    # Average across classes for each threshold
    ap_per_threshold_avg = {
        thr: np.mean(aps) if len(aps) > 0 else 0.0
        for thr, aps in ap_per_threshold.items()
    }
    
    return {
        'mAP': float(mAP),
        'AP_per_class': ap_per_class_avg,
        'AP_per_threshold': ap_per_threshold_avg,
        'num_classes': len([c for c in class_names if len([g for g in ground_truth if g.get('class') == c]) > 0])
    }


def calculate_ap_coco_style(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate AP using COCO-style evaluation (average over multiple IoU thresholds).
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_thresholds: List of IoU thresholds (default: [0.5:0.95:0.05])
    
    Returns:
        Dictionary with AP metrics:
            - 'AP': Average over [0.5:0.95]
            - 'AP50': AP at IoU=0.5
            - 'AP75': AP at IoU=0.75
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    
    aps = []
    ap_50 = None
    ap_75 = None
    
    for iou_thr in iou_thresholds:
        result = calculate_ap(
            predictions=predictions,
            ground_truth=ground_truth,
            iou_threshold=iou_thr,
            num_recall_points=101  # COCO uses 101 points
        )
        
        ap = result['ap']
        aps.append(ap)
        
        if abs(iou_thr - 0.5) < 1e-5:
            ap_50 = ap
        if abs(iou_thr - 0.75) < 1e-5:
            ap_75 = ap
    
    return {
        'AP': float(np.mean(aps)),
        'AP50': float(ap_50) if ap_50 is not None else 0.0,
        'AP75': float(ap_75) if ap_75 is not None else 0.0,
        'AP_per_threshold': {f'{thr:.2f}': float(ap) for thr, ap in zip(iou_thresholds, aps)}
    }


def calculate_precision_recall_curve(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    metric_type: str = "3d"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold
        metric_type: '3d' or 'bev'
    
    Returns:
        Tuple of (precision_array, recall_array, score_thresholds)
    """
    result = calculate_ap(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=iou_threshold,
        metric_type=metric_type
    )
    
    return result['precision'], result['recall'], result['scores']
