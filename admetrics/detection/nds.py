"""
NuScenes Detection Score (NDS) calculation.

NDS is a composite metric used in the nuScenes detection benchmark that combines
mAP with several error metrics.
"""

import numpy as np
from typing import List, Dict, Optional, Union


def calculate_nds(
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5,
    distance_thresholds: Optional[List[float]] = None
) -> float:
    """
    Calculate nuScenes Detection Score (NDS).
    
    NDS = 1/10 * [5*mAP + sum(1 - min(1, TP_error_i))]
    
    where TP_error includes:
    - Average Translation Error (ATE)
    - Average Scale Error (ASE)
    - Average Orientation Error (AOE)
    - Average Velocity Error (AVE)
    - Average Attribute Error (AAE)
    
    Args:
        predictions: List of prediction dicts with keys:
            - 'box': [x, y, z, w, h, l, yaw]
            - 'score': confidence score
            - 'class': class name
            - 'velocity': (optional) [vx, vy]
            - 'attributes': (optional) attribute predictions
        ground_truth: List of ground truth dicts
        class_names: List of class names
        iou_threshold: IoU threshold for matching
        distance_thresholds: Distance thresholds for different classes
    
    Returns:
        NDS score (0-1)
        
    Example:
        >>> nds = calculate_nds(predictions, ground_truth, ['car', 'pedestrian'])
        >>> print(f"NDS: {nds:.4f}")
    """
    from admetrics.detection.ap import calculate_map
    
    # Calculate mAP
    map_result = calculate_map(
        predictions=predictions,
        ground_truth=ground_truth,
        class_names=class_names,
        iou_thresholds=iou_threshold,
        metric_type="3d"
    )
    
    mAP = map_result['mAP']
    
    # Calculate True Positive errors
    tp_metrics = calculate_tp_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=iou_threshold
    )
    
    # NDS formula: 1/10 * [5*mAP + sum(1 - min(1, error_i))]
    # TP metrics: ATE, ASE, AOE, AVE, AAE (5 metrics)
    error_sum = 0.0
    for metric_name in ['ate', 'ase', 'aoe', 'ave', 'aae']:
        error = tp_metrics.get(metric_name, 1.0)
        error_sum += (1.0 - min(1.0, error))
    
    nds = (5 * mAP + error_sum) / 10.0
    
    return float(nds)


def calculate_tp_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate True Positive error metrics for NDS.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with error metrics:
            - 'ate': Average Translation Error (meters)
            - 'ase': Average Scale Error (1 - IoU)
            - 'aoe': Average Orientation Error (radians)
            - 'ave': Average Velocity Error (m/s)
            - 'aae': Average Attribute Error
    """
    from admetrics.detection.iou import calculate_iou_3d
    from admetrics.detection.distance import calculate_center_distance, calculate_orientation_error
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Track matched pairs
    matched_pairs = []
    gt_matched = [False] * len(ground_truth)
    
    # Match predictions to ground truth
    for pred in predictions:
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
        
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            matched_pairs.append((pred, ground_truth[max_gt_idx]))
            gt_matched[max_gt_idx] = True
    
    if len(matched_pairs) == 0:
        return {
            'ate': 1.0,
            'ase': 1.0,
            'aoe': 1.0,
            'ave': 1.0,
            'aae': 1.0,
            'num_tp': 0
        }
    
    # Calculate errors for matched pairs
    translation_errors = []
    scale_errors = []
    orientation_errors = []
    velocity_errors = []
    attribute_errors = []
    
    for pred, gt in matched_pairs:
        # Translation error (2D center distance in BEV)
        pred_center = np.array(pred['box'][:2])
        gt_center = np.array(gt['box'][:2])
        trans_error = np.linalg.norm(pred_center - gt_center)
        translation_errors.append(trans_error)
        
        # Scale error (1 - IoU)
        iou = calculate_iou_3d(pred['box'], gt['box'])
        scale_error = 1.0 - iou
        scale_errors.append(scale_error)
        
        # Orientation error
        orient_error = calculate_orientation_error(pred['box'], gt['box'])
        orientation_errors.append(orient_error)
        
        # Velocity error (if available)
        if 'velocity' in pred and 'velocity' in gt:
            pred_vel = np.array(pred['velocity'])
            gt_vel = np.array(gt['velocity'])
            vel_error = np.linalg.norm(pred_vel - gt_vel)
            velocity_errors.append(vel_error)
        
        # Attribute error (if available)
        if 'attributes' in pred and 'attributes' in gt:
            attr_error = 0.0 if pred['attributes'] == gt['attributes'] else 1.0
            attribute_errors.append(attr_error)
    
    # Calculate average errors
    ate = np.mean(translation_errors) if translation_errors else 1.0
    ase = np.mean(scale_errors) if scale_errors else 1.0
    aoe = np.mean(orientation_errors) if orientation_errors else 1.0
    ave = np.mean(velocity_errors) if velocity_errors else 1.0
    aae = np.mean(attribute_errors) if attribute_errors else 1.0
    
    return {
        'ate': float(ate),
        'ase': float(ase),
        'aoe': float(aoe),
        'ave': float(ave),
        'aae': float(aae),
        'num_tp': len(matched_pairs)
    }


def calculate_nds_detailed(
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5
) -> Dict[str, Union[float, Dict]]:
    """
    Calculate NDS with detailed breakdown of all components.
    
    Returns:
        Dictionary containing:
            - 'nds': Overall NDS score
            - 'mAP': Mean Average Precision
            - 'tp_metrics': Dictionary of TP error metrics
            - 'per_class_nds': NDS for each class
    """
    from admetrics.detection.ap import calculate_map
    
    # Calculate mAP
    map_result = calculate_map(
        predictions=predictions,
        ground_truth=ground_truth,
        class_names=class_names,
        iou_thresholds=iou_threshold,
        metric_type="3d"
    )
    
    mAP = map_result['mAP']
    
    # Calculate TP metrics
    tp_metrics = calculate_tp_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=iou_threshold
    )
    
    # Calculate NDS
    error_sum = 0.0
    for metric_name in ['ate', 'ase', 'aoe', 'ave', 'aae']:
        error = tp_metrics.get(metric_name, 1.0)
        error_sum += (1.0 - min(1.0, error))
    
    nds = (5 * mAP + error_sum) / 10.0
    
    # Per-class NDS
    per_class_nds = {}
    for cls in class_names:
        cls_preds = [p for p in predictions if p.get('class') == cls]
        cls_gt = [g for g in ground_truth if g.get('class') == cls]
        
        if len(cls_gt) > 0:
            cls_nds = calculate_nds(
                predictions=cls_preds,
                ground_truth=cls_gt,
                class_names=[cls],
                iou_threshold=iou_threshold
            )
            per_class_nds[cls] = cls_nds
    
    return {
        'nds': float(nds),
        'mAP': float(mAP),
        'tp_metrics': tp_metrics,
        'per_class_nds': per_class_nds,
        'AP_per_class': map_result['AP_per_class']
    }
