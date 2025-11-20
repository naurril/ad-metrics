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
    metric_type: str = "3d",
    matching: Union[str, callable] = "iou",
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
    # Support two kinds of matching: IoU-based (default) or center-distance-based.
    # When `metric_type` == 'bev' or '3d' and IoU is used, we call the iou helpers.
    use_center_distance = False
    from admetrics.detection.iou import calculate_iou_3d, calculate_iou_bev

    # Default iou-based affinity function
    iou_func = calculate_iou_3d if metric_type == "3d" else calculate_iou_bev

    # Handle matching selection
    if isinstance(matching, str):
        if matching == 'center_distance':
            # define a center-distance function (2D ground plane)
            def _center_dist(pb, gb):
                try:
                    dx = float(pb[0]) - float(gb[0])
                    dy = float(pb[1]) - float(gb[1])
                    return float(np.hypot(dx, dy))
                except Exception:
                    return float('inf')

            iou_func = _center_dist
            use_center_distance = True
        elif matching == 'iou':
            # keep iou_func as set above
            use_center_distance = False
        else:
            raise ValueError("matching must be 'iou' or 'center_distance' or a callable")
    elif callable(matching):
        iou_func = matching
        use_center_distance = False
    else:
        raise ValueError("matching must be 'iou' or 'center_distance' or a callable")
    
    # Validate prediction entries
    for p in predictions:
        if 'score' not in p:
            raise ValueError("Each prediction must include a 'score' field")
        if 'box' not in p:
            raise ValueError("Each prediction must include a 'box' field")
        if 'class' not in p:
            raise ValueError("Each prediction must include a 'class' field")

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
        max_iou = float('inf') if use_center_distance else 0
        max_gt_idx = -1
        
        # Find best matching ground truth box
        for gt_idx, gt in enumerate(ground_truth):
            # Check if same class
            if pred.get('class') != gt.get('class'):
                continue
            
            # Skip if already matched
            if gt_matched[gt_idx]:
                continue
            
            # Calculate affinity (IoU or center-distance)
            affinity = 0.0
            try:
                affinity = iou_func(pred['box'], gt['box'])
            except Exception:
                affinity = 0.0

            # For IoU we want the maximum affinity; for center-distance (smaller is better)
            # we treat affinity as a distance and select the smallest value.
            if use_center_distance:
                # here lower is better
                if max_gt_idx == -1 or affinity < max_iou:
                    max_iou = affinity
                    max_gt_idx = gt_idx
            else:
                if affinity > max_iou:
                    max_iou = affinity
                    max_gt_idx = gt_idx
        
        # Assign TP or FP
        # Decide TP/FP depending on whether affinity is IoU (>= threshold)
        # or center-distance (<= threshold).
        if use_center_distance:
            match_condition = (max_gt_idx >= 0 and max_iou <= iou_threshold)
        else:
            match_condition = (max_gt_idx >= 0 and max_iou >= iou_threshold)

        if match_condition:
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
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    
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


def calculate_map_multi_frame(
    predictions_per_frame: Dict[int, List[Dict]],
    ground_truth_per_frame: Dict[int, List[Dict]],
    class_names: List[str],
    mode: str = 'framewise_then_global',
    sample_rate: int = 1,
    iou_thresholds: Union[List[float], float] = 0.5,
    num_recall_points: int = 40,
    metric_type: str = '3d',
    matching: Union[str, callable] = 'iou'
) -> Dict[str, Union[float, Dict]]:
    """
    Compute mAP for multi-frame detection data with several modes.

        Modes (default: 'framewise_then_global'):
            - 'framewise_then_global' (recommended): match per-frame first, collect TP/FP with scores, sort globally, compute global mAP.
            - 'concatenate': flatten all frames into single lists and compute mAP.
      - 'per_frame': compute per-frame mAP (using `calculate_map`) and return the mean across frames.
      - 'sample': sample every `sample_rate`-th frame, flatten those frames and compute mAP.

    Args:
        predictions_per_frame: Dict mapping frame_id -> list of prediction dicts
        ground_truth_per_frame: Dict mapping frame_id -> list of gt dicts
        class_names: list of classes to evaluate
        sample_rate: for 'sample' mode, take every Nth frame (N=sample_rate)
        iou_thresholds: IoU threshold(s) for AP
        num_recall_points: recall interpolation points
        metric_type: '3d' or 'bev'

    Returns:
        Dict with keys similar to `calculate_map`, plus mode-specific fields:
          - 'mAP': overall mAP
          - 'AP_per_class', 'AP_per_threshold'
          - 'frames_used': list of frame ids used
          - For 'per_frame': 'per_frame_mAPs' mapping frame->mAP and 'mAP_mean'
    """
    
    frame_ids = sorted(set(list(predictions_per_frame.keys()) + list(ground_truth_per_frame.keys())))

    if len(frame_ids) == 0:
        return {'mAP': 0.0, 'AP_per_class': {}, 'AP_per_threshold': {}, 'mode': mode, 'frames_used': []}

    # Handle simple modes first
    if mode == 'concatenate':
        # Flatten all frames into single lists
        all_preds = []
        all_gts = []
        for fid in frame_ids:
            all_preds.extend(predictions_per_frame.get(fid, []))
            all_gts.extend(ground_truth_per_frame.get(fid, []))
        return calculate_map(all_preds, all_gts, class_names, iou_thresholds=iou_thresholds, num_recall_points=num_recall_points, metric_type=metric_type, matching=matching)

    if mode == 'sample':
        sampled_ids = frame_ids[::max(1, sample_rate)]
        all_preds = []
        all_gts = []
        for fid in sampled_ids:
            all_preds.extend(predictions_per_frame.get(fid, []))
            all_gts.extend(ground_truth_per_frame.get(fid, []))
        return calculate_map(all_preds, all_gts, class_names, iou_thresholds=iou_thresholds, num_recall_points=num_recall_points, metric_type=metric_type, matching=matching)

    if mode == 'per_frame':
        per_frame_mAPs = {}
        for fid in frame_ids:
            res = calculate_map(predictions_per_frame.get(fid, []), ground_truth_per_frame.get(fid, []), class_names, iou_thresholds=iou_thresholds, num_recall_points=num_recall_points, metric_type=metric_type, matching=matching)
            per_frame_mAPs[fid] = res.get('mAP', 0.0)
        return {'mode': mode, 'per_frame_mAPs': per_frame_mAPs, 'mAP_mean': float(np.mean(list(per_frame_mAPs.values()))) if len(per_frame_mAPs) > 0 else 0.0}

    # Default: 'framewise_then_global' (nuScenes-style aggregation)
    from admetrics.detection.iou import calculate_iou_3d, calculate_iou_bev

    # Prepare thresholds list
    if isinstance(iou_thresholds, (int, float)):
        thresholds = [float(iou_thresholds)]
    else:
        thresholds = [float(t) for t in iou_thresholds]

    # Choose affinity function once
    if isinstance(matching, str) and matching == 'center_distance':
        def _center_dist(pb, gb):
            try:
                dx = float(pb[0]) - float(gb[0])
                dy = float(pb[1]) - float(gb[1])
                return float(np.hypot(dx, dy))
            except Exception:
                return float('inf')

        affinity_fn = _center_dist
        use_center_distance = True
    elif isinstance(matching, str) and matching == 'iou':
        affinity_fn = calculate_iou_3d if metric_type == '3d' else calculate_iou_bev
        use_center_distance = False
    elif callable(matching):
        affinity_fn = matching
        use_center_distance = False
    else:
        raise ValueError("matching must be 'iou' or 'center_distance' or a callable")

    AP_per_threshold = {}
    AP_per_class = {c: [] for c in class_names}
    total_gt_all = 0
    total_preds_all = 0

    # For each threshold, compute per-class APs and then average
    for thr in thresholds:
        ap_per_class_this_thr = {}
        for cls in class_names:
            # collect across frames for this class
            scores = []
            tps = []
            fps = []
            total_gt_cls = 0

            for fid in frame_ids:
                preds = [p for p in predictions_per_frame.get(fid, []) if p.get('class') == cls]
                preds = sorted(preds, key=lambda x: x.get('score', 0.0), reverse=True)
                gts = [g for g in ground_truth_per_frame.get(fid, []) if g.get('class') == cls]
                gt_matched = [False] * len(gts)
                total_gt_cls += len(gts)

                for pred in preds:
                    score = float(pred.get('score', 0.0))
                    max_aff = (float('inf') if use_center_distance else 0.0)
                    max_gt_idx = -1
                    for gt_idx, gt in enumerate(gts):
                        if gt_matched[gt_idx]:
                            continue
                        try:
                            aff = affinity_fn(pred['box'], gt['box'])
                        except Exception:
                            aff = (float('inf') if use_center_distance else 0.0)

                        if use_center_distance:
                            if max_gt_idx == -1 or aff < max_aff:
                                max_aff = aff
                                max_gt_idx = gt_idx
                        else:
                            if aff > max_aff:
                                max_aff = aff
                                max_gt_idx = gt_idx

                    # determine match
                    if use_center_distance:
                        match = (max_gt_idx >= 0 and max_aff <= thr)
                    else:
                        match = (max_gt_idx >= 0 and max_aff >= thr)

                    if match:
                        scores.append(score)
                        tps.append(1)
                        fps.append(0)
                        gt_matched[max_gt_idx] = True
                    else:
                        scores.append(score)
                        tps.append(0)
                        fps.append(1)

            total_gt_all += total_gt_cls
            total_preds_all += len(scores)

            if len(scores) == 0 or total_gt_cls == 0:
                ap_val = 0.0
            else:
                order = np.argsort(-np.array(scores))
                tp_arr = np.array(tps)[order]
                fp_arr = np.array(fps)[order]
                tp_cumsum = np.cumsum(tp_arr)
                fp_cumsum = np.cumsum(fp_arr)
                recalls = tp_cumsum / max(total_gt_cls, 1)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
                ap_val = _compute_ap_interp(recalls, precisions, num_recall_points)

            ap_per_class_this_thr[cls] = float(ap_val)
            AP_per_class[cls].append(float(ap_val))

        # average across classes for this threshold
        AP_per_threshold[thr] = float(np.mean(list(ap_per_class_this_thr.values()))) if len(ap_per_class_this_thr) > 0 else 0.0

    # Final mAP: mean across thresholds (nuScenes averages thresholds then classes)
    mAP = float(np.mean(list(AP_per_threshold.values()))) if len(AP_per_threshold) > 0 else 0.0

    # Average AP per class across thresholds
    ap_per_class_avg = {c: float(np.mean(vals)) if len(vals) > 0 else 0.0 for c, vals in AP_per_class.items()}

    return {
        'mAP': mAP,
        'AP_per_class': ap_per_class_avg,
        'AP_per_threshold': AP_per_threshold,
        'mode': mode,
        'frames_used': frame_ids,
        'num_gt': int(total_gt_all),
        'num_preds': int(total_preds_all)
    }


def calculate_map(
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_names: List[str],
    iou_thresholds: Union[List[float], float] = 0.5,
    num_recall_points: int = 40,
    metric_type: str = "3d",
    matching: Union[str, callable] = 'iou'
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
    # Validate that all predictions include a class field; tests expect a
    # ValueError when a prediction is missing 'class'.
    for p in predictions:
        if 'class' not in p:
            raise ValueError("Each prediction must include a 'class' field")
    
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
            if len(cls_preds) > 0:
                for iou_thr in iou_thresholds:
                    ap_per_class[cls].append(0.0)
                    ap_per_threshold[iou_thr].append(0.0)
                    all_aps.append(0.0)
            continue
        
        for iou_thr in iou_thresholds:
            result = calculate_ap(
                predictions=cls_preds,
                ground_truth=cls_gt,
                iou_threshold=iou_thr,
                num_recall_points=num_recall_points,
                metric_type=metric_type,
                matching=matching
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
    metric_type: str = "3d",
    matching: Union[str, callable] = 'iou'
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
        metric_type=metric_type,
        matching=matching
    )
    
    return result['precision'], result['recall'], result['scores']


def calculate_coco_metrics(*args, **kwargs):
    """
    Backwards-compatible alias for COCO-style AP calculation.

    This function wraps :func:`calculate_ap_coco_style` for projects that
    reference the older name ``calculate_coco_metrics`` in the documentation.
    """
    return calculate_ap_coco_style(*args, **kwargs)