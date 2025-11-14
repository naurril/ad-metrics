"""
Basic usage example for metrics3d library.

This script demonstrates how to use the metrics3d library to evaluate
3D object detection results.
"""

import numpy as np
from admetrics import (
    calculate_iou_3d,
    calculate_ap,
    calculate_map,
    calculate_nds,
    calculate_confusion_metrics
)
from admetrics.utils import match_detections, nms_3d


def create_sample_data():
    """Create sample prediction and ground truth data."""
    
    # Sample predictions with confidence scores
    predictions = [
        {
            'box': [10.0, 5.0, 0.5, 4.5, 1.8, 2.0, 0.1],  # [x, y, z, w, h, l, yaw]
            'score': 0.95,
            'class': 'car'
        },
        {
            'box': [20.0, 10.0, 0.5, 4.2, 1.7, 1.8, 0.05],
            'score': 0.88,
            'class': 'car'
        },
        {
            'box': [15.0, 2.0, 0.3, 0.8, 1.9, 0.6, 0.0],
            'score': 0.82,
            'class': 'pedestrian'
        },
        {
            'box': [5.0, 8.0, 0.4, 1.5, 1.2, 1.8, 0.2],
            'score': 0.75,
            'class': 'cyclist'
        },
        {
            'box': [100.0, 100.0, 0.5, 4.0, 1.8, 2.0, 0.0],  # False positive
            'score': 0.65,
            'class': 'car'
        }
    ]
    
    # Ground truth annotations
    ground_truth = [
        {
            'box': [10.2, 5.1, 0.5, 4.5, 1.8, 2.0, 0.12],
            'class': 'car'
        },
        {
            'box': [20.3, 10.2, 0.5, 4.0, 1.7, 1.9, 0.08],
            'class': 'car'
        },
        {
            'box': [15.1, 2.1, 0.3, 0.8, 1.9, 0.6, 0.02],
            'class': 'pedestrian'
        },
        {
            'box': [5.2, 8.1, 0.4, 1.5, 1.2, 1.8, 0.18],
            'class': 'cyclist'
        },
        {
            'box': [30.0, 15.0, 0.5, 4.0, 1.8, 2.0, 0.0],  # Missed detection
            'class': 'car'
        }
    ]
    
    return predictions, ground_truth


def example_iou_calculation():
    """Example: Calculate IoU between two boxes."""
    print("=" * 80)
    print("Example 1: IoU Calculation")
    print("=" * 80)
    
    box1 = [0, 0, 0, 4, 2, 1.5, 0]
    box2 = [1, 0, 0, 4, 2, 1.5, 0]
    
    iou_3d = calculate_iou_3d(box1, box2)
    
    print(f"\nBox 1: {box1}")
    print(f"Box 2: {box2}")
    print(f"\n3D IoU: {iou_3d:.4f}")
    print()


def example_average_precision():
    """Example: Calculate Average Precision."""
    print("=" * 80)
    print("Example 2: Average Precision")
    print("=" * 80)
    
    predictions, ground_truth = create_sample_data()
    
    # Filter for car class
    car_preds = [p for p in predictions if p['class'] == 'car']
    car_gt = [g for g in ground_truth if g['class'] == 'car']
    
    result = calculate_ap(
        predictions=car_preds,
        ground_truth=car_gt,
        iou_threshold=0.7,
        num_recall_points=40
    )
    
    print(f"\nClass: car")
    print(f"IoU Threshold: 0.7")
    print(f"\nResults:")
    print(f"  AP: {result['ap']:.4f}")
    print(f"  True Positives: {result['num_tp']}")
    print(f"  False Positives: {result['num_fp']}")
    print(f"  Ground Truth: {result['num_gt']}")
    print()


def example_mean_average_precision():
    """Example: Calculate mAP across multiple classes."""
    print("=" * 80)
    print("Example 3: Mean Average Precision (mAP)")
    print("=" * 80)
    
    predictions, ground_truth = create_sample_data()
    
    class_names = ['car', 'pedestrian', 'cyclist']
    iou_thresholds = [0.5, 0.7]
    
    result = calculate_map(
        predictions=predictions,
        ground_truth=ground_truth,
        class_names=class_names,
        iou_thresholds=iou_thresholds,
        metric_type='3d'
    )
    
    print(f"\nOverall mAP: {result['mAP']:.4f}")
    print(f"\nAP per class:")
    for cls, ap in result['AP_per_class'].items():
        print(f"  {cls}: {ap:.4f}")
    
    print(f"\nAP per IoU threshold:")
    for thr, ap in result['AP_per_threshold'].items():
        print(f"  IoU@{thr}: {ap:.4f}")
    print()


def example_confusion_metrics():
    """Example: Calculate confusion metrics."""
    print("=" * 80)
    print("Example 4: Confusion Metrics (Precision, Recall, F1)")
    print("=" * 80)
    
    predictions, ground_truth = create_sample_data()
    
    metrics = calculate_confusion_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5
    )
    
    print(f"\nIoU Threshold: 0.5")
    print(f"\nMetrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"\nCounts:")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print()


def example_nms():
    """Example: Non-Maximum Suppression."""
    print("=" * 80)
    print("Example 5: Non-Maximum Suppression (NMS)")
    print("=" * 80)
    
    # Create overlapping detections
    boxes = [
        {'box': [10, 5, 0.5, 4.5, 1.8, 2.0, 0.1], 'score': 0.95, 'class': 'car'},
        {'box': [10.3, 5.2, 0.5, 4.5, 1.8, 2.0, 0.1], 'score': 0.88, 'class': 'car'},  # Duplicate
        {'box': [10.1, 5.1, 0.5, 4.5, 1.8, 2.0, 0.1], 'score': 0.82, 'class': 'car'},  # Duplicate
        {'box': [20, 10, 0.5, 4.2, 1.7, 1.8, 0.05], 'score': 0.90, 'class': 'car'},
    ]
    
    print(f"\nBefore NMS: {len(boxes)} detections")
    
    keep_indices = nms_3d(boxes, iou_threshold=0.5, score_threshold=0.0)
    
    print(f"After NMS: {len(keep_indices)} detections")
    print(f"\nKept indices: {keep_indices}")
    print(f"\nKept detections:")
    for idx in keep_indices:
        box = boxes[idx]
        print(f"  Index {idx}: score={box['score']:.2f}, class={box['class']}")
    print()


def example_detection_matching():
    """Example: Detection matching."""
    print("=" * 80)
    print("Example 6: Detection Matching")
    print("=" * 80)
    
    predictions, ground_truth = create_sample_data()
    
    # Use greedy matching
    matches, unmatched_preds, unmatched_gts = match_detections(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5,
        method='greedy'
    )
    
    print(f"\nMatching Results (IoU threshold: 0.5):")
    print(f"  Matches: {len(matches)}")
    print(f"  Unmatched predictions: {len(unmatched_preds)}")
    print(f"  Unmatched ground truth: {len(unmatched_gts)}")
    
    print(f"\nMatched pairs (pred_idx, gt_idx):")
    for pred_idx, gt_idx in matches:
        pred_class = predictions[pred_idx]['class']
        gt_class = ground_truth[gt_idx]['class']
        print(f"  Prediction {pred_idx} ({pred_class}) <-> Ground Truth {gt_idx} ({gt_class})")
    
    if unmatched_preds:
        print(f"\nUnmatched predictions (False Positives):")
        for idx in unmatched_preds:
            print(f"  Index {idx}: {predictions[idx]['class']}, score={predictions[idx]['score']:.2f}")
    
    if unmatched_gts:
        print(f"\nUnmatched ground truth (False Negatives):")
        for idx in unmatched_gts:
            print(f"  Index {idx}: {ground_truth[idx]['class']}")
    print()


def example_nuscenes_detection_score():
    """Example: NuScenes Detection Score."""
    print("=" * 80)
    print("Example 7: NuScenes Detection Score (NDS)")
    print("=" * 80)
    
    # Create data with velocity information
    predictions = [
        {
            'box': [10.0, 5.0, 0.5, 4.5, 1.8, 2.0, 0.1],
            'score': 0.95,
            'class': 'car',
            'velocity': [5.0, 0.0]
        },
        {
            'box': [15.0, 2.0, 0.3, 0.8, 1.9, 0.6, 0.0],
            'score': 0.82,
            'class': 'pedestrian',
            'velocity': [1.0, 0.5]
        }
    ]
    
    ground_truth = [
        {
            'box': [10.2, 5.1, 0.5, 4.5, 1.8, 2.0, 0.12],
            'class': 'car',
            'velocity': [5.2, 0.1]
        },
        {
            'box': [15.1, 2.1, 0.3, 0.8, 1.9, 0.6, 0.02],
            'class': 'pedestrian',
            'velocity': [1.1, 0.6]
        }
    ]
    
    nds_score = calculate_nds(
        predictions=predictions,
        ground_truth=ground_truth,
        class_names=['car', 'pedestrian'],
        iou_threshold=0.5
    )
    
    print(f"\nNuScenes Detection Score (NDS): {nds_score:.4f}")
    print(f"\nNDS combines mAP with error metrics:")
    print(f"  - Average Translation Error (ATE)")
    print(f"  - Average Scale Error (ASE)")
    print(f"  - Average Orientation Error (AOE)")
    print(f"  - Average Velocity Error (AVE)")
    print(f"  - Average Attribute Error (AAE)")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("3D Object Detection Metrics - Usage Examples")
    print("=" * 80 + "\n")
    
    # Run examples
    example_iou_calculation()
    example_average_precision()
    example_mean_average_precision()
    example_confusion_metrics()
    example_nms()
    example_detection_matching()
    example_nuscenes_detection_score()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
