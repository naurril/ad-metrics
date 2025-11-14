"""
KITTI 3D Object Detection Evaluation Example.

This script demonstrates how to evaluate 3D object detection results
using KITTI dataset format and metrics.
"""

import numpy as np
from admetrics import calculate_ap, calculate_aos
from admetrics.confusion import calculate_confusion_metrics


def load_kitti_predictions(file_path=None):
    """
    Load predictions in KITTI format.
    
    KITTI format per line:
    type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry score
    
    For this example, we'll create synthetic data.
    """
    # Example predictions for car class
    predictions = []
    
    # Format: [type, truncated, occluded, alpha, bbox_2d(4), dimensions(3), location(3), ry, score]
    kitti_data = [
        # High confidence detections
        ['Car', 0.00, 0, -1.57, 100, 120, 180, 200, 1.5, 1.8, 4.2, 10.0, 1.0, 20.0, 0.1, 0.95],
        ['Car', 0.00, 0, -1.58, 200, 130, 280, 210, 1.5, 1.7, 4.0, 20.0, 1.0, 25.0, 0.05, 0.90],
        # Medium confidence
        ['Car', 0.50, 1, -1.60, 300, 140, 380, 220, 1.6, 1.8, 4.3, 30.0, 1.0, 30.0, 0.0, 0.75],
        # Lower confidence (might be FP)
        ['Car', 0.00, 0, -1.55, 400, 150, 480, 230, 1.5, 1.8, 4.1, 100.0, 1.0, 100.0, 0.0, 0.60],
        
        # Pedestrians
        ['Pedestrian', 0.00, 0, -3.14, 150, 100, 170, 150, 1.8, 0.6, 0.8, 15.0, 0.3, 10.0, 0.0, 0.85],
        ['Pedestrian', 0.00, 0, -3.10, 250, 110, 270, 160, 1.9, 0.6, 0.8, 25.0, 0.3, 15.0, 0.02, 0.80],
        
        # Cyclists
        ['Cyclist', 0.00, 0, 0.00, 180, 120, 210, 170, 1.5, 0.8, 1.8, 18.0, 0.5, 12.0, 0.1, 0.78],
    ]
    
    for data in kitti_data:
        obj_type, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry, score = data
        
        # Convert to our format [x, y, z, w, h, l, yaw]
        box_3d = [x, y, z, w, h, l, ry]
        
        predictions.append({
            'box': box_3d,
            'score': score,
            'class': obj_type,
            'truncated': truncated,
            'occluded': occluded,
            'alpha': alpha,
            'bbox_2d': [x1, y1, x2, y2]
        })
    
    return predictions


def load_kitti_ground_truth(file_path=None):
    """
    Load ground truth in KITTI format.
    """
    # Example ground truth
    ground_truth = []
    
    kitti_gt = [
        # Cars
        ['Car', 0.00, 0, -1.57, 100, 120, 180, 200, 1.5, 1.8, 4.2, 10.2, 1.0, 20.1, 0.12, 'easy'],
        ['Car', 0.00, 0, -1.58, 200, 130, 280, 210, 1.5, 1.7, 4.0, 20.3, 1.0, 25.2, 0.08, 'easy'],
        ['Car', 0.50, 1, -1.60, 300, 140, 380, 220, 1.6, 1.8, 4.3, 30.5, 1.0, 30.5, 0.02, 'moderate'],
        ['Car', 0.00, 0, 0.00, 500, 160, 580, 240, 1.5, 1.8, 4.2, 50.0, 1.0, 50.0, 0.0, 'easy'],  # Missed
        
        # Pedestrians
        ['Pedestrian', 0.00, 0, -3.14, 150, 100, 170, 150, 1.8, 0.6, 0.8, 15.1, 0.3, 10.1, 0.02, 'easy'],
        ['Pedestrian', 0.00, 0, -3.10, 250, 110, 270, 160, 1.9, 0.6, 0.8, 25.2, 0.3, 15.2, 0.04, 'moderate'],
        
        # Cyclists
        ['Cyclist', 0.00, 0, 0.00, 180, 120, 210, 170, 1.5, 0.8, 1.8, 18.2, 0.5, 12.1, 0.12, 'easy'],
    ]
    
    for data in kitti_gt:
        obj_type, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry, difficulty = data
        
        box_3d = [x, y, z, w, h, l, ry]
        
        ground_truth.append({
            'box': box_3d,
            'class': obj_type,
            'difficulty': difficulty,
            'truncated': truncated,
            'occluded': occluded,
            'alpha': alpha,
            'bbox_2d': [x1, y1, x2, y2]
        })
    
    return ground_truth


def evaluate_kitti_car_detection():
    """Evaluate car detection using KITTI metrics."""
    print("\n" + "=" * 80)
    print("KITTI Car Detection Evaluation")
    print("=" * 80)
    
    predictions = load_kitti_predictions()
    ground_truth = load_kitti_ground_truth()
    
    # Filter for car class
    car_preds = [p for p in predictions if p['class'] == 'Car']
    car_gt = [g for g in ground_truth if g['class'] == 'Car']
    
    # KITTI uses different IoU thresholds for different classes
    # Car: 0.7, Pedestrian: 0.5, Cyclist: 0.5
    iou_threshold = 0.7
    
    # Calculate AP (3D)
    ap_3d = calculate_ap(
        predictions=car_preds,
        ground_truth=car_gt,
        iou_threshold=iou_threshold,
        metric_type='3d'
    )
    
    # Calculate AP (BEV)
    ap_bev = calculate_ap(
        predictions=car_preds,
        ground_truth=car_gt,
        iou_threshold=iou_threshold,
        metric_type='bev'
    )
    
    # Calculate AOS (Average Orientation Similarity)
    aos_result = calculate_aos(
        predictions=car_preds,
        ground_truth=car_gt,
        iou_threshold=iou_threshold
    )
    
    print(f"\nClass: Car")
    print(f"IoU Threshold: {iou_threshold}")
    print(f"\n3D Detection Metrics:")
    print(f"  AP (3D):  {ap_3d['ap']:.4f}")
    print(f"  AP (BEV): {ap_bev['ap']:.4f}")
    print(f"  AOS:      {aos_result['aos']:.4f}")
    print(f"\nOrientation Similarity: {aos_result['orientation_similarity']:.4f}")
    print(f"\nDetection Counts:")
    print(f"  True Positives:  {ap_3d['num_tp']}")
    print(f"  False Positives: {ap_3d['num_fp']}")
    print(f"  Ground Truth:    {ap_3d['num_gt']}")


def evaluate_by_difficulty():
    """Evaluate by KITTI difficulty levels (Easy, Moderate, Hard)."""
    print("\n" + "=" * 80)
    print("KITTI Evaluation by Difficulty")
    print("=" * 80)
    
    predictions = load_kitti_predictions()
    ground_truth = load_kitti_ground_truth()
    
    car_preds = [p for p in predictions if p['class'] == 'Car']
    car_gt = [g for g in ground_truth if g['class'] == 'Car']
    
    difficulties = ['easy', 'moderate', 'hard']
    iou_threshold = 0.7
    
    print(f"\nCar Detection Results (IoU@{iou_threshold}):")
    print(f"\n{'Difficulty':<12} {'AP (3D)':<10} {'AP (BEV)':<10} {'AOS':<10}")
    print("-" * 50)
    
    for difficulty in difficulties:
        # Filter GT by difficulty
        diff_gt = [g for g in car_gt if g['difficulty'] == difficulty]
        
        if len(diff_gt) == 0:
            continue
        
        # Calculate metrics
        ap_3d = calculate_ap(car_preds, diff_gt, iou_threshold, metric_type='3d')
        ap_bev = calculate_ap(car_preds, diff_gt, iou_threshold, metric_type='bev')
        aos = calculate_aos(car_preds, diff_gt, iou_threshold)
        
        print(f"{difficulty.capitalize():<12} {ap_3d['ap']:<10.4f} {ap_bev['ap']:<10.4f} {aos['aos']:<10.4f}")


def evaluate_all_classes():
    """Evaluate all classes with class-specific IoU thresholds."""
    print("\n" + "=" * 80)
    print("KITTI Multi-Class Evaluation")
    print("=" * 80)
    
    predictions = load_kitti_predictions()
    ground_truth = load_kitti_ground_truth()
    
    # KITTI class-specific IoU thresholds
    class_iou_thresholds = {
        'Car': 0.7,
        'Pedestrian': 0.5,
        'Cyclist': 0.5
    }
    
    print(f"\n{'Class':<15} {'IoU Thr':<10} {'AP (3D)':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 70)
    
    for cls, iou_thr in class_iou_thresholds.items():
        cls_preds = [p for p in predictions if p['class'] == cls]
        cls_gt = [g for g in ground_truth if g['class'] == cls]
        
        if len(cls_gt) == 0:
            continue
        
        # Calculate AP
        ap_result = calculate_ap(cls_preds, cls_gt, iou_thr, metric_type='3d')
        
        # Calculate confusion metrics
        conf_metrics = calculate_confusion_metrics(cls_preds, cls_gt, iou_thr)
        
        print(f"{cls:<15} {iou_thr:<10.1f} {ap_result['ap']:<10.4f} "
              f"{conf_metrics['precision']:<12.4f} {conf_metrics['recall']:<10.4f}")


def main():
    """Run KITTI evaluation examples."""
    print("\n" + "=" * 80)
    print("KITTI 3D Object Detection Benchmark - Evaluation Example")
    print("=" * 80)
    
    # Run evaluations
    evaluate_kitti_car_detection()
    evaluate_by_difficulty()
    evaluate_all_classes()
    
    print("\n" + "=" * 80)
    print("KITTI evaluation completed!")
    print("=" * 80)
    print("\nNote: This example uses synthetic data for demonstration.")
    print("Replace load_kitti_predictions() and load_kitti_ground_truth()")
    print("with your actual KITTI data loading functions.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
