"""
NuScenes Detection Evaluation Example.

This script demonstrates evaluation using nuScenes detection metrics,
including NDS (nuScenes Detection Score).
"""

import numpy as np
from admetrics import calculate_nds, calculate_map
from admetrics.nds import calculate_nds_detailed, calculate_tp_metrics


def create_nuscenes_sample_data():
    """
    Create sample data in nuScenes format.
    
    nuScenes uses specific class names and includes velocity information.
    """
    
    # nuScenes detection classes
    nuscenes_classes = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    ]
    
    # Sample predictions with velocity
    predictions = [
        # Cars
        {
            'box': [10.0, 5.0, 0.5, 1.9, 1.6, 4.7, 0.1],  # [x, y, z, w, h, l, yaw]
            'score': 0.92,
            'class': 'car',
            'velocity': [5.2, 0.3],  # [vx, vy] in m/s
            'attributes': 'vehicle.moving'
        },
        {
            'box': [20.0, -3.0, 0.5, 1.8, 1.5, 4.5, -0.05],
            'score': 0.88,
            'class': 'car',
            'velocity': [8.1, -0.2],
            'attributes': 'vehicle.moving'
        },
        {
            'box': [30.0, 10.0, 0.5, 1.9, 1.6, 4.6, 0.0],
            'score': 0.85,
            'class': 'car',
            'velocity': [0.0, 0.0],
            'attributes': 'vehicle.parked'
        },
        
        # Pedestrians
        {
            'box': [8.0, 2.0, 0.3, 0.7, 1.8, 0.6, 0.0],
            'score': 0.78,
            'class': 'pedestrian',
            'velocity': [1.2, 0.5],
            'attributes': 'pedestrian.moving'
        },
        {
            'box': [15.0, -5.0, 0.3, 0.6, 1.7, 0.6, 0.0],
            'score': 0.72,
            'class': 'pedestrian',
            'velocity': [1.5, -0.3],
            'attributes': 'pedestrian.moving'
        },
        
        # Truck
        {
            'box': [25.0, 8.0, 1.0, 2.5, 2.8, 7.2, 0.15],
            'score': 0.90,
            'class': 'truck',
            'velocity': [6.0, 0.5],
            'attributes': 'vehicle.moving'
        },
        
        # Bicycle
        {
            'box': [12.0, 3.0, 0.4, 0.7, 1.3, 1.8, 0.2],
            'score': 0.68,
            'class': 'bicycle',
            'velocity': [3.5, 0.8],
            'attributes': 'cycle.with_rider'
        },
    ]
    
    # Ground truth
    ground_truth = [
        # Cars
        {
            'box': [10.2, 5.1, 0.5, 1.9, 1.6, 4.7, 0.12],
            'class': 'car',
            'velocity': [5.3, 0.4],
            'attributes': 'vehicle.moving',
            'num_lidar_pts': 450,
            'num_radar_pts': 5
        },
        {
            'box': [20.3, -2.9, 0.5, 1.8, 1.5, 4.5, -0.03],
            'class': 'car',
            'velocity': [8.0, -0.1],
            'attributes': 'vehicle.moving',
            'num_lidar_pts': 380,
            'num_radar_pts': 6
        },
        {
            'box': [30.5, 10.2, 0.5, 1.9, 1.6, 4.6, 0.02],
            'class': 'car',
            'velocity': [0.1, 0.0],
            'attributes': 'vehicle.parked',
            'num_lidar_pts': 520,
            'num_radar_pts': 2
        },
        
        # Pedestrians
        {
            'box': [8.1, 2.1, 0.3, 0.7, 1.8, 0.6, 0.02],
            'class': 'pedestrian',
            'velocity': [1.3, 0.6],
            'attributes': 'pedestrian.moving',
            'num_lidar_pts': 45,
            'num_radar_pts': 1
        },
        {
            'box': [15.2, -4.9, 0.3, 0.6, 1.7, 0.6, 0.01],
            'class': 'pedestrian',
            'velocity': [1.4, -0.2],
            'attributes': 'pedestrian.moving',
            'num_lidar_pts': 38,
            'num_radar_pts': 1
        },
        
        # Truck
        {
            'box': [25.3, 8.2, 1.0, 2.5, 2.8, 7.2, 0.18],
            'class': 'truck',
            'velocity': [6.2, 0.6],
            'attributes': 'vehicle.moving',
            'num_lidar_pts': 680,
            'num_radar_pts': 8
        },
        
        # Bicycle
        {
            'box': [12.2, 3.1, 0.4, 0.7, 1.3, 1.8, 0.22],
            'class': 'bicycle',
            'velocity': [3.6, 0.9],
            'attributes': 'cycle.with_rider',
            'num_lidar_pts': 65,
            'num_radar_pts': 2
        },
        
        # Missed detection
        {
            'box': [50.0, 20.0, 0.5, 1.9, 1.6, 4.7, 0.0],
            'class': 'car',
            'velocity': [10.0, 0.0],
            'attributes': 'vehicle.moving',
            'num_lidar_pts': 120,
            'num_radar_pts': 3
        },
    ]
    
    return predictions, ground_truth


def evaluate_nuscenes_nds():
    """Calculate nuScenes Detection Score (NDS)."""
    print("\n" + "=" * 80)
    print("NuScenes Detection Score (NDS) Evaluation")
    print("=" * 80)
    
    predictions, ground_truth = create_nuscenes_sample_data()
    
    # NuScenes detection classes
    detection_classes = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    ]
    
    # Calculate NDS with detailed breakdown
    result = calculate_nds_detailed(
        predictions=predictions,
        ground_truth=ground_truth,
        class_names=detection_classes,
        iou_threshold=0.5  # NuScenes typically uses center distance, but we use IoU here
    )
    
    print(f"\nOverall Metrics:")
    print(f"  NDS (nuScenes Detection Score): {result['nds']:.4f}")
    print(f"  mAP (Mean Average Precision):   {result['mAP']:.4f}")
    
    print(f"\nTrue Positive Error Metrics:")
    tp_metrics = result['tp_metrics']
    print(f"  ATE (Translation Error):  {tp_metrics['ate']:.4f} m")
    print(f"  ASE (Scale Error):        {tp_metrics['ase']:.4f}")
    print(f"  AOE (Orientation Error):  {tp_metrics['aoe']:.4f} rad")
    print(f"  AVE (Velocity Error):     {tp_metrics['ave']:.4f} m/s")
    print(f"  AAE (Attribute Error):    {tp_metrics['aae']:.4f}")
    
    print(f"\nPer-Class NDS:")
    for cls, nds_score in result['per_class_nds'].items():
        print(f"  {cls:<20}: {nds_score:.4f}")
    
    print(f"\nPer-Class AP:")
    for cls, ap_score in result['AP_per_class'].items():
        print(f"  {cls:<20}: {ap_score:.4f}")


def evaluate_nuscenes_map():
    """Calculate mAP for nuScenes."""
    print("\n" + "=" * 80)
    print("NuScenes mAP Evaluation")
    print("=" * 80)
    
    predictions, ground_truth = create_nuscenes_sample_data()
    
    detection_classes = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    ]
    
    # NuScenes uses multiple distance thresholds
    # Here we use IoU thresholds as a simplification
    iou_thresholds = [0.5, 0.7]
    
    result = calculate_map(
        predictions=predictions,
        ground_truth=ground_truth,
        class_names=detection_classes,
        iou_thresholds=iou_thresholds,
        metric_type='3d'
    )
    
    print(f"\nOverall mAP: {result['mAP']:.4f}")
    
    print(f"\nmAP per IoU threshold:")
    for thr, ap in result['AP_per_threshold'].items():
        print(f"  IoU@{thr}: {ap:.4f}")
    
    print(f"\nAP per class (averaged over thresholds):")
    for cls, ap in sorted(result['AP_per_class'].items(), key=lambda x: -x[1]):
        if ap > 0:
            print(f"  {cls:<20}: {ap:.4f}")


def evaluate_tp_errors():
    """Detailed analysis of True Positive errors."""
    print("\n" + "=" * 80)
    print("True Positive Error Analysis")
    print("=" * 80)
    
    predictions, ground_truth = create_nuscenes_sample_data()
    
    tp_metrics = calculate_tp_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5
    )
    
    print(f"\nError Metrics for Matched Detections:")
    print(f"  Number of True Positives: {tp_metrics['num_tp']}")
    print(f"\nAverage Errors:")
    print(f"  Translation (ATE): {tp_metrics['ate']:.4f} meters")
    print(f"    - 2D Euclidean distance between predicted and GT centers")
    print(f"  ")
    print(f"  Scale (ASE): {tp_metrics['ase']:.4f}")
    print(f"    - 1 - IoU (lower is better)")
    print(f"  ")
    print(f"  Orientation (AOE): {tp_metrics['aoe']:.4f} radians ({np.degrees(tp_metrics['aoe']):.2f}°)")
    print(f"    - Angular difference in yaw")
    print(f"  ")
    print(f"  Velocity (AVE): {tp_metrics['ave']:.4f} m/s")
    print(f"    - L2 distance between velocity vectors")
    print(f"  ")
    print(f"  Attribute (AAE): {tp_metrics['aae']:.4f}")
    print(f"    - 0 if attributes match, 1 otherwise")


def compare_detection_ranges():
    """Analyze performance at different distance ranges."""
    print("\n" + "=" * 80)
    print("Performance by Distance Range")
    print("=" * 80)
    
    from admetrics.distance import calculate_translation_error_bins
    
    predictions, ground_truth = create_nuscenes_sample_data()
    
    # Define distance bins
    distance_bins = [0, 10, 20, 30, 50, 100]
    
    bin_results = calculate_translation_error_bins(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5,
        bins=distance_bins
    )
    
    print(f"\n{'Distance Range':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<10}")
    print("-" * 70)
    
    for bin_name, counts in bin_results.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        print(f"{bin_name:<15} {tp:<6} {fp:<6} {fn:<6} {precision:<12.4f} {recall:<10.4f}")


def main():
    """Run nuScenes evaluation examples."""
    print("\n" + "=" * 80)
    print("NuScenes 3D Object Detection - Evaluation Example")
    print("=" * 80)
    
    # Run evaluations
    evaluate_nuscenes_nds()
    evaluate_nuscenes_map()
    evaluate_tp_errors()
    compare_detection_ranges()
    
    print("\n" + "=" * 80)
    print("NuScenes evaluation completed!")
    print("=" * 80)
    print("\nNote: This example uses synthetic data for demonstration.")
    print("NuScenes uses center distance thresholds, not IoU.")
    print("Adapt the code for actual nuScenes evaluation protocol.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
