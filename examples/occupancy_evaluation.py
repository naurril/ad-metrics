"""
Example usage of 3D occupancy prediction metrics.

This script demonstrates how to evaluate 3D semantic occupancy predictions
using various metrics including IoU, precision/recall, scene completion,
and geometric distance metrics.
"""

import numpy as np
from admetrics.occupancy import (
    occupancy_iou,
    mean_iou,
    occupancy_precision_recall,
    scene_completion,
    chamfer_distance,
    surface_distance,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def example_1_basic_iou():
    """Example 1: Basic IoU calculation for a single class."""
    print_section("Example 1: Basic IoU Calculation")
    
    # Create simple 3D occupancy grids (10x10x10 voxels)
    # Class 0: free space, Class 1: vehicle, Class 2: road
    pred_occupancy = np.zeros((10, 10, 10), dtype=int)
    pred_occupancy[3:7, 3:7, 3:7] = 1  # Predicted vehicle occupancy
    
    gt_occupancy = np.zeros((10, 10, 10), dtype=int)
    gt_occupancy[4:8, 4:8, 4:8] = 1  # Ground truth vehicle occupancy
    
    # Calculate IoU for vehicle class (class_id=1)
    iou = occupancy_iou(pred_occupancy, gt_occupancy, class_id=1)
    
    print(f"Predicted vehicle volume: {np.sum(pred_occupancy == 1)} voxels")
    print(f"Ground truth vehicle volume: {np.sum(gt_occupancy == 1)} voxels")
    print(f"IoU for vehicle class: {iou:.4f}")
    
    # Calculate binary occupancy IoU (occupied vs free)
    binary_iou = occupancy_iou(pred_occupancy, gt_occupancy, class_id=None)
    print(f"Binary occupancy IoU: {binary_iou:.4f}")


def example_2_multi_class_miou():
    """Example 2: Mean IoU across multiple semantic classes."""
    print_section("Example 2: Multi-Class Mean IoU")
    
    # Create occupancy grid with multiple classes
    # 0: free, 1: vehicle, 2: pedestrian, 3: road, 4: building
    np.random.seed(42)
    pred_occupancy = np.zeros((20, 20, 20), dtype=int)
    gt_occupancy = np.zeros((20, 20, 20), dtype=int)
    
    # Road (ground plane)
    pred_occupancy[:, :, 0:2] = 3
    gt_occupancy[:, :, 0:2] = 3
    
    # Vehicle
    pred_occupancy[5:9, 5:9, 2:5] = 1
    gt_occupancy[6:10, 6:10, 2:5] = 1
    
    # Pedestrian
    pred_occupancy[12:14, 12:14, 2:4] = 2
    gt_occupancy[11:13, 11:13, 2:4] = 2
    
    # Building
    pred_occupancy[15:20, 0:5, 2:10] = 4
    gt_occupancy[15:20, 0:5, 2:11] = 4
    
    # Calculate mean IoU
    num_classes = 5
    result = mean_iou(pred_occupancy, gt_occupancy, num_classes=num_classes)
    
    print(f"Overall mIoU: {result['mIoU']:.4f}")
    print(f"Number of valid classes: {result['valid_classes']}")
    print("\nPer-class IoU:")
    class_names = ['free', 'vehicle', 'pedestrian', 'road', 'building']
    for class_id, class_name in enumerate(class_names):
        iou = result['class_iou'][class_id]
        if not np.isnan(iou):
            print(f"  {class_name}: {iou:.4f}")
        else:
            print(f"  {class_name}: N/A (class not present)")


def example_3_precision_recall():
    """Example 3: Precision, Recall, and F1-Score."""
    print_section("Example 3: Precision, Recall, and F1-Score")
    
    # Create occupancy grids with some errors
    np.random.seed(42)
    pred_occupancy = np.zeros((15, 15, 15), dtype=int)
    gt_occupancy = np.zeros((15, 15, 15), dtype=int)
    
    # Ground truth occupied region
    gt_occupancy[5:10, 5:10, 5:10] = 1
    
    # Prediction: slightly shifted and smaller
    pred_occupancy[6:9, 6:9, 6:9] = 1
    # Add some false positives
    pred_occupancy[12:14, 12:14, 12:14] = 1
    
    metrics = occupancy_precision_recall(pred_occupancy, gt_occupancy, class_id=1)
    
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")


def example_4_scene_completion():
    """Example 4: Scene Completion metrics."""
    print_section("Example 4: Scene Completion Metrics")
    
    # Create scene with multiple objects
    np.random.seed(42)
    pred_occupancy = np.zeros((25, 25, 25), dtype=int)
    gt_occupancy = np.zeros((25, 25, 25), dtype=int)
    
    # Ground truth: multiple objects with different classes
    gt_occupancy[:, :, 0:3] = 1  # Road
    gt_occupancy[5:10, 5:10, 3:7] = 2  # Vehicle 1
    gt_occupancy[15:18, 15:18, 3:5] = 2  # Vehicle 2
    gt_occupancy[8:10, 15:17, 3:5] = 3  # Pedestrian
    
    # Prediction: partially complete scene
    pred_occupancy[:, :, 0:3] = 1  # Road (correct)
    pred_occupancy[6:9, 6:9, 3:6] = 2  # Vehicle 1 (partial)
    pred_occupancy[15:18, 15:18, 3:5] = 2  # Vehicle 2 (correct)
    # Missing pedestrian
    
    sc_metrics = scene_completion(pred_occupancy, gt_occupancy, free_class=0)
    
    print("Scene Completion Metrics:")
    print(f"  SC IoU (occupied vs free): {sc_metrics['SC_IoU']:.4f}")
    print(f"  SC Precision: {sc_metrics['SC_Precision']:.4f}")
    print(f"  SC Recall: {sc_metrics['SC_Recall']:.4f}")
    print(f"  Semantic Scene Completion mIoU: {sc_metrics['SSC_mIoU']:.4f}")
    print(f"  Completion Ratio: {sc_metrics['completion_ratio']:.4f}")
    
    if sc_metrics['completion_ratio'] < 1.0:
        print(f"\n  ⚠ Under-completion detected (missing {(1-sc_metrics['completion_ratio'])*100:.1f}% of occupied voxels)")
    elif sc_metrics['completion_ratio'] > 1.0:
        print(f"\n  ⚠ Over-completion detected (predicting {(sc_metrics['completion_ratio']-1)*100:.1f}% extra occupied voxels)")


def example_5_chamfer_distance():
    """Example 5: Chamfer Distance between point clouds."""
    print_section("Example 5: Chamfer Distance")
    
    # Extract occupied voxel centers as point clouds
    np.random.seed(42)
    
    # Create two similar but not identical occupancy patterns
    pred_occupancy = np.zeros((20, 20, 20), dtype=int)
    pred_occupancy[5:15, 5:15, 5:15] = 1
    
    gt_occupancy = np.zeros((20, 20, 20), dtype=int)
    gt_occupancy[6:14, 6:14, 6:14] = 1
    
    # Get occupied voxel coordinates
    pred_points = np.argwhere(pred_occupancy > 0).astype(float)
    gt_points = np.argwhere(gt_occupancy > 0).astype(float)
    
    print(f"Predicted occupied voxels: {len(pred_points)}")
    print(f"Ground truth occupied voxels: {len(gt_points)}")
    
    # Calculate Chamfer Distance
    cd_metrics = chamfer_distance(pred_points, gt_points, bidirectional=True)
    
    print(f"\nChamfer Distance Metrics:")
    print(f"  Bidirectional Chamfer Distance: {cd_metrics['chamfer_distance']:.4f} voxels")
    print(f"  Pred → GT distance: {cd_metrics['pred_to_gt']:.4f} voxels")
    print(f"  GT → Pred distance: {cd_metrics['gt_to_pred']:.4f} voxels")


def example_6_surface_distance():
    """Example 6: Surface distance metrics."""
    print_section("Example 6: Surface Distance Metrics")
    
    # Create occupancy grids representing objects with surfaces
    pred_occupancy = np.zeros((30, 30, 30), dtype=int)
    pred_occupancy[8:22, 8:22, 8:22] = 1  # Large cube
    
    gt_occupancy = np.zeros((30, 30, 30), dtype=int)
    gt_occupancy[10:20, 10:20, 10:20] = 1  # Smaller cube
    
    # Calculate surface distances with voxel size of 0.2 meters
    voxel_size = 0.2
    sd_metrics = surface_distance(
        pred_occupancy, gt_occupancy, 
        voxel_size=voxel_size, 
        percentile=95
    )
    
    print(f"Surface Distance Metrics (voxel size: {voxel_size}m):")
    print(f"  Mean surface distance: {sd_metrics['mean_surface_distance']:.4f} m")
    print(f"  Median surface distance: {sd_metrics['median_surface_distance']:.4f} m")
    print(f"  Std surface distance: {sd_metrics['std_surface_distance']:.4f} m")
    print(f"  Max surface distance: {sd_metrics['max_surface_distance']:.4f} m")
    print(f"  95th percentile distance: {sd_metrics['percentile_distance']:.4f} m")


def example_7_realistic_scenario():
    """Example 7: Realistic autonomous driving scenario."""
    print_section("Example 7: Realistic Autonomous Driving Scenario")
    
    # Simulate a realistic occupancy prediction scenario
    # Grid: 50x50x20 voxels, voxel size: 0.2m (10m x 10m x 4m space)
    np.random.seed(42)
    
    grid_size = (50, 50, 20)
    pred_occupancy = np.zeros(grid_size, dtype=int)
    gt_occupancy = np.zeros(grid_size, dtype=int)
    
    # Class definitions:
    # 0: free space
    # 1: road surface
    # 2: vehicle
    # 3: pedestrian
    # 4: building/static obstacle
    
    # Road surface (bottom layer)
    gt_occupancy[:, :, 0:2] = 1
    pred_occupancy[:, :, 0:2] = 1
    
    # Ego vehicle (center)
    gt_occupancy[22:28, 22:28, 2:5] = 2
    pred_occupancy[22:28, 22:28, 2:5] = 2  # Correct prediction
    
    # Other vehicle (offset)
    gt_occupancy[10:16, 35:41, 2:6] = 2
    pred_occupancy[11:15, 35:40, 2:5] = 2  # Slightly smaller prediction
    
    # Pedestrian
    gt_occupancy[35:37, 15:17, 2:5] = 3
    pred_occupancy[35:37, 15:17, 2:5] = 3  # Correct
    
    # Building
    gt_occupancy[0:10, 0:10, 2:15] = 4
    pred_occupancy[0:10, 0:10, 2:14] = 4  # Slightly lower height
    
    # Another pedestrian (missed by prediction)
    gt_occupancy[40:42, 30:32, 2:5] = 3
    
    print("Scenario: Urban intersection with ego vehicle, other vehicles, pedestrians, and buildings")
    print(f"Grid size: {grid_size} voxels")
    print(f"Physical size: 10m x 10m x 4m (voxel size: 0.2m)\n")
    
    # Calculate all metrics
    num_classes = 5
    miou_result = mean_iou(pred_occupancy, gt_occupancy, num_classes=num_classes)
    
    print("1. Mean IoU Metrics:")
    print(f"   Overall mIoU: {miou_result['mIoU']:.4f}")
    
    class_names = ['free', 'road', 'vehicle', 'pedestrian', 'building']
    for i, name in enumerate(class_names):
        iou = miou_result['class_iou'][i]
        if not np.isnan(iou):
            print(f"   {name}: {iou:.4f}")
    
    # Scene completion
    sc = scene_completion(pred_occupancy, gt_occupancy, free_class=0)
    print(f"\n2. Scene Completion:")
    print(f"   SC IoU: {sc['SC_IoU']:.4f}")
    print(f"   SSC mIoU: {sc['SSC_mIoU']:.4f}")
    print(f"   Completion ratio: {sc['completion_ratio']:.4f}")
    
    # Per-class precision/recall for vehicles
    vehicle_metrics = occupancy_precision_recall(pred_occupancy, gt_occupancy, class_id=2)
    print(f"\n3. Vehicle Detection Performance:")
    print(f"   Precision: {vehicle_metrics['precision']:.4f}")
    print(f"   Recall: {vehicle_metrics['recall']:.4f}")
    print(f"   F1-Score: {vehicle_metrics['f1']:.4f}")
    
    # Pedestrian detection
    ped_metrics = occupancy_precision_recall(pred_occupancy, gt_occupancy, class_id=3)
    print(f"\n4. Pedestrian Detection Performance:")
    print(f"   Precision: {ped_metrics['precision']:.4f}")
    print(f"   Recall: {ped_metrics['recall']:.4f}")
    print(f"   F1-Score: {ped_metrics['f1']:.4f}")
    
    # Surface distance
    sd = surface_distance(pred_occupancy, gt_occupancy, voxel_size=0.2, percentile=95)
    print(f"\n5. Surface Distance Metrics:")
    print(f"   Mean: {sd['mean_surface_distance']:.4f} m")
    print(f"   95th percentile: {sd['percentile_distance']:.4f} m")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("3D Occupancy Prediction Metrics - Usage Examples")
    print("=" * 80)
    
    example_1_basic_iou()
    example_2_multi_class_miou()
    example_3_precision_recall()
    example_4_scene_completion()
    example_5_chamfer_distance()
    example_6_surface_distance()
    example_7_realistic_scenario()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")
