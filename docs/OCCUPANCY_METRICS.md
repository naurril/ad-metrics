# 3D Occupancy Prediction Metrics

This document describes the metrics for evaluating 3D semantic occupancy prediction models in autonomous driving.

## Overview

3D occupancy prediction involves predicting which voxels in a 3D grid around the vehicle are occupied and their semantic classes (e.g., vehicle, pedestrian, road, building). This is a fundamental task for scene understanding and planning in autonomous driving.

## Metrics Categories

### 1. Voxel-wise Classification Metrics

#### Intersection over Union (IoU)
Measures the overlap between predicted and ground truth occupancy for a specific class.

**Formula:**
```
IoU = Intersection / Union
    = TP / (TP + FP + FN)
```

**Properties:**
- Range: [0, 1], higher is better
- 1.0 = perfect prediction
- 0.0 = no overlap

**Usage:**
```python
from admetrics.occupancy import occupancy_iou

# Calculate IoU for a specific class
iou = occupancy_iou(pred_occupancy, gt_occupancy, class_id=1)

# Calculate binary IoU (occupied vs free)
binary_iou = occupancy_iou(pred_occupancy, gt_occupancy, class_id=None)
```

**Parameters:**
- `pred_occupancy`: Predicted occupancy grid (X, Y, Z) with class labels
- `gt_occupancy`: Ground truth occupancy grid (X, Y, Z) with class labels
- `class_id`: Class ID to evaluate (None for binary occupancy)
- `ignore_index`: Label value to ignore (default: 255)

---

#### Mean Intersection over Union (mIoU)
Average IoU across all semantic classes, the primary metric for semantic occupancy prediction.

**Formula:**
```
mIoU = (1/C) * Σ(IoU_c) for c in valid_classes
```

**Properties:**
- Range: [0, 1], higher is better
- Only averages over classes present in the data
- Standard metric for semantic segmentation tasks

**Usage:**
```python
from admetrics.occupancy import mean_iou

result = mean_iou(pred_occupancy, gt_occupancy, num_classes=5)
print(f"mIoU: {result['mIoU']}")
print(f"Per-class IoU: {result['class_iou']}")
```

**Returns:**
- `mIoU`: Mean IoU across valid classes
- `class_iou`: Dictionary of per-class IoU scores
- `valid_classes`: Number of classes with non-zero union

---

#### Precision, Recall, and F1-Score
Standard classification metrics applied to voxel predictions.

**Formulas:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

**Usage:**
```python
from admetrics.occupancy import occupancy_precision_recall

metrics = occupancy_precision_recall(pred_occupancy, gt_occupancy, class_id=1)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

**Applications:**
- Evaluating detection vs. missed detections trade-off
- Analyzing per-class performance
- Understanding false positive/negative distributions

---

### 2. Scene Completion Metrics

#### Scene Completion IoU (SC-IoU)
Measures how well the model completes the scene by predicting both occupied and free space.

**Formula:**
```
SC-IoU = IoU(occupied_pred, occupied_gt)
```

**Properties:**
- Binary metric: occupied vs. free space
- Evaluates geometric completion quality
- Independent of semantic class accuracy

---

#### Semantic Scene Completion mIoU (SSC-mIoU)
Mean IoU computed only over occupied voxels with semantic classes.

**Formula:**
```
SSC-mIoU = mIoU over non-free classes
```

**Properties:**
- Focuses on semantic accuracy of occupied regions
- Excludes free space from calculation
- Combined metric for completion + semantics

---

#### Completion Ratio
Ratio of predicted occupied voxels to ground truth occupied voxels.

**Formula:**
```
Completion_Ratio = |predicted_occupied| / |gt_occupied|
```

**Properties:**
- Ratio = 1.0: perfect completion
- Ratio < 1.0: under-completion (missing objects)
- Ratio > 1.0: over-completion (false occupancy)

**Usage:**
```python
from admetrics.occupancy import scene_completion

sc = scene_completion(pred_occupancy, gt_occupancy, free_class=0)
print(f"SC IoU: {sc['SC_IoU']:.4f}")
print(f"SSC mIoU: {sc['SSC_mIoU']:.4f}")
print(f"Completion Ratio: {sc['completion_ratio']:.4f}")
```

---

### 3. Geometric Quality Metrics

#### Chamfer Distance (CD)
Measures the average distance between predicted and ground truth point clouds.

**Formula:**
```
CD_pred→gt = (1/|P|) * Σ min ||p - g||₂  for p in P, g in G
CD_gt→pred = (1/|G|) * Σ min ||g - p||₂  for g in G, p in P
CD = (CD_pred→gt + CD_gt→pred) / 2
```

Where:
- P = predicted occupied voxel centers
- G = ground truth occupied voxel centers

**Properties:**
- Measures point-to-point distances
- Symmetric (bidirectional) version recommended
- Sensitive to shape and boundary accuracy
- Lower is better

**Usage:**
```python
from admetrics.occupancy import chamfer_distance

# Extract occupied voxel coordinates
pred_points = np.argwhere(pred_occupancy > 0).astype(float)
gt_points = np.argwhere(gt_occupancy > 0).astype(float)

cd = chamfer_distance(pred_points, gt_points, bidirectional=True)
print(f"Chamfer Distance: {cd['chamfer_distance']:.4f} voxels")
```

**Applications:**
- Evaluating geometric accuracy
- Measuring shape similarity
- Detecting boundary errors

---

#### Surface Distance (SD)
Measures distances between predicted and ground truth surfaces.

**Method:**
1. Extract surface voxels (occupied voxels with at least one free neighbor)
2. Compute distances from predicted surface to nearest GT surface voxel
3. Calculate statistics (mean, median, percentiles)

**Properties:**
- Focuses on boundary accuracy
- More sensitive than Chamfer Distance to surface misalignments
- Scales with voxel size parameter
- Lower is better

**Usage:**
```python
from admetrics.occupancy import surface_distance

sd = surface_distance(
    pred_occupancy, 
    gt_occupancy, 
    voxel_size=0.2,  # meters
    percentile=95
)
print(f"Mean surface distance: {sd['mean_surface_distance']:.4f} m")
print(f"95th percentile: {sd['percentile_distance']:.4f} m")
```

**Applications:**
- Evaluating reconstruction quality
- Measuring boundary alignment
- Detecting systematic shape errors

---

## Benchmark Datasets

### nuScenes-Occupancy
- **Description**: 3D occupancy annotations for nuScenes dataset
- **Resolution**: 200 x 200 x 16 voxels (0.5m resolution)
- **Classes**: 16 semantic classes
- **Primary Metric**: mIoU

### SemanticKITTI
- **Description**: Semantic segmentation of LiDAR point clouds
- **Resolution**: Variable (point cloud based)
- **Classes**: 19 semantic classes
- **Primary Metric**: mIoU

### Occ3D
- **Description**: Large-scale 3D occupancy prediction benchmark
- **Resolution**: 200 x 200 x 16 voxels
- **Classes**: 16 semantic classes
- **Primary Metric**: mIoU, Ray-based metrics

---

## Best Practices

### 1. Metric Selection

**For Semantic Occupancy:**
- Primary: mIoU
- Secondary: Per-class IoU, Precision/Recall

**For Scene Completion:**
- Primary: SC-IoU, SSC-mIoU
- Secondary: Completion Ratio

**For Geometric Quality:**
- Primary: Surface Distance
- Secondary: Chamfer Distance

### 2. Evaluation Protocol

1. **Resolution Matching**: Ensure pred and GT use same voxel resolution
2. **Coordinate Alignment**: Align coordinate systems before evaluation
3. **Ignore Index**: Use consistent ignore index (255) for unknown/invalid voxels
4. **Class Balance**: Report per-class metrics to identify weak classes

### 3. Common Pitfalls

**Pitfall 1: Ignoring Free Space**
- Some datasets don't annotate all free space
- Use `ignore_index` to exclude unknown regions

**Pitfall 2: Resolution Mismatch**
- Different models may predict at different resolutions
- Resample to common resolution before comparison

**Pitfall 3: Class Imbalance**
- Rare classes (e.g., pedestrians) may have low IoU
- Consider class-weighted metrics or per-class analysis

**Pitfall 4: Boundary Ambiguity**
- Voxel discretization creates boundary uncertainty
- Use surface distance metrics for boundary evaluation

---

## Example: Complete Evaluation

```python
import numpy as np
from admetrics.occupancy import (
    mean_iou, 
    scene_completion,
    surface_distance,
    occupancy_precision_recall
)

# Load predictions and ground truth
pred_occupancy = load_prediction()  # Shape: (X, Y, Z)
gt_occupancy = load_ground_truth()  # Shape: (X, Y, Z)

# 1. Semantic occupancy evaluation
miou_result = mean_iou(pred_occupancy, gt_occupancy, num_classes=16)
print(f"mIoU: {miou_result['mIoU']:.4f}")

# 2. Scene completion
sc = scene_completion(pred_occupancy, gt_occupancy, free_class=0)
print(f"SC-IoU: {sc['SC_IoU']:.4f}")
print(f"SSC-mIoU: {sc['SSC_mIoU']:.4f}")

# 3. Per-class analysis (e.g., vehicles)
vehicle_metrics = occupancy_precision_recall(
    pred_occupancy, gt_occupancy, class_id=1
)
print(f"Vehicle Precision: {vehicle_metrics['precision']:.4f}")
print(f"Vehicle Recall: {vehicle_metrics['recall']:.4f}")

# 4. Geometric quality
sd = surface_distance(pred_occupancy, gt_occupancy, voxel_size=0.5)
print(f"Mean surface distance: {sd['mean_surface_distance']:.4f} m")
```

---

## Performance Benchmarks

Typical performance ranges for state-of-the-art models:

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| mIoU | < 20% | 20-35% | 35-50% | > 50% |
| SC-IoU | < 40% | 40-60% | 60-80% | > 80% |
| Surface Distance | > 1.0m | 0.5-1.0m | 0.2-0.5m | < 0.2m |
| Vehicle IoU | < 40% | 40-60% | 60-75% | > 75% |

---

## References

### Foundational Methods

1. **SSCNet: Semantic Scene Completion from a Single Depth Image**
   - Song, S., Yu, F., Zeng, A., Chang, A.X., Savva, M., & Funkhouser, T. (2017)
   - CVPR 2017
   - https://arxiv.org/abs/1611.08974
   - https://doi.org/10.48550/arXiv.1611.08974
   - First end-to-end 3D convolutional network for semantic scene completion from single depth image

2. **MonoScene: Monocular 3D Semantic Scene Completion**
   - Cao, A.Q., & de Charette, R. (2022)
   - CVPR 2022
   - https://arxiv.org/abs/2112.00726
   - https://github.com/astra-vision/MonoScene
   - https://astra-vision.github.io/MonoScene/
   - First method to predict 3D semantic occupancy from a single monocular RGB image

3. **TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction**
   - Huang, Y., Zheng, W., Zhang, Y., Zhou, J., & Lu, J. (2023)
   - CVPR 2023
   - https://arxiv.org/abs/2302.07817
   - https://doi.org/10.48550/arXiv.2302.07817
   - https://github.com/wzzheng/TPVFormer
   - https://wzzheng.net/TPVFormer/
   - Introduces tri-perspective view representation to accompany BEV with perpendicular planes

### Benchmarks and Datasets

4. **SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences**
   - Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., & Gall, J. (2019)
   - ICCV 2019
   - https://arxiv.org/abs/1904.01416
   - https://doi.org/10.48550/arXiv.1904.01416
   - http://semantic-kitti.org/
   - Large-scale LiDAR-based semantic segmentation and scene completion benchmark

5. **Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving**
   - Tian, X., Jiang, T., Yun, L., Mao, Y., Yang, H., Wang, Y., Wang, Y., & Zhao, H. (2023)
   - NeurIPS 2023
   - https://arxiv.org/abs/2304.14365
   - https://doi.org/10.48550/arXiv.2304.14365
   - https://github.com/Tsinghua-MARS-Lab/Occ3D
   - https://tsinghua-mars-lab.github.io/Occ3D/
   - Dense occupancy labels for nuScenes and Waymo datasets with 200×200×16 voxel resolution

6. **nuScenes: A multimodal dataset for autonomous driving**
   - Caesar, H., Bankiti, V., Lang, A.H., et al. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1903.11027
   - https://www.nuscenes.org/
   - Full sensor suite dataset used as base for Occ3D-nuScenes benchmark

7. **Waymo Open Dataset: Scalable Autonomous Driving**
   - Sun, P., Kretzschmar, H., Dotiwalla, X., et al. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1912.04838
   - https://waymo.com/open
   - Large-scale autonomous driving dataset used as base for Occ3D-Waymo benchmark

## See Also

- [DETECTION_METRICS.md](DETECTION_METRICS.md) - 3D object detection metrics
- [LOCALIZATION_METRICS.md](LOCALIZATION_METRICS.md) - Ego-pose accuracy metrics
- [TRACKING_METRICS.md](TRACKING_METRICS.md) - Multi-object tracking metrics
- [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) - HD map prediction metrics
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Quick reference for all metrics
