# Detection Metrics

Comprehensive guide to 3D object detection metrics for autonomous driving.

## Table of Contents

1. [3D IoU (Intersection over Union)](#3d-iou)
2. [Bird's Eye View (BEV) IoU](#bev-iou)
3. [Average Precision (AP)](#average-precision)
4. [Mean Average Precision (mAP)](#mean-average-precision)
5. [NuScenes Detection Score (NDS)](#nuscenes-detection-score)
6. [Average Orientation Similarity (AOS)](#average-orientation-similarity)
7. [Confusion Metrics](#confusion-metrics)
8. [Distance-Based Metrics](#distance-based-metrics)

---

## 3D IoU (Intersection over Union)

### Definition

3D IoU measures the overlap between two 3D bounding boxes in 3D space.

$$
\text{IoU}_{3D} = \frac{\text{Volume of Intersection}}{\text{Volume of Union}}
$$

### Calculation Steps

1. **BEV Intersection**: Calculate the intersection area in the bird's eye view (x-y plane)
2. **Height Overlap**: Calculate the overlap in the z-axis
3. **3D Intersection**: Multiply BEV intersection area by height overlap
4. **3D Union**: Calculate total volume occupied by both boxes
5. **IoU**: Divide intersection by union

### Properties

- **Range**: [0, 1]
- **Perfect Match**: IoU = 1.0 when boxes are identical
- **No Overlap**: IoU = 0.0 when boxes don't intersect
- **Rotation Aware**: Accounts for box orientation (yaw angle)

### Use Cases

- Primary metric for matching predictions to ground truth
- Evaluation threshold: IoU ≥ threshold (typically 0.5 or 0.7)
- KITTI uses IoU = 0.7 for cars, 0.5 for pedestrians and cyclists

### Limitations

- Sensitive to box size (small boxes have lower IoU for same error)
- Doesn't differentiate between different types of misalignment
- Discontinuous gradient (not ideal for loss functions)

---

## Bird's Eye View (BEV) IoU

### Definition

BEV IoU projects 3D boxes onto the ground plane (x-y) and calculates 2D IoU.

$$
\text{IoU}_{BEV} = \frac{\text{Area of Intersection (x-y)}}{\text{Area of Union (x-y)}}
$$

### When to Use

- When height information is less critical
- Autonomous driving scenarios (ground-level detection)
- Faster computation than full 3D IoU
- KITTI benchmark reports both 3D and BEV AP

### Advantages

- More forgiving than 3D IoU for height errors
- Computationally more efficient
- Better for localization in autonomous driving

---

## Average Precision (AP)

### Definition

AP summarizes the precision-recall curve, measuring detection quality at a specific IoU threshold.

### Calculation

1. **Sort Predictions**: Order by confidence score (descending)
2. **Match to GT**: Assign each prediction to highest-IoU ground truth
3. **TP/FP Assignment**: 
   - TP if IoU ≥ threshold and GT not already matched
   - FP otherwise
4. **Compute Precision/Recall**:
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
5. **Interpolate**: Create precision-recall curve
6. **Integrate**: AP = area under the curve

### Interpolation Methods

**N-Point Interpolation** (PASCAL VOC style):
- Sample precision at N evenly-spaced recall points
- Average the precision values
- Typical: N = 40 or 101

**11-Point Interpolation** (Original PASCAL):
- Sample at recall = [0, 0.1, 0.2, ..., 1.0]

### IoU Thresholds

Different benchmarks use different thresholds:

| Dataset | Car | Pedestrian | Cyclist | Other |
|---------|-----|------------|---------|-------|
| KITTI   | 0.7 | 0.5        | 0.5     | 0.5   |
| nuScenes| 0.5 | 0.5        | 0.5     | 0.5   |
| Waymo   | 0.7 | 0.5        | 0.5     | varies|

### Example

```python
# AP at IoU=0.7 for cars
result = calculate_ap(
    predictions=car_predictions,
    ground_truth=car_ground_truth,
    iou_threshold=0.7,
    num_recall_points=40
)
print(f"AP@0.7: {result['ap']:.4f}")
```

---

## Mean Average Precision (mAP)

### Definition

mAP averages AP across multiple classes and/or IoU thresholds.

$$
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
$$

### Variants

**mAP across Classes**:
- Average AP for each class
- Equal weight per class (not per sample)

**mAP across IoU Thresholds** (COCO style):
- AP at IoU = [0.50, 0.55, 0.60, ..., 0.95]
- Encourages better localization

**Combined**:
- Average over both classes and thresholds

### COCO Metrics

- **AP**: Average over IoU ∈ [0.5:0.95:0.05]
- **AP50**: AP at IoU = 0.5
- **AP75**: AP at IoU = 0.75
- **APsmall**, **APmedium**, **APlarge**: By object size

---

## NuScenes Detection Score (NDS)

### Definition

NDS is a composite metric that balances detection accuracy with localization quality.

$$
\text{NDS} = \frac{1}{10} \left[ 5 \times \text{mAP} + \sum_{i=1}^{5} (1 - \min(1, \text{TP}_i)) \right]
$$

### Components

**mAP** (50% weight):
- Standard mean Average Precision

**True Positive Error Metrics** (50% weight):
1. **ATE** (Average Translation Error): Mean L2 distance between centers
2. **ASE** (Average Scale Error): Mean 1 - IoU for matched pairs
3. **AOE** (Average Orientation Error): Mean angular difference in yaw
4. **AVE** (Average Velocity Error): Mean velocity vector difference
5. **AAE** (Average Attribute Error): Mean attribute mismatch rate

### Normalization

Each error metric is:
- Capped at 1.0
- Converted to score: (1 - error)
- Averaged across all true positives

### Why NDS?

- Penalizes poor localization even with correct detection
- Encourages accurate velocity estimation
- More comprehensive than AP alone
- Standard metric for nuScenes benchmark

### Example

```python
nds_score = calculate_nds(
    predictions=predictions,
    ground_truth=ground_truth,
    class_names=['car', 'pedestrian', 'cyclist']
)
# NDS typically ranges from 0.3 to 0.7 for good detectors
```

---

## Average Orientation Similarity (AOS)

### Definition

AOS extends AP by incorporating orientation accuracy, used in KITTI benchmark.

$$
\text{AOS} = \frac{1}{N} \sum_{i=1}^{N} s_i \cdot \delta_i
$$

Where:
- $s_i$ = orientation similarity = $(1 + \cos(\Delta\theta)) / 2$
- $\delta_i$ = 1 if detection (IoU ≥ threshold), 0 otherwise
- $\Delta\theta$ = angle difference between predicted and GT yaw

### Orientation Similarity

$$
s(\theta_{pred}, \theta_{gt}) = \frac{1 + \cos(\theta_{pred} - \theta_{gt})}{2}
$$

- Range: [0, 1]
- 1.0 for perfect orientation match
- 0.5 for perpendicular (90°)
- 0.0 for opposite direction (180°)

### Relationship to AP

- AOS ≤ AP (always)
- AOS = AP when all orientations are perfect
- Lower AOS indicates orientation estimation problems

### KITTI Usage

KITTI reports:
- **AP (3D)**: Standard average precision
- **AP (BEV)**: Bird's eye view AP
- **AOS**: Orientation-aware metric

All three reported for Easy/Moderate/Hard difficulties.

---

## Confusion Metrics

### True Positives (TP)

Predictions that match a ground truth box with IoU ≥ threshold.

### False Positives (FP)

Predictions that:
- Don't match any ground truth (IoU < threshold), or
- Match an already-matched ground truth (duplicates)

### False Negatives (FN)

Ground truth boxes that have no matching prediction.

### Derived Metrics

**Precision**:
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
- "What fraction of predictions are correct?"
- High precision → few false alarms

**Recall**:
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
- "What fraction of objects are detected?"
- High recall → few missed detections

**F1-Score**:
$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- Harmonic mean of precision and recall
- Balances both metrics equally

### Confusion Matrix (Multi-Class)

|              | Pred: Car | Pred: Ped | Pred: Cyclist |
|--------------|-----------|-----------|---------------|
| **GT: Car**  | TP (car)  | FP→TP     | FP→TP         |
| **GT: Ped**  | TP→FP     | TP (ped)  | FP→TP         |
| **GT: Cyclist** | TP→FP  | TP→FP     | TP (cyclist)  |

Diagonal: Correct classifications (TP)  
Off-diagonal: Misclassifications

---

## Distance-Based Metrics

### Center Distance Error

L2 distance between predicted and ground truth centers:

$$
\text{Distance} = \sqrt{(x_{pred} - x_{gt})^2 + (y_{pred} - y_{gt})^2 + (z_{pred} - z_{gt})^2}
$$

**Variants**:
- **3D Distance**: Full Euclidean distance
- **BEV Distance**: x-y plane only
- **Vertical Distance**: z-axis only

### Orientation Error

Absolute angular difference:

$$
\text{Error} = |\theta_{pred} - \theta_{gt}|
$$

Normalized to [-π, π].

### Size/Scale Error

Dimension differences:

$$
\text{Size Error} = |w_{pred} - w_{gt}| + |h_{pred} - h_{gt}| + |l_{pred} - l_{gt}|
$$

Or relative:

$$
\text{Relative Error} = \frac{|d_{pred} - d_{gt}|}{d_{gt}}
$$

### Velocity Error

For moving objects:

$$
\text{Velocity Error} = \sqrt{(v_x^{pred} - v_x^{gt})^2 + (v_y^{pred} - v_y^{gt})^2}
$$

---

## Comparison of Metrics

| Metric | Localization | Orientation | Velocity | Multi-Class | Complexity |
|--------|-------------|-------------|----------|-------------|------------|
| IoU    | ✓           | ✓           | ✗        | ✗           | Low        |
| AP     | ✓           | ✗           | ✗        | ✓           | Medium     |
| mAP    | ✓           | ✗           | ✗        | ✓           | Medium     |
| AOS    | ✓           | ✓           | ✗        | ✓           | Medium     |
| NDS    | ✓           | ✓           | ✓        | ✓           | High       |

---

## Choosing the Right Metric

**For General Detection**:
- Use **mAP** for overall performance
- Report at multiple IoU thresholds

**For KITTI Benchmark**:
- Report **AP (3D)**, **AP (BEV)**, and **AOS**
- Separate by difficulty (Easy/Moderate/Hard)

**For nuScenes Benchmark**:
- Use **NDS** as primary metric
- Report individual error components (ATE, ASE, AOE, AVE, AAE)

**For Analysis**:
- **Precision/Recall/F1** for threshold tuning
- **Confusion Matrix** for class-specific issues
- **Distance Errors** for localization debugging

**For Research**:
- Report multiple metrics for comprehensive comparison
- Include error analysis at different distances
- Consider computational cost vs. metric informativeness

---

## References

### Datasets and Benchmarks

1. **Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite**
   - Geiger, A., Lenz, P., & Urtasun, R. (2012)
   - CVPR 2012
   - http://www.cvlibs.net/datasets/kitti/
   - https://www.cvlibs.net/publications/Geiger2012CVPR.pdf
   - Pioneering benchmark for stereo, optical flow, odometry, 3D object detection, and tracking

2. **nuScenes: A multimodal dataset for autonomous driving**
   - Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., & Beijbom, O. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1903.11027
   - https://www.nuscenes.org/
   - Full sensor suite (6 cameras, 5 radars, 1 lidar), 1000 scenes with 3D bounding boxes for 23 classes, introduces NDS metric

3. **Scalability in perception for autonomous driving: Waymo open dataset**
   - Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Patnaik, V., Tsui, P., Guo, J., Zhou, Y., Chai, Y., Caine, B., Vasudevan, V., Han, W., Ngiam, J., Zhao, H., Timofeev, A., Ettinger, S., Krivokon, M., Gao, A., Joshi, A., Zhang, Y., et al. (2020)
   - CVPR 2020
   - https://arxiv.org/abs/1912.04838
   - https://waymo.com/open
   - Large-scale dataset with 1,150 scenes, high-quality LiDAR and camera data, diverse urban/suburban environments

4. **Microsoft COCO: Common Objects in Context**
   - Lin, T.Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C.L., & Dollár, P. (2014)
   - ECCV 2014
   - https://arxiv.org/abs/1405.0312
   - https://cocodataset.org/
   - Large-scale dataset with 2.5M labeled instances, defines standard AP metric computation for object detection

### Detection Methods and Metrics

5. **Focal Loss for Dense Object Detection (RetinaNet)**
   - Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017)
   - ICCV 2017
   - https://arxiv.org/abs/1708.02002
   - https://github.com/facebookresearch/Detectron
   - Introduces focal loss to address class imbalance, one-stage detector achieving high accuracy

6. **PointPillars: Fast Encoders for Object Detection from Point Clouds**
   - Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019)
   - CVPR 2019
   - https://arxiv.org/abs/1812.05784
   - Fast 3D detection using learned pillar representations, achieves 62-105 Hz on KITTI benchmark

### Additional Datasets

7. **The Cityscapes Dataset for Semantic Urban Scene Understanding**
   - Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., & Schiele, B. (2016)
   - CVPR 2016
   - https://arxiv.org/abs/1604.01685
   - https://www.cityscapes-dataset.com/
   - Large-scale dataset for urban scene understanding with pixel-level and instance-level annotations

## See Also

- [TRACKING_METRICS.md](TRACKING_METRICS.md) - Multi-object tracking metrics (CLEAR MOT, HOTA, IDF1)
- [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) - Motion forecasting metrics
- [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) - 3D occupancy prediction metrics
- [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) - HD map detection metrics
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Quick reference for all metrics
