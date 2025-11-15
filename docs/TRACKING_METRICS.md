# Multi-Object Tracking Metrics

This document provides an overview of the multi-object tracking (MOT) metrics implemented in `admetrics`.

## Overview

The tracking module (`admetrics.tracking`) provides comprehensive metrics for evaluating 3D multi-object tracking systems. These metrics go beyond single-frame detection to measure temporal consistency, identity preservation, and trajectory quality.

## Metrics

### 1. CLEAR MOT Metrics

The CLEAR MOT (Multiple Object Tracking) framework provides two fundamental metrics:

#### MOTA (Multiple Object Tracking Accuracy)

**Formula:**
```
MOTA = 1 - (FN + FP + IDSW) / GT
```

Where:
- FN: False Negatives (missed detections)
- FP: False Positives (spurious detections)
- IDSW: ID Switches (identity switches)
- GT: Total number of ground truth objects

**Characteristics:**
- Ranges from -∞ to 1.0 (can be negative with many errors)
- Penalizes detection errors AND tracking errors (ID switches)
- Higher is better
- Most commonly used tracking accuracy metric

**Usage:**
```python
from admetrics.tracking import calculate_multi_frame_mota

# predictions: Dict[frame_id -> List[detection]]
# ground_truth: Dict[frame_id -> List[detection]]
# Each detection: {'box': [...], 'track_id': int, 'class': str}

result = calculate_multi_frame_mota(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"MOTA: {result['mota']:.4f}")
print(f"ID Switches: {result['num_switches']}")
print(f"Fragmentations: {result['num_fragmentations']}")
```

#### MOTP (Multiple Object Tracking Precision)

**Formula:**
```
MOTP = sum(distance_i) / TP
```

**Characteristics:**
- Average localization error for all true positive detections
- Measured in same units as coordinates (typically meters)
- Lower is better
- Measures how precisely objects are localized, not tracking consistency

**Usage:**
```python
from admetrics.tracking import calculate_motp

result = calculate_motp(
    predictions,
    ground_truth,
    iou_threshold=0.5,
    distance_type="euclidean"  # or "bev", "vertical"
)

print(f"MOTP: {result['motp']:.4f} meters")
```

### 2. HOTA (Higher Order Tracking Accuracy)

HOTA is a more recent metric that provides a balanced view of detection and association performance.

**Formula:**
```
HOTA = sqrt(DetA × AssA)
```

Where:
- DetA: Detection Accuracy (how well objects are detected)
- AssA: Association Accuracy (how well identities are maintained)

**Characteristics:**
- Ranges from 0 to 1
- Geometric mean balances detection and association equally
- More intuitive than MOTA for comparing trackers
- Less sensitive to class imbalance
- Better for understanding tracker behavior

**Usage:**
```python
from admetrics.tracking import calculate_hota

result = calculate_hota(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"HOTA: {result['hota']:.4f}")
print(f"DetA: {result['det_a']:.4f}")
print(f"AssA: {result['ass_a']:.4f}")
```

### 3. IDF1 (ID F1 Score)

IDF1 measures identity preservation using an F1-based approach.

**Formula:**
```
IDF1 = 2 × IDTP / (2 × IDTP + IDFP + IDFN)
IDP = IDTP / (IDTP + IDFP)  # ID Precision
IDR = IDTP / (IDTP + IDFN)  # ID Recall
```

Where:
- IDTP: Correctly identified detections
- IDFP: Incorrectly identified detections
- IDFN: Missed identifications

**Characteristics:**
- Ranges from 0 to 1
- Directly measures how well identities are preserved
- F1 score balances precision and recall of ID assignments
- More sensitive to ID switches than MOTA

**Usage:**
```python
from admetrics.tracking import calculate_id_f1

result = calculate_id_f1(
    predictions,
    ground_truth,
    iou_threshold=0.5
)

print(f"IDF1: {result['idf1']:.4f}")
print(f"ID Precision: {result['idp']:.4f}")
print(f"ID Recall: {result['idr']:.4f}")
```

## Additional Metrics

### Trajectory Classification

Tracks are classified based on how much of their lifetime they are successfully tracked:

- **Mostly Tracked (MT)**: ≥80% of frames
- **Partially Tracked (PT)**: 20-80% of frames
- **Mostly Lost (ML)**: <20% of frames

```python
result = calculate_multi_frame_mota(predictions, ground_truth)
print(f"Mostly Tracked: {result['mostly_tracked']} trajectories")
print(f"Partially Tracked: {result['partially_tracked']} trajectories")
print(f"Mostly Lost: {result['mostly_lost']} trajectories")
```

### Fragmentations

Number of times a ground truth trajectory is interrupted (matched → unmatched → matched).

```python
result = calculate_multi_frame_mota(predictions, ground_truth)
print(f"Fragmentations: {result['num_fragmentations']}")
```

## Metric Comparison

| Metric | Range | Best Value | Measures | Use Case |
|--------|-------|------------|----------|----------|
| **MOTA** | -∞ to 1 | Higher | Overall tracking + detection | Standard benchmark |
| **MOTP** | 0 to ∞ | Lower | Localization precision | Complementary to MOTA |
| **HOTA** | 0 to 1 | Higher | Balanced det + assoc | Comparing trackers |
| **IDF1** | 0 to 1 | Higher | Identity consistency | ID switch analysis |

## Choosing the Right Metric

- **For benchmarking:** Use MOTA (most established, widely used)
- **For balanced evaluation:** Use HOTA (better interpretability)
- **For ID consistency:** Use IDF1 (directly measures ID preservation)
- **For localization quality:** Use MOTP (complementary to accuracy metrics)

## Common Patterns

### Perfect Tracking

```python
# All objects correctly detected with consistent IDs
result = calculate_multi_frame_mota(perfect_preds, ground_truth)
# MOTA = 1.0, num_switches = 0, num_fragmentations = 0
```

### ID Switch Detection

```python
# Track 1 switches to Track 2 for same object
result = calculate_multi_frame_mota(preds_with_switch, ground_truth)
# num_switches > 0, IDF1 decreases
```

### Fragmentation Detection

```python
# Track lost in middle frames then recovered
result = calculate_multi_frame_mota(preds_with_gaps, ground_truth)
# num_fragmentations > 0, affects trajectory classification
```

## Implementation Details

### Matching Algorithm

All metrics use Hungarian algorithm for optimal bipartite matching:
- Constructs IoU cost matrix for each frame
- Filters by class (only same-class matches)
- Uses `scipy.optimize.linear_sum_assignment`
- Threshold filtering after optimal assignment

### ID Switch Detection

ID switches are detected by tracking GT→Pred mapping across frames:
```python
if gt_id in prev_gt_to_pred:
    if prev_gt_to_pred[gt_id] != pred_id:
        # ID switch detected
```

### Fragmentation Counting

Fragmentations occur when a track is interrupted:
```python
# matched → unmatched → matched = 1 fragmentation
for track in gt_tracks:
    if has_gap_then_recovery(track):
        fragmentations += 1
```

## Examples

See `examples/tracking_evaluation.py` for comprehensive usage examples demonstrating:
- Multi-frame sequence tracking
- ID switch scenarios
- Fragmentation detection
- Trajectory classification
- Error analysis

## Testing

Comprehensive test suite in `tests/test_tracking.py`:
- 13 test cases covering all metrics
- Single-frame and multi-frame scenarios
- Perfect tracking and error cases
- ID switches, fragmentations, and trajectory classification

Run tests:
```bash
pytest tests/test_tracking.py -v
```

## References

1. **Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics**
   - Bernardin, K., & Stiefelhagen, R. (2008)
   - EURASIP Journal on Image and Video Processing, 2008
   - https://link.springer.com/article/10.1155/2008/246309
   - https://doi.org/10.1155/2008/246309
   - Introduced MOTA, MOTP, and foundational tracking metrics

2. **HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking**
   - Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2020)
   - International Journal of Computer Vision (IJCV), 2020
   - https://arxiv.org/abs/2009.07736
   - https://github.com/JonathonLuiten/TrackEval
   - https://doi.org/10.1007/s11263-020-01375-2
   - Unified metric balancing detection, association, and localization

3. **Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking**
   - Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016)
   - ECCV 2016 Workshop on Benchmarking Multi-Target Tracking
   - https://arxiv.org/abs/1609.01775
   - https://github.com/ergysr/DeepCC
   - Introduced IDF1 (ID F1 Score) for identity-preserving tracking
