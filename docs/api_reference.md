# API Reference

Complete API documentation for the admetrics library - a comprehensive metrics suite for autonomous driving evaluation.

## Core Metrics

### IoU Metrics (`admetrics.iou`)

#### `calculate_iou_3d(box1, box2, box_format='xyzwhlr')`

Calculate 3D Intersection over Union between two 3D bounding boxes.

**Parameters:**
- `box1` (array-like): First 3D bounding box `[x, y, z, w, h, l, rotation]`
  - `x, y, z`: Center coordinates
  - `w`: Width, `h`: Height, `l`: Length
  - `rotation`: Yaw angle in radians
- `box2` (array-like): Second 3D bounding box in same format
- `box_format` (str): Format of boxes ('xyzwhlr' or 'xyzhwlr')

**Returns:**
- `float`: IoU value between 0 and 1

**Example:**
```python
from admetrics import calculate_iou_3d

box1 = [0, 0, 0, 4, 2, 1.5, 0]
box2 = [1, 0, 0, 4, 2, 1.5, 0]
iou = calculate_iou_3d(box1, box2)
print(f"3D IoU: {iou:.4f}")
```

---

#### `calculate_iou_bev(box1, box2, box_format='xyzwhlr')`

Calculate Bird's Eye View (BEV) IoU by projecting boxes to the x-y plane.

**Parameters:**
- `box1` (array-like): First 3D bounding box
- `box2` (array-like): Second 3D bounding box
- `box_format` (str): Format of boxes

**Returns:**
- `float`: BEV IoU value between 0 and 1

---

#### `calculate_iou_batch(boxes1, boxes2, box_format='xyzwhlr', mode='3d')`

Calculate IoU for batches of boxes efficiently.

**Parameters:**
- `boxes1` (np.ndarray): Shape (N, 7) array of N bounding boxes
- `boxes2` (np.ndarray): Shape (M, 7) array of M bounding boxes
- `box_format` (str): Format of boxes
- `mode` (str): '3d' for 3D IoU, 'bev' for BEV IoU

**Returns:**
- `np.ndarray`: Shape (N, M) array of IoU values

---

#### `calculate_giou_3d(box1, box2, box_format='xyzwhlr')`

Calculate Generalized IoU (GIoU) for 3D boxes.

GIoU = IoU - |C - (A ∪ B)| / |C|, where C is the smallest enclosing box.

**Parameters:**
- `box1` (array-like): First box
- `box2` (array-like): Second box
- `box_format` (str): Format of boxes

**Returns:**
- `float`: GIoU value between -1 and 1

---

### Average Precision (`admetrics.ap`)

#### `calculate_ap(predictions, ground_truth, iou_threshold=0.5, num_recall_points=40, metric_type='3d')`

Calculate Average Precision for 3D object detection.

**Parameters:**
- `predictions` (List[Dict]): List of prediction dicts with keys:
  - `'box'`: 3D bounding box [x, y, z, w, h, l, r]
  - `'score'`: Confidence score
  - `'class'`: Class name
- `ground_truth` (List[Dict]): List of ground truth dicts with keys:
  - `'box'`: 3D bounding box
  - `'class'`: Class name
  - `'difficulty'`: (optional) Difficulty level
- `iou_threshold` (float): IoU threshold for matching
- `num_recall_points` (int): Number of recall points for interpolation
- `metric_type` (str): '3d' or 'bev'

**Returns:**
- `Dict`: Dictionary containing:
  - `'ap'`: Average Precision value
  - `'precision'`: Precision values
  - `'recall'`: Recall values
  - `'scores'`: Confidence scores
  - `'num_tp'`, `'num_fp'`, `'num_gt'`: Counts

**Example:**
```python
from admetrics import calculate_ap

predictions = [
    {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
    {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}
]
ground_truth = [
    {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
]
result = calculate_ap(predictions, ground_truth)
print(f"AP: {result['ap']:.4f}")
```

---

#### `calculate_map(predictions, ground_truth, class_names, iou_thresholds=0.5, num_recall_points=40, metric_type='3d')`

Calculate Mean Average Precision across multiple classes and IoU thresholds.

**Parameters:**
- `predictions` (List[Dict]): All predictions
- `ground_truth` (List[Dict]): All ground truth
- `class_names` (List[str]): List of class names to evaluate
- `iou_thresholds` (float or List[float]): Single threshold or list
- `num_recall_points` (int): Number of recall points
- `metric_type` (str): '3d' or 'bev'

**Returns:**
- `Dict`: Dictionary containing:
  - `'mAP'`: Overall mean AP
  - `'AP_per_class'`: AP for each class
  - `'AP_per_threshold'`: AP for each IoU threshold
  - `'num_classes'`: Number of evaluated classes

---

#### `calculate_ap_coco_style(predictions, ground_truth, iou_thresholds=None)`

Calculate AP using COCO-style evaluation (average over IoU thresholds 0.5:0.95).

**Returns:**
- `Dict`: Dictionary with:
  - `'AP'`: Average over [0.5:0.95]
  - `'AP50'`: AP at IoU=0.5
  - `'AP75'`: AP at IoU=0.75
  - `'AP_per_threshold'`: AP for each threshold

---

### NuScenes Detection Score (`admetrics.nds`)

#### `calculate_nds(predictions, ground_truth, class_names, iou_threshold=0.5, distance_thresholds=None)`

Calculate nuScenes Detection Score (NDS).

NDS combines mAP with error metrics: Translation, Scale, Orientation, Velocity, and Attribute errors.

**Parameters:**
- `predictions` (List[Dict]): Predictions with keys:
  - `'box'`: [x, y, z, w, h, l, yaw]
  - `'score'`: Confidence score
  - `'class'`: Class name
  - `'velocity'`: (optional) [vx, vy]
  - `'attributes'`: (optional) Attribute predictions
- `ground_truth` (List[Dict]): Ground truth annotations
- `class_names` (List[str]): List of class names
- `iou_threshold` (float): IoU threshold
- `distance_thresholds` (List[float], optional): Distance thresholds per class

**Returns:**
- `float`: NDS score (0-1)

---

#### `calculate_nds_detailed(predictions, ground_truth, class_names, iou_threshold=0.5)`

Calculate NDS with detailed breakdown.

**Returns:**
- `Dict`: Dictionary containing:
  - `'nds'`: Overall NDS score
  - `'mAP'`: Mean Average Precision
  - `'tp_metrics'`: Dict of TP error metrics (ATE, ASE, AOE, AVE, AAE)
  - `'per_class_nds'`: NDS for each class
  - `'AP_per_class'`: AP for each class

---

### Average Orientation Similarity (`admetrics.aos`)

#### `calculate_aos(predictions, ground_truth, iou_threshold=0.7, num_recall_points=40)`

Calculate Average Orientation Similarity (AOS) for KITTI benchmark.

AOS combines detection accuracy with orientation estimation quality.

**Parameters:**
- `predictions` (List[Dict]): Predictions
- `ground_truth` (List[Dict]): Ground truth
- `iou_threshold` (float): IoU threshold (KITTI uses 0.7 for cars)
- `num_recall_points` (int): Number of recall points

**Returns:**
- `Dict`: Dictionary with:
  - `'aos'`: Average Orientation Similarity
  - `'ap'`: Average Precision
  - `'orientation_similarity'`: Mean orientation similarity for TPs
  - `'num_tp'`, `'num_fp'`, `'num_gt'`: Counts

---

### Confusion Metrics (`admetrics.confusion`)

#### `calculate_tp_fp_fn(predictions, ground_truth, iou_threshold=0.5, metric_type='3d')`

Calculate True Positives, False Positives, and False Negatives.

**Returns:**
- `Dict`: Dictionary with `'tp'`, `'fp'`, `'fn'` counts

---

#### `calculate_confusion_metrics(predictions, ground_truth, iou_threshold=0.5, metric_type='3d')`

Calculate comprehensive confusion matrix metrics.

**Returns:**
- `Dict`: Dictionary with:
  - `'precision'`: TP / (TP + FP)
  - `'recall'`: TP / (TP + FN)
  - `'f1_score'`: 2 * (precision * recall) / (precision + recall)
  - `'tp'`, `'fp'`, `'fn'`: Raw counts

---

#### `calculate_confusion_matrix_multiclass(predictions, ground_truth, class_names, iou_threshold=0.5, metric_type='3d')`

Calculate confusion matrix for multi-class detection.

**Returns:**
- `Dict`: Dictionary with:
  - `'confusion_matrix'`: (N, N) numpy array
  - `'class_names'`: List of class names
  - `'per_class_metrics'`: Metrics for each class

---

### Distance Metrics (`admetrics.distance`)

#### `calculate_center_distance(box1, box2, distance_type='euclidean')`

Calculate distance between centers of two 3D bounding boxes.

**Parameters:**
- `box1`, `box2`: Boxes [x, y, z, w, h, l, r]
- `distance_type` (str): 'euclidean' (3D), 'bev' (2D), or 'vertical' (z-axis)

**Returns:**
- `float`: Distance in meters

---

#### `calculate_orientation_error(box1, box2, error_type='absolute')`

Calculate orientation/heading error.

**Parameters:**
- `error_type` (str): 'absolute' (radians) or 'degrees'

**Returns:**
- `float`: Orientation error

---

## Utility Functions

### Transforms (`admetrics.utils.transforms`)

#### `transform_box(box, translation=None, rotation=None, scale=None)`

Apply transformation to a 3D bounding box.

#### `rotate_box(box, rotation, origin=None)`

Rotate a box around a point.

#### `translate_box(box, translation)`

Translate a box.

#### `convert_box_format(box, src_format, dst_format)`

Convert between box formats ('xyzwhlr', 'xyzhwlr', 'corners').

#### `center_to_corners(box)`

Convert center-based box to 8 corner points.

#### `corners_to_center(corners)`

Convert 8 corner points to center-based representation.

---

### Matching (`admetrics.utils.matching`)

#### `match_detections(predictions, ground_truth, iou_threshold=0.5, method='greedy', metric_type='3d')`

Match predictions to ground truth boxes.

**Parameters:**
- `method` (str): 'greedy' or 'hungarian'

**Returns:**
- `Tuple`: (matches, unmatched_preds, unmatched_gts)
  - `matches`: List of (pred_idx, gt_idx) tuples
  - `unmatched_preds`: List of unmatched prediction indices
  - `unmatched_gts`: List of unmatched GT indices

---

### NMS (`admetrics.utils.nms`)

#### `nms_3d(boxes, scores=None, iou_threshold=0.5, score_threshold=0.0)`

3D Non-Maximum Suppression.

**Parameters:**
- `boxes`: List of dicts with 'box' and 'score', or (N, 7) array
- `scores` (np.ndarray): Scores if boxes is array
- `iou_threshold` (float): IoU threshold for suppression
- `score_threshold` (float): Minimum score threshold

**Returns:**
- `List[int]`: Indices of boxes to keep

#### `nms_bev(boxes, scores=None, iou_threshold=0.5, score_threshold=0.0)`

Bird's Eye View NMS.

#### `nms_per_class(boxes, iou_threshold=0.5, score_threshold=0.0, mode='3d')`

Apply NMS separately for each class.

---

### Visualization (`admetrics.utils.visualization`)

#### `plot_boxes_3d(boxes, labels=None, colors=None, ax=None, show=True)`

Plot 3D bounding boxes.

#### `plot_boxes_bev(boxes, labels=None, colors=None, ax=None, show=True, x_range=None, y_range=None)`

Plot Bird's Eye View of boxes.

#### `plot_precision_recall_curve(precision, recall, ap, title='Precision-Recall Curve', ax=None, show=True)`

Plot precision-recall curve.

#### `visualize_detection_results(predictions, ground_truth, matches, mode='bev', show=True)`

Visualize detection results with TP, FP, FN highlighted.

#### `plot_confusion_matrix(confusion_matrix, class_names, normalize=False, cmap='Blues', show=True)`

Plot confusion matrix.
