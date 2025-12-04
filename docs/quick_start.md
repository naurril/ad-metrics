## 🚀 Quick Start

### Detection Evaluation

```python
from admetrics.detection import calculate_ap, calculate_iou_3d, calculate_nds

# 3D IoU
box1 = [0, 0, 0, 4, 2, 1.5, 0]  # [x, y, z, w, h, l, yaw]
box2 = [1, 0, 0, 4, 2, 1.5, 0]
iou = calculate_iou_3d(box1, box2)

# Average Precision
ap_result = calculate_ap(predictions, ground_truth, iou_threshold=0.7)
print(f"AP@0.7: {ap_result['ap']:.4f}")

# NuScenes Detection Score
nds = calculate_nds(predictions, ground_truth, class_names=['car', 'pedestrian'])
print(f"NDS: {nds:.4f}")
```

### Tracking Evaluation

```python
from admetrics.tracking import (
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_amota,
    calculate_tid_lgd,
    calculate_id_f1
)

# CLEAR MOT metrics
results = calculate_multi_frame_mota(predictions, ground_truth)
print(f"MOTA: {results['mota']:.4f}")
print(f"ID Switches: {results['num_switches']}")

# HOTA (balanced tracking)
hota_results = calculate_hota(predictions, ground_truth)
print(f"HOTA: {hota_results['hota']:.4f}")

# nuScenes tracking metrics
amota_results = calculate_amota(predictions, ground_truth, recall_thresholds=[0.2, 0.4, 0.6, 0.8])
print(f"AMOTA: {amota_results['amota']:.4f}")

# Track initialization and gap metrics
tid_lgd = calculate_tid_lgd(predictions, ground_truth)
print(f"TID: {tid_lgd['tid']:.2f}%, LGD: {tid_lgd['lgd']:.2f}%")

# Identity preservation
idf1_results = calculate_id_f1(predictions, ground_truth)
print(f"IDF1: {idf1_results['idf1']:.4f}")
```

### Trajectory Prediction

```python
from admetrics.prediction import calculate_ade, calculate_multimodal_ade

# Single-modal prediction
ade = calculate_ade(predicted_traj, ground_truth_traj)
fde = calculate_fde(predicted_traj, ground_truth_traj)

# Multi-modal prediction (K modes)
result = calculate_multimodal_ade(predicted_modes, ground_truth_traj)
print(f"minADE: {result['minADE']:.2f}m")
```

### Localization Evaluation

```python
from admetrics.localization import calculate_localization_metrics

# Comprehensive ego pose evaluation
# Supports 3D (x,y,z), 4D (x,y,z,yaw), or 7D (x,y,z,qw,qx,qy,qz)
metrics = calculate_localization_metrics(
    predicted_poses,      # (N, 4) or (N, 7)
    ground_truth_poses,
    timestamps=timestamps,
    lane_width=3.5,
    align=False  # Set True for SLAM drift analysis
)

print(f"ATE: {metrics['ate_mean']:.3f}m")
print(f"Lateral Error: {metrics['lateral_mean']:.3f}m")
print(f"Heading Error: {metrics['are_mean']:.2f}°")
```

### Occupancy Prediction

```python
from admetrics.occupancy import calculate_mean_iou, calculate_scene_completion

# Voxel-based metrics
miou = calculate_mean_iou(pred_occupancy, gt_occupancy, num_classes=18)
print(f"mIoU: {miou:.4f}")

# Scene completion
sc_result = calculate_scene_completion(pred_occupancy, gt_occupancy)
print(f"SC-IoU: {sc_result['sc_iou']:.4f}")
```

### Vector Map Detection

```python
from admetrics.vectormap import calculate_chamfer_distance_polyline, calculate_topology_metrics

# Geometric accuracy
cd = calculate_chamfer_distance_polyline(pred_polylines, gt_polylines)
print(f"Chamfer Distance: {cd['calculate_chamfer_distance']:.3f}m")

# Topology evaluation
topo = calculate_topology_metrics(pred_graph, gt_graph)
print(f"Successor Accuracy: {topo['successor_accuracy']:.4f}")
```

### Planning Evaluation

```python
from admetrics.planning import calculate_driving_score, calculate_collision_rate

# nuPlan-style comprehensive evaluation
score = calculate_driving_score(
    predicted_trajectory,
    expert_trajectory,
    obstacles,
    route
)
print(f"Driving Score: {score:.2f}/100")

# Safety metrics
collision = calculate_collision_rate(predicted_trajectory, obstacles)
print(f"Collision Rate: {collision:.2%}")
```