# Metrics Reference Guide

A comprehensive reference of all metrics available in the `admetrics` library for autonomous driving evaluation: detection, tracking, trajectory prediction, and localization.

## Table of Contents

- [Metrics Table](#metrics-table)
- [Detection Metrics](#detection-metrics)
- [Tracking Metrics](#tracking-metrics)
- [Trajectory Prediction Metrics](#trajectory-prediction-metrics)
- [Localization Metrics](#localization-metrics)
- [Occupancy Metrics](#occupancy-metrics)
- [End-to-End Planning Metrics](#end-to-end-planning-metrics)
- [Vector Map Metrics](#vector-map-metrics)
- [Simulation Quality Metrics](#simulation-quality-metrics)
- [Utility Metrics](#utility-metrics)
- [Usage Patterns](#usage-patterns)
- [Metric Selection Guide](#metric-selection-guide)
- [Quick Reference Table](#quick-reference-table)
- [Additional Resources](#additional-resources)

---


## Metrics Table

| Category | Count | Key Metrics | Python Module | Documentation |
|----------|-------|-------------|---------------|---------------|
| **Detection** | 25 | IoU (3D/BEV/GIoU), AP, mAP, NDS, AOS, Precision/Recall/F1, Center Distance, Orientation Error | `admetrics.detection.*` | [DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Tracking** | 14 | MOTA, MOTP, HOTA, DetA, AssA, IDF1, ID Switches, Fragmentations | `admetrics.tracking` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **Trajectory Prediction** | 11 | ADE, FDE, minADE/minFDE, Brier-FDE, NLL, Miss Rate, Collision Rate | `admetrics.prediction` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **Localization** | 8 | ATE, RTE, ARE, Lateral/Longitudinal Error, Convergence, Map Alignment | `admetrics.localization` | [LOCALIZATION_METRICS.md](LOCALIZATION_METRICS.md) |
| **Occupancy** | 10 | IoU, mIoU, Precision/Recall/F1, Scene Completion, Chamfer Distance | `admetrics.occupancy` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **Planning** | 12 | L2 Distance, ADE/FDE, Collision Rate, Progress, Route Completion, Lateral Deviation, Driving Score | `admetrics.planning` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |
| **Vector Map** | 8 | Chamfer Distance, Fréchet Distance, Polyline IoU, Lane Detection, Topology, Endpoint Error, Direction Accuracy, AP | `admetrics.vectormap` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |
| **Simulation Quality** | 29 | Camera (PSNR, SSIM), LiDAR (Chamfer, Density), Radar (Velocity, RCS), Noise, Alignment, Temporal, Sim2Real Gap | `admetrics.simulation` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Utility** | 8+ | Hungarian/Greedy Matching, NMS (3D/BEV), Box Transforms, Visualization | `admetrics.utils.*` | [api_reference.md](docs/api_reference.md) |

**Total: 125+ metrics** across all categories for comprehensive autonomous driving evaluation.


---

## Detection Metrics

Metrics for evaluating single-frame 3D object detection performance.

### IoU Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **3D IoU** | 3D Intersection over Union - Volumetric overlap between 3D bounding boxes | Detection | `admetrics/iou.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **BEV IoU** | Bird's Eye View IoU - 2D overlap in top-down view (ignores height) | Detection | `admetrics/iou.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **GIoU** | Generalized IoU - IoU variant that penalizes non-overlapping boxes | Detection | `admetrics/iou.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Batch IoU** | Vectorized IoU calculation for multiple box pairs | Detection | `admetrics/iou.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_iou_3d()`, `calculate_iou_bev()`, `calculate_iou_batch()`, `calculate_giou_3d()`

**Range:** [0, 1] where 1 = perfect overlap, 0 = no overlap

**Use Cases:** Box matching, detection evaluation, NMS

---

### Average Precision Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **AP** | Average Precision - Area under precision-recall curve at IoU threshold | Detection | `admetrics/ap.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **mAP** | Mean Average Precision - Average AP across multiple classes | Detection | `admetrics/ap.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **AP@IoU** | AP at specific IoU threshold (e.g., AP@0.5, AP@0.7) | Detection | `admetrics/ap.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **COCO-style mAP** | mAP averaged over IoU thresholds [0.5:0.95:0.05] | Detection | `admetrics/ap.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_ap()`, `calculate_map()`, `calculate_coco_metrics()`

**Range:** [0, 1] where 1 = perfect detection

**Use Cases:** Model comparison, benchmark leaderboards, overall detection quality

---

### NuScenes Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **NDS** | NuScenes Detection Score - Composite metric combining mAP and error metrics | Detection | `admetrics/nds.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **ATE** | Average Translation Error - Mean center distance for true positives | Detection | `admetrics/nds.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **ASE** | Average Scale Error - Mean size/dimension error | Detection | `admetrics/nds.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **AOE** | Average Orientation Error - Mean heading angle error | Detection | `admetrics/nds.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **AVE** | Average Velocity Error - Mean velocity estimation error | Detection | `admetrics/nds.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **AAE** | Average Attribute Error - Mean attribute classification error | Detection | `admetrics/nds.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_nds()`, `calculate_tp_metrics()`

**Range:** [0, 1] for NDS where 1 = perfect; [0, ∞) for error metrics where 0 = perfect

**Use Cases:** nuScenes benchmark evaluation, comprehensive detection assessment

---

### KITTI Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **AOS** | Average Orientation Similarity - AP weighted by orientation accuracy | Detection | `admetrics/aos.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_aos()`, `calculate_orientation_similarity()`

**Range:** [0, 1] where 1 = perfect orientation + detection

**Use Cases:** KITTI benchmark, orientation-sensitive detection evaluation

---

### Confusion Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **TP** | True Positives - Correctly detected objects | Detection | `admetrics/confusion.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **FP** | False Positives - Incorrectly detected objects (hallucinations) | Detection | `admetrics/confusion.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **FN** | False Negatives - Missed ground truth objects | Detection | `admetrics/confusion.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Precision** | TP / (TP + FP) - Fraction of detections that are correct | Detection | `admetrics/confusion.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Recall** | TP / (TP + FN) - Fraction of ground truth objects detected | Detection | `admetrics/confusion.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean | Detection | `admetrics/confusion.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_tp_fp_fn()`, `calculate_confusion_metrics()`, `calculate_confusion_matrix()`

**Range:** [0, 1] for Precision, Recall, F1 where 1 = perfect; [0, ∞) for counts

**Use Cases:** Per-class analysis, error analysis, threshold tuning

---

### Distance Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Center Distance** | Euclidean distance between box centers (3D, BEV, or vertical) | Detection | `admetrics/distance.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Orientation Error** | Angular difference in heading/yaw angle | Detection | `admetrics/distance.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_center_distance()`, `calculate_orientation_error()`

**Range:** [0, ∞) meters for distance; [0, π] radians for orientation

**Use Cases:** Localization error analysis, orientation accuracy evaluation

---

## Tracking Metrics

Metrics for evaluating multi-frame object tracking with identity management.

### CLEAR MOT Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **MOTA** | Multiple Object Tracking Accuracy - Overall tracking quality penalizing FP, FN, ID switches | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **MOTP** | Multiple Object Tracking Precision - Average localization error for matched objects | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **ID Switches** | Number of times a track ID changes for the same ground truth object | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **Fragmentations** | Number of times a track is interrupted (lost then recovered) | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |

**Functions:** `calculate_mota()`, `calculate_motp()`, `calculate_multi_frame_mota()`

**Range:** MOTA: [-∞, 1] where 1 = perfect; MOTP: [0, ∞) meters where 0 = perfect

**Use Cases:** MOT benchmarks, tracking quality assessment, ID consistency evaluation

---

### HOTA Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **HOTA** | Higher Order Tracking Accuracy - Geometric mean of detection and association accuracy | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **DetA** | Detection Accuracy - How well objects are detected | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **AssA** | Association Accuracy - How well identities are maintained | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |

**Functions:** `calculate_hota()`

**Range:** [0, 1] where 1 = perfect

**Use Cases:** Balanced tracking evaluation, comparing trackers, understanding detection vs association trade-offs

---

### Identity Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **IDF1** | ID F1 Score - F1 score for identity preservation | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **IDP** | ID Precision - Precision of ID assignments | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **IDR** | ID Recall - Recall of ID assignments | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |

**Functions:** `calculate_id_f1()`

**Range:** [0, 1] where 1 = perfect identity consistency

**Use Cases:** ID switch analysis, re-identification quality, long-term tracking

---

### Trajectory-Level Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Mostly Tracked** | Number of trajectories tracked ≥80% of their lifetime | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **Partially Tracked** | Number of trajectories tracked 20-80% of their lifetime | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |
| **Mostly Lost** | Number of trajectories tracked <20% of their lifetime | Tracking | `admetrics/tracking.py` | [TRACKING_METRICS.md](TRACKING_METRICS.md) |

**Functions:** `calculate_multi_frame_mota()` (includes trajectory classification)

**Range:** [0, ∞) counts

**Use Cases:** Understanding tracking coverage, failure mode analysis

---

## Trajectory Prediction Metrics

Metrics for evaluating predicted future trajectories (motion forecasting).

### Core Displacement Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **ADE** | Average Displacement Error - Mean position error across all timesteps | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **FDE** | Final Displacement Error - Position error at final timestep | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **Miss Rate** | Percentage of predictions where FDE exceeds threshold (e.g., 2m) | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |

**Functions:** `calculate_ade()`, `calculate_fde()`, `calculate_miss_rate()`

**Range:** [0, ∞) meters for ADE/FDE; [0, 1] for Miss Rate

**Use Cases:** Single-modal prediction evaluation, average trajectory quality

---

### Multi-Modal Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **minADE** | Minimum ADE across K predicted modes - Best-case average error | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **minFDE** | Minimum FDE across K predicted modes - Best-case final error | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **meanADE** | Mean ADE across all predicted modes | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **meanFDE** | Mean FDE across all predicted modes | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |

**Functions:** `calculate_multimodal_ade()`, `calculate_multimodal_fde()`

**Range:** [0, ∞) meters

**Use Cases:** Multi-modal prediction benchmarks (Argoverse, nuScenes, Waymo)

---

### Probabilistic Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Brier-FDE** | Probability-weighted FDE - Evaluates confidence calibration | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **NLL** | Negative Log-Likelihood - Probabilistic prediction quality for Gaussian mixtures | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |

**Functions:** `calculate_brier_fde()`, `calculate_nll()`

**Range:** [0, ∞) for Brier-FDE; [-∞, ∞) for NLL (lower is better)

**Use Cases:** Uncertainty quantification, probabilistic model evaluation, calibration testing

---

### Safety & Constraint Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Collision Rate** | Percentage of timesteps where trajectory collides with obstacles | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |
| **Drivable Area Compliance** | Percentage of timesteps where trajectory stays within legal bounds | Trajectory | `admetrics/trajectory.py` | [TRAJECTORY_PREDICTION.md](TRAJECTORY_PREDICTION.md) |

**Functions:** `calculate_collision_rate()`, `calculate_drivable_area_compliance()`

**Range:** [0, 1] where 1 = 100% safe/compliant

**Use Cases:** Safety validation, deployment readiness, realistic trajectory generation

---

## Localization Metrics

Metrics for evaluating ego vehicle pose estimation and localization systems in autonomous driving.

### Position Accuracy Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **ATE** | Absolute Trajectory Error - Mean Euclidean distance between predicted and ground truth positions | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **RTE** | Relative Trajectory Error - Drift over specific distances (e.g., 100m, 200m) | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Lateral Error** | Cross-track error - Perpendicular distance from ground truth path | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Longitudinal Error** | Along-track error - Distance ahead/behind on path | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_ate()`, `calculate_rte()`, `calculate_lateral_error()`, `calculate_longitudinal_error()`

**Range:** [0, ∞) meters where 0 = perfect localization

**Use Cases:** GPS evaluation, SLAM accuracy, sensor fusion validation

---

### Orientation Accuracy Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **ARE** | Absolute Rotation Error - Mean heading angle error (supports yaw or quaternion) | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_are()`

**Range:** [0, 180°] where 0 = perfect orientation

**Use Cases:** Heading estimation, IMU calibration, orientation accuracy assessment

---

### Convergence & Map Alignment Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Convergence Rate** | Speed of localization initialization/recovery (time to reach accuracy threshold) | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |
| **Map Alignment Score** | Distance to HD map lane centerlines and alignment rate | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_convergence_rate()`, `calculate_map_alignment_score()`

**Range:** Time in seconds for convergence; [0, ∞) meters for map distance

**Use Cases:** Initialization assessment, HD map-based localization, lane-level accuracy

---

### Comprehensive Localization Evaluation

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Localization Metrics** | All-in-one evaluation: ATE, RTE, lateral, longitudinal, ARE, convergence | Localization | `admetrics/localization.py` | [docs/DETECTION_METRICS.md](docs/DETECTION_METRICS.md) |

**Functions:** `calculate_localization_metrics()`

**Features:**
- Supports 3D poses (x, y, z), 4D poses (x, y, z, yaw), and 7D poses (x, y, z, qw, qx, qy, qz)
- Optional trajectory alignment using Umeyama algorithm (for SLAM evaluation)
- Lane violation detection (cross-track error > lane_width/2)
- Convergence analysis with timestamps

**Use Cases:** 
- RTK-GPS vs consumer GPS comparison
- SLAM drift analysis
- Sensor fusion evaluation (GPS+IMU+LiDAR)
- HD map-based localization assessment
- Modular autonomous driving pipeline evaluation

---

## Occupancy Metrics

Metrics for evaluating 3D semantic occupancy prediction in voxel grids.

### Voxel-wise Classification Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Occupancy IoU** | Intersection over Union for specific class or binary occupancy | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **mIoU** | Mean IoU across all semantic classes in occupancy grid | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **Precision** | TP / (TP + FP) for voxel-level predictions | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **Recall** | TP / (TP + FN) for voxel-level predictions | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **F1-Score** | Harmonic mean of precision and recall | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |

**Functions:** `occupancy_iou()`, `mean_iou()`, `occupancy_precision_recall()`

**Range:** [0, 1] where 1 = perfect prediction

**Use Cases:** Semantic scene understanding, voxel-based scene completion, occupancy prediction benchmarks

---

### Scene Completion Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **SC-IoU** | Scene Completion IoU - Binary IoU for occupied vs free space | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **SSC-mIoU** | Semantic Scene Completion mIoU - Mean IoU over occupied semantic classes | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **Completion Ratio** | Ratio of predicted occupied voxels to ground truth occupied voxels | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |

**Functions:** `scene_completion()`

**Range:** [0, 1] for IoU metrics; [0, ∞) for Completion Ratio (1.0 = perfect)

**Use Cases:** 3D scene completion, geometry reconstruction, free space estimation

---

### Geometric Quality Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Chamfer Distance** | Average distance between predicted and GT point clouds (bidirectional) | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |
| **Surface Distance** | Distance between predicted and GT surface voxels | Occupancy | `admetrics/occupancy.py` | [OCCUPANCY_METRICS.md](OCCUPANCY_METRICS.md) |

**Functions:** `chamfer_distance()`, `surface_distance()`

**Range:** [0, ∞) meters/voxels, lower is better

**Use Cases:** Shape accuracy evaluation, boundary quality assessment, reconstruction fidelity

---

## End-to-End Planning Metrics

Metrics for evaluating end-to-end autonomous driving models that directly output planned trajectories or driving actions.

### Planning Accuracy Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **L2 Distance** | Euclidean distance between predicted and expert trajectories | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |
| **ADE (Planning)** | Average Displacement Error across trajectory timesteps | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |
| **FDE (Planning)** | Final Displacement Error at end of trajectory | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |

**Functions:** `l2_distance()`, `average_displacement_error_planning()`

**Range:** [0, ∞) meters where 0 = perfect trajectory match

**Use Cases:** nuPlan benchmark, trajectory matching, imitation learning evaluation

---

### Safety Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Collision Rate** | Percentage of timesteps with collision (static/dynamic obstacles) | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |

**Functions:** `collision_rate()`

**Range:** [0, 1] where 0 = no collisions (ideal)

**Use Cases:** Safety validation, simulation evaluation, deployment readiness

---

### Progress & Navigation Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Progress Score** | Distance traveled along reference path and completion ratio | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |
| **Route Completion** | Waypoint-based navigation success rate | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |

**Functions:** `progress_score()`, `route_completion()`

**Range:** [0, 1] for completion rates; [0, ∞) meters for progress

**Use Cases:** Task completion, goal-reaching evaluation, CARLA benchmark

---

### Control Accuracy Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Lateral Deviation** | Cross-track error from reference path (lane keeping) | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |
| **Heading Error** | Orientation/yaw angle difference from expert | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |
| **Velocity Error** | Speed control accuracy (mean, RMSE, max) | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |

**Functions:** `lateral_deviation()`, `heading_error()`, `velocity_error()`

**Range:** [0, ∞) meters/radians where 0 = perfect tracking

**Use Cases:** Path following, speed control, control system validation

---

### Comfort Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Comfort Metrics** | Acceleration, jerk, comfort violations, comfort rate | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |

**Functions:** `comfort_metrics()`

**Range:** [0, ∞) for acceleration/jerk; [0, 1] for comfort rate

**Use Cases:** Passenger comfort evaluation, smooth driving assessment

---

### Composite Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Driving Score** | Comprehensive score combining planning, safety, progress, comfort | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |

**Functions:** `driving_score()`

**Range:** [0, 100] where 100 = perfect driving

**Use Cases:** nuPlan/CARLA-style benchmarks, overall performance assessment

---

### Imitation Learning Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Planning KL Divergence** | KL divergence between learned and expert action distributions | Planning | `admetrics/planning.py` | [END_TO_END_METRICS.md](END_TO_END_METRICS.md) |

**Functions:** `planning_kl_divergence()`

**Range:** [0, ∞) where 0 = perfect match

**Use Cases:** Behavioral cloning, policy distillation, expert matching

---

## Vector Map Metrics

Metrics for evaluating HD map vector detection, reconstruction, and topology quality.

### Geometric Accuracy Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Chamfer Distance (Polyline)** | Bidirectional nearest-neighbor distance between predicted and GT polylines | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |
| **Fréchet Distance** | Curve similarity metric considering point ordering and continuity | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |
| **Polyline IoU** | Area-based overlap using buffered lane widths | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |

**Functions:** `chamfer_distance_polyline()`, `frechet_distance()`, `polyline_iou()`

**Range:** [0, ∞) meters for distances (lower better); [0, 1] for IoU (higher better)

**Use Cases:** Lane centerline accuracy, curve reconstruction quality, geometric fidelity

---

### Detection & Matching Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Lane Detection Metrics** | Precision, Recall, F1 for lane detection with distance threshold matching | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |
| **Vector Map AP** | Average Precision at multiple distance thresholds (0.5m, 1.0m, 2.0m) | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |

**Functions:** `lane_detection_metrics()`, `vector_map_ap()`

**Range:** [0, 1] where 1 = perfect detection

**Use Cases:** Lane detection benchmarks, matching quality, threshold sensitivity analysis

---

### Topology Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Topology Metrics** | Successor and neighbor connectivity accuracy for lane graph structure | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |

**Functions:** `topology_metrics()`

**Range:** [0, 1] for accuracy (higher better)

**Use Cases:** Lane graph evaluation, routing feasibility, connectivity validation

---

### Endpoint & Direction Metrics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Endpoint Error** | Distance error for lane merge/split start and end points | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |
| **Direction Accuracy** | Tangent vector alignment at lane points (angle error) | Vector Map | `admetrics/vectormap.py` | [VECTORMAP_METRICS.md](VECTORMAP_METRICS.md) |

**Functions:** `endpoint_error()`, `direction_accuracy()`

**Range:** [0, ∞) meters for endpoint; [0, 180°] for direction (lower better)

**Use Cases:** Merge/split detection, lane heading estimation, curvature accuracy

---

## Simulation Quality Metrics

Metrics for evaluating the fidelity and realism of simulated sensor data in autonomous vehicle simulators.

### Camera Image Quality

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **PSNR** | Peak Signal-to-Noise Ratio - Image similarity measure | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **SSIM** | Structural Similarity Index - Perceptual image quality | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Color Distribution** | KL divergence of RGB histograms | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Brightness/Contrast** | Mean brightness and contrast ratio comparison | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |

**Functions:** `camera_image_quality()`

**Range:** PSNR [0, ∞) dB (higher better); Others [0, ∞) (lower better for KL)

**Use Cases:** Visual realism validation, camera simulation tuning, domain adaptation

---

### LiDAR Point Cloud Quality

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Chamfer Distance** | Bidirectional nearest-neighbor distance between point clouds | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Point Density** | Ratio of point counts between sim and real | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Range Distribution** | KL divergence of distance-from-sensor distributions | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Vertical Angle Distribution** | Beam pattern matching (elevation angles) | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Intensity Correlation** | Correlation of intensity values | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |

**Functions:** `lidar_point_cloud_quality()`

**Range:** [0, ∞) meters for Chamfer (lower better); [0, 1] for correlation (higher better)

**Use Cases:** Geometric fidelity validation, LiDAR simulator tuning, occlusion modeling

---

### Radar Quality

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Detection Density** | Ratio of detection counts between sim and real | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Velocity Distribution** | KL divergence of radial velocity histograms | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **RCS Distribution** | Radar Cross-Section distribution similarity | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |

**Functions:** `radar_quality()`

**Range:** [0, ∞) for KL divergence (lower better)

**Use Cases:** Radar simulation validation, clutter modeling, velocity estimation

---

### Sensor Noise Characteristics

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Noise Std Ratio** | Ratio of noise standard deviations | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **KS Test** | Kolmogorov-Smirnov distribution similarity test | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **SNR** | Signal-to-Noise Ratio comparison | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Bias** | Mean error from ground truth | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |

**Functions:** `sensor_noise_characteristics()`

**Range:** Ratio ~1.0 ideal; p-value > 0.05 for KS test

**Use Cases:** Noise modeling validation, robustness testing, filter tuning

---

### Multimodal Sensor Alignment

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Detection Agreement** | Fraction of detections visible in both sensors | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Spatial Alignment** | Mean position error for matched detections | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Size Consistency** | Bounding box size difference | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Orientation Error** | Heading angle consistency | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |

**Functions:** `multimodal_sensor_alignment()`

**Range:** [0, 1] for agreement (higher better); [0, ∞) for errors (lower better)

**Use Cases:** Calibration validation, sensor fusion testing, extrinsic parameter tuning

---

### Temporal Consistency

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Detection Count Variance** | Stability of detection counts over time | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Frame-to-Frame Consistency** | Fraction of detections persisting between frames | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Flicker Rate** | Rate of appearance/disappearance events | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |

**Functions:** `temporal_consistency()`

**Range:** [0, 1] for consistency (higher better); [0, 1] for flicker rate (lower better)

**Use Cases:** Tracking validation, temporal stability, occlusion modeling

---

### Sim-to-Real Gap

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Precision Gap** | Difference in precision between sim and real | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Recall Gap** | Difference in recall between sim and real | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **F1 Gap** | Difference in F1-score | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |
| **Performance Drop** | Percentage degradation from sim to real | Simulation | `admetrics/simulation.py` | [SIMULATION_QUALITY.md](SIMULATION_QUALITY.md) |

**Functions:** `perception_sim2real_gap()`

**Range:** [-1, 1] for gaps; [0, 100] for performance drop %

**Use Cases:** Domain shift quantification, deployment readiness, domain adaptation

---

## Utility Metrics

Helper functions and transformations.

### Matching Algorithms

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Hungarian Matching** | Optimal bipartite matching using Hungarian algorithm | Utility | `admetrics/utils/matching.py` | [docs/api_reference.md](docs/api_reference.md) |
| **Greedy Matching** | Fast greedy matching by highest IoU | Utility | `admetrics/utils/matching.py` | [docs/api_reference.md](docs/api_reference.md) |

**Functions:** `hungarian_matching()`, `greedy_matching()`, `match_detections()`

**Use Cases:** Box assignment for evaluation, tracking data association

---

### Non-Maximum Suppression

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **NMS 3D** | Remove duplicate 3D detections based on IoU overlap | Utility | `admetrics/utils/nms.py` | [docs/api_reference.md](docs/api_reference.md) |
| **NMS BEV** | Remove duplicate detections in bird's eye view | Utility | `admetrics/utils/nms.py` | [docs/api_reference.md](docs/api_reference.md) |

**Functions:** `nms_3d()`, `nms_bev()`, `nms_per_class()`

**Use Cases:** Post-processing detections, removing duplicates

---

### Coordinate Transforms

| Metric | Meaning | Category | Python File | Documentation |
|--------|---------|----------|-------------|---------------|
| **Box Translation** | Translate box center by offset | Utility | `admetrics/utils/transforms.py` | [docs/api_reference.md](docs/api_reference.md) |
| **Box Rotation** | Rotate box around center | Utility | `admetrics/utils/transforms.py` | [docs/api_reference.md](docs/api_reference.md) |
| **Format Conversion** | Convert between box representations | Utility | `admetrics/utils/transforms.py` | [docs/api_reference.md](docs/api_reference.md) |

**Functions:** `translate_box()`, `rotate_box()`, `convert_box_format()`

**Use Cases:** Coordinate system conversion, data augmentation

---

## Usage Patterns

### Basic Detection Evaluation
```python
from admetrics import calculate_iou_3d, calculate_ap

# IoU for matching
iou = calculate_iou_3d(pred_box, gt_box)

# AP for overall quality
ap_result = calculate_ap(predictions, ground_truth, iou_threshold=0.7)
```

### Tracking Evaluation
```python
from admetrics.tracking import calculate_multi_frame_mota, calculate_hota

# Multi-frame tracking
mota = calculate_multi_frame_mota(predictions, ground_truth)

# Balanced tracking metric
hota = calculate_hota(predictions, ground_truth)
```

### Trajectory Prediction Evaluation
```python
from admetrics.trajectory import calculate_ade, calculate_multimodal_ade

# Single-modal
ade = calculate_ade(predicted_traj, ground_truth_traj)

# Multi-modal (K modes)
result = calculate_multimodal_ade(predicted_modes, ground_truth_traj)
```

### Localization Evaluation
```python
from admetrics.localization import calculate_localization_metrics, calculate_map_alignment_score

# Comprehensive ego pose evaluation
metrics = calculate_localization_metrics(
    predicted_poses,   # (N, 4) for [x, y, z, yaw] or (N, 7) for quaternions
    ground_truth_poses,
    timestamps=timestamps,
    lane_width=3.5,
    align=False  # Set True for SLAM drift analysis
)

# HD map alignment
map_score = calculate_map_alignment_score(
    predicted_poses[:, :2],  # 2D positions
    hd_map_lanes  # List of lane centerlines
)
```

---

## Metric Selection Guide

### For Detection Tasks
- **Quick check:** Precision, Recall, F1
- **Standard benchmark:** AP, mAP
- **NuScenes:** NDS (composite)
- **KITTI:** AOS (orientation-aware)
- **Error analysis:** ATE, ASE, AOE

### For Tracking Tasks
- **Standard benchmark:** MOTA, MOTP
- **Modern approach:** HOTA (balanced)
- **ID consistency:** IDF1
- **Detailed analysis:** ID Switches, Fragmentations

### For Trajectory Prediction
- **Single-modal:** ADE, FDE
- **Multi-modal:** minADE, minFDE
- **Probabilistic:** Brier-FDE, NLL
- **Safety:** Collision Rate, Drivable Area Compliance

### For Localization/Odometry
- **Absolute accuracy:** ATE, ARE
- **Drift analysis:** RTE (at 100m, 200m)
- **Path following:** Lateral/Longitudinal Errors
- **Initialization:** Convergence Rate
- **HD maps:** Map Alignment Score

### For Occupancy Prediction
- **Standard benchmark:** mIoU
- **Scene completion:** SC-IoU, SSC-mIoU
- **Per-class analysis:** Class-specific IoU, Precision, Recall
- **Geometric quality:** Chamfer Distance, Surface Distance
- **Completeness:** Completion Ratio

### For End-to-End Planning
- **Planning accuracy:** L2 Distance, ADE, FDE
- **Safety (critical):** Collision Rate
- **Task completion:** Progress Score, Route Completion
- **Control quality:** Lateral Deviation, Heading Error, Velocity Error
- **Comfort:** Acceleration, Jerk, Comfort Rate
- **Comprehensive:** Driving Score (nuPlan/CARLA style)

### For Simulation Quality
- **Camera:** PSNR, SSIM, Color Distribution
- **LiDAR:** Chamfer Distance, Point Density, Range Distribution
- **Radar:** Detection Density, Velocity/RCS Distribution
- **Noise:** Std Ratio, KS Test, SNR
- **Calibration:** Multimodal Alignment, Spatial Error
- **Temporal:** Frame Consistency, Flicker Rate
- **Deployment:** Sim2Real Gap, Performance Drop

### For Vector Map Detection
- **Geometric accuracy:** Chamfer Distance, Fréchet Distance
- **Detection quality:** Lane Detection Metrics, Vector Map AP
- **Topology/Routing:** Topology Metrics (successors, neighbors)
- **Shape matching:** Polyline IoU
- **Critical points:** Endpoint Error (merges/splits)
- **Orientation:** Direction Accuracy (tangent vectors)

---

## Quick Reference Table

### File Structure Reference

```
admetrics/
├── __init__.py           # Main exports
├── detection/
│   ├── iou.py           # IoU metrics (3D, BEV, GIoU)
│   ├── ap.py            # AP, mAP, COCO-style metrics
│   ├── nds.py           # NuScenes Detection Score
│   ├── aos.py           # Average Orientation Similarity (KITTI)
│   ├── confusion.py     # TP/FP/FN, Precision, Recall, F1
│   └── distance.py      # Center distance, orientation error
├── tracking/
│   └── tracking.py      # MOTA, MOTP, HOTA, IDF1
├── prediction/
│   └── trajectory.py    # ADE, FDE, NLL, safety metrics
├── localization/
│   └── localization.py  # ATE, RTE, ARE, lateral/longitudinal errors
├── occupancy/
│   └── occupancy.py     # mIoU, scene completion, Chamfer distance
├── planning/
│   └── planning.py      # L2 distance, collision rate, driving score
├── vectormap/
│   └── vectormap.py     # HD map detection and topology
├── simulation/
│   └── sensor_quality.py # Camera/LiDAR/radar quality, noise, sim2real gap
└── utils/
    ├── matching.py      # Hungarian, greedy matching
    ├── nms.py          # Non-maximum suppression
    ├── transforms.py   # Coordinate transformations
    └── visualization.py # Plotting and visualization
```

### Documentation Map

| Topic | Documentation File | Description |
|-------|-------------------|-------------|
| **General API** | `docs/api_reference.md` | Complete API documentation |
| **Detection Metrics** | `docs/DETECTION_METRICS.md` | IoU, AP, NDS, AOS details |
| **Tracking Metrics** | `TRACKING_METRICS.md` | MOTA, HOTA, IDF1 guide |
| **Trajectory Metrics** | `TRAJECTORY_PREDICTION.md` | ADE, FDE, probabilistic metrics |
| **Localization Metrics** | `LOCALIZATION_METRICS.md` | Ego pose and odometry metrics |
| **Occupancy Metrics** | `OCCUPANCY_METRICS.md` | Voxel-based scene understanding |
| **End-to-End Metrics** | `END_TO_END_METRICS.md` | Planning and driving evaluation |
| **Vector Map Metrics** | `VECTORMAP_METRICS.md` | HD map detection and topology |
| **Simulation Quality** | `SIMULATION_QUALITY.md` | Sensor fidelity and sim-to-real gap |
| **Dataset Formats** | `docs/dataset_formats.md` | KITTI, nuScenes, Waymo formats |
| **Examples** | `examples/` directory | Working code examples |

### Test Coverage

| Module | Test File | Test Cases | Coverage |
|--------|-----------|------------|----------|
| `detection/iou.py` | `tests/test_iou.py` | 14 | 91% |
| `detection/ap.py` | `tests/test_ap.py` | 9 | 98% |
| `detection/nds.py` | `tests/test_nds.py` | 5 | 96% |
| `detection/aos.py` | `tests/test_aos.py` | 18 | 96% |
| `detection/distance.py` | `tests/test_distance.py` | 31 | 94% |
| `detection/confusion.py` | `tests/test_confusion.py` | 5 | 90% |
| `tracking/tracking.py` | `tests/test_tracking.py` | 16 | 89% |
| `prediction/trajectory.py` | `tests/test_trajectory.py` | 26 | 95% |
| `localization/localization.py` | `tests/test_localization.py` | 27 | 91% |
| `occupancy/occupancy.py` | `tests/test_occupancy.py` | 39 | 98% |
| `planning/planning.py` | `tests/test_planning.py` | 42 | 95% |
| `vectormap/vectormap.py` | `tests/test_vectormap.py` | 45 | 98% |
| `simulation/sensor_quality.py` | `tests/test_simulation.py` | 17 | 66% |
| `utils/*` | `tests/test_utils.py` | 12 | 50% |
| **Total** | | **304** | **80%** |

### Benchmark Compatibility

| Benchmark | Supported Metrics | Example File |
|-----------|------------------|--------------|
| **KITTI** | AP, AOS, 3D IoU | `examples/kitti_evaluation.py` |
| **nuScenes Detection** | NDS, mAP, TP errors | `examples/nuscenes_evaluation.py` |
| **nuScenes Tracking** | AMOTA, AMOTP | `examples/tracking_evaluation.py` |
| **nuScenes Occupancy** | mIoU, SC-IoU | `examples/occupancy_evaluation.py` |
| **nuScenes Map** | Chamfer Distance, AP | `examples/vectormap_evaluation.py` |
| **Argoverse 2 Map** | Chamfer Distance, Fréchet Distance | `examples/vectormap_evaluation.py` |
| **OpenLane-V2** | Topology Metrics, Lane F1 | `examples/vectormap_evaluation.py` |
| **nuPlan** | Driving Score, L2 Distance | `examples/end_to_end_evaluation.py` |
| **CARLA** | Route Completion, Collision Rate | `examples/end_to_end_evaluation.py` |
| **CARLA/LGSVL/AirSim** | Camera/LiDAR/Radar Quality | `examples/simulation_quality_evaluation.py` |
| **Argoverse** | minADE₆, minFDE₆, Miss Rate | `examples/trajectory_prediction.py` |
| **Waymo Open** | AP, APH, mAP | `examples/basic_usage.py` |
| **KITTI Odometry** | ATE, RTE (drift) | `examples/localization_evaluation.py` |
| **SemanticKITTI** | Occupancy mIoU | `examples/occupancy_evaluation.py` |
| **Occ3D** | mIoU, Ray-based metrics | `examples/occupancy_evaluation.py` |

---
## Additional Resources

- **GitHub Repository:** [Link to repository]
- **Examples:** See `examples/` directory
- **Test Suite:** See `tests/` directory
- **Issue Tracker:** [Link to issues]
- **Contributing Guide:** `CONTRIBUTING.md`

---

*Last Updated: November 14, 2025*
