# API Interface Reference

Complete reference of all 80 exported metric calculation interfaces in the `admetrics` library.

---

## Summary

| Category | Functions | Metric Outputs | Description |
|----------|-----------|----------------|-------------|
| **Detection** | 24 | 40+ metrics | IoU, AP, NDS, AOS, Confusion, Distance |
| **Tracking** | 6 | 15+ metrics | MOTA, MOTP, HOTA, IDF1 |
| **Trajectory Prediction** | 10 | 10+ metrics | ADE, FDE, Miss Rate, NLL |
| **Localization** | 8 | 25+ metrics | ATE, RTE, ARE, Lateral/Longitudinal Error |
| **Occupancy** | 6 | 15+ metrics | IoU, mIoU, Chamfer, Scene Completion |
| **Planning** | 11 | 20+ metrics | L2, Collision, Progress, Comfort, Driving Score |
| **Vector Map** | 8 | 15+ metrics | Chamfer, Fréchet, Lane Detection, Topology |
| **Simulation Quality** | 7 | 25+ metrics | Camera, LiDAR, Radar, Sensor Alignment |
| **Utility** | 9 | N/A | Transform, Matching, NMS |
| **TOTAL** | **89** | **165+** | Comprehensive AD evaluation metrics |


**Note:** Many functions return dictionaries with multiple metrics (e.g., mean, std, min, max), so the total number of individual metric values is much higher than the function count.


---

## Detection Metrics (24)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 1 | `calculate_iou_3d` | Calculate 3D IoU between two 3D bounding boxes | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,rotation]<br>`box_format`: str = "xyzwhlr" | `float`: IoU [0, 1] | 1 |
| 2 | `calculate_iou_bev` | Calculate Bird's Eye View IoU (2D overlap) | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,rotation]<br>`box_format`: str = "xyzwhlr" | `float`: BEV IoU [0, 1] | 1 |
| 3 | `calculate_iou_batch` | Vectorized IoU calculation for multiple box pairs | `boxes1`: np.ndarray (N, 7)<br>`boxes2`: np.ndarray (M, 7)<br>`metric_type`: "3d" or "bev" | `np.ndarray`: IoU matrix (N, M) | 1 |
| 4 | `calculate_giou_3d` | Calculate Generalized IoU (penalizes non-overlap) | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,rotation]<br>`box_format`: str = "xyzwhlr" | `float`: GIoU [-1, 1] | 1 |
| 5 | `calculate_ap` | Calculate Average Precision for 3D object detection | `predictions`: List[Dict] (box, score, class)<br>`ground_truth`: List[Dict] (box, class)<br>`iou_threshold`: float = 0.5 | `dict`: {ap, precision, recall, scores} | 1 (AP) + arrays |
| 6 | `calculate_map` | Calculate Mean Average Precision across classes | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`class_names`: List[str]<br>`iou_threshold`: float = 0.5 | `dict`: {map, class_ap} | 1 (mAP) + per-class |
| 7 | `calculate_ap_coco_style` | Calculate COCO-style mAP (averaged over IoU thresholds) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_thresholds`: List[float] | `dict`: {map, map_50, map_75} | 3 (mAP@IoU) |
| 8 | `calculate_precision_recall_curve` | Calculate Precision-Recall curve data | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `dict`: {precision, recall, scores, tp, fp} | 5 arrays |
| 9 | `calculate_nds` | Calculate NuScenes Detection Score | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`class_weights`: Dict (optional) | `dict`: {nds, map, tp_errors} | 1 (NDS) + components |
| 10 | `calculate_tp_metrics` | Calculate TP error metrics (ATE, ASE, AOE, AVE, AAE) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`matches`: List[Tuple] | `dict`: {ate, ase, aoe, ave, aae} | 5 (TP errors) |
| 11 | `calculate_nds_detailed` | Calculate detailed NDS with per-class breakdown | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`class_names`: List[str] | `dict`: {nds, map, tp_errors, class_scores} | 1 (NDS) + detailed |
| 12 | `calculate_aos` | Calculate Average Orientation Similarity (KITTI) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.7 | `dict`: {aos, ap, orientation_similarity} | 3 (AOS, AP, OS) |
| 13 | `calculate_aos_per_difficulty` | Calculate AOS per difficulty level (easy/moderate/hard) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict] (with difficulty)<br>`iou_threshold`: float = 0.7 | `dict`: {easy, moderate, hard} | 3 (per difficulty) |
| 14 | `calculate_orientation_similarity` | Calculate orientation similarity between two boxes | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,yaw] | `float`: Orientation similarity [0, 1] | 1 |
| 15 | `calculate_tp_fp_fn` | Calculate True Positives, False Positives, False Negatives | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5<br>`metric_type`: "3d" or "bev" | `dict`: {tp, fp, fn} | 3 (TP, FP, FN) |
| 16 | `calculate_confusion_metrics` | Calculate Precision, Recall, F1-Score, Accuracy | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `dict`: {precision, recall, f1, accuracy} | 4 |
| 17 | `calculate_confusion_matrix_multiclass` | Calculate multi-class confusion matrix | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`class_names`: List[str]<br>`iou_threshold`: float = 0.5 | `dict`: {confusion_matrix, class_accuracy} | N×N matrix + per-class |
| 18 | `calculate_specificity` | Calculate specificity (true negative rate) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `float`: Specificity [0, 1] | 1 |
| 19 | `calculate_center_distance` | Calculate distance between box centers | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,yaw]<br>`distance_type`: "euclidean", "bev", "vertical" | `float`: Distance in meters | 1 |
| 20 | `calculate_orientation_error` | Calculate orientation/heading error | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,yaw]<br>`output_unit`: "radians" or "degrees" | `float`: Orientation error | 1 |
| 21 | `calculate_size_error` | Calculate size/dimension error | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,yaw]<br>`error_type`: "absolute", "relative", "percentage" | `dict`: {width_error, height_error, length_error, volume_error} | 4 (size errors) |
| 22 | `calculate_velocity_error` | Calculate velocity error | `box1`, `box2`: np.ndarray [x,y,z,w,h,l,yaw,vx,vy,vz]<br>`error_type`: "euclidean", "manhattan", "angular" | `dict`: {error, relative_error, angle_error} | 3 (velocity errors) |
| 23 | `calculate_average_distance_error` | Calculate average distance error across all matches | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`matches`: List[Tuple]<br>`distance_type`: str = "euclidean" | `dict`: {mean, std, min, max} | 4 statistics |
| 24 | `calculate_translation_error_bins` | Calculate translation error binned by distance | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`matches`: List[Tuple]<br>`distance_bins`: List[Tuple] | `dict`: {f'{start}-{end}m': float} | N (per bin) |

---

### Tracking Metrics (6)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 25 | `calculate_mota` | Multiple Object Tracking Accuracy | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `float`: MOTA score | 1 |
| 26 | `calculate_motp` | Multiple Object Tracking Precision | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `float`: MOTP score | 1 |
| 27 | `calculate_clear_metrics` | CLEAR MOT metrics (MOTA, MOTP, MT, ML, IDS, etc.) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `dict`: {'mota', 'motp', 'mt', 'ml', 'ids', 'frag', 'recall', 'precision', 'far', 'num_frames'} | 10 |
| 28 | `calculate_hota` | Higher Order Tracking Accuracy | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `dict`: {'hota', 'deta', 'assa', 'localization', 'detection'} | 5 |
| 29 | `calculate_identity_switches` | Count identity switches (ID switches) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `int`: Number of ID switches | 1 |
| 30 | `calculate_fragmentations` | Count trajectory fragmentations | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `int`: Number of fragmentations | 1 |

---

### Trajectory Prediction Metrics (10)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 31 | `calculate_ade` | Average Displacement Error | `predictions`: np.ndarray (T, 2/3)<br>`ground_truth`: np.ndarray (T, 2/3) | `float`: ADE in meters | 1 |
| 32 | `calculate_fde` | Final Displacement Error | `predictions`: np.ndarray (T, 2/3)<br>`ground_truth`: np.ndarray (T, 2/3) | `float`: FDE in meters | 1 |
| 33 | `calculate_min_ade` | Minimum ADE over K predictions | `predictions`: List[np.ndarray] K×(T, 2/3)<br>`ground_truth`: np.ndarray (T, 2/3) | `float`: min ADE | 1 |
| 34 | `calculate_min_fde` | Minimum FDE over K predictions | `predictions`: List[np.ndarray] K×(T, 2/3)<br>`ground_truth`: np.ndarray (T, 2/3) | `float`: min FDE | 1 |
| 35 | `calculate_miss_rate` | Miss rate at distance threshold | `predictions`: List[np.ndarray]<br>`ground_truth`: np.ndarray<br>`distance_threshold`: float = 2.0 | `float`: Miss rate [0, 1] | 1 |
| 36 | `calculate_nll` | Negative Log-Likelihood for probabilistic predictions | `predictions`: np.ndarray<br>`ground_truth`: np.ndarray<br>`covariance`: np.ndarray | `float`: NLL score | 1 |
| 37 | `calculate_brier_fde` | Brier-FDE for probabilistic multi-modal predictions | `predictions`: List[np.ndarray] K×(T, 2/3)<br>`probabilities`: np.ndarray (K,)<br>`ground_truth`: np.ndarray (T, 2/3) | `dict`: {'brier_fde', 'brier_minFDE', 'ECE'} | 3 |
| 38 | `calculate_collision_rate` | Predicted collision rate with map/agents | `trajectories`: List[np.ndarray]<br>`map_data`: Dict<br>`agent_boxes`: List[np.ndarray] | `dict`: {'collision_rate', 'off_road_rate', 'agent_collision_rate', 'map_collision_rate'} | 4 |
| 39 | `calculate_diversity` | Diversity of multi-modal predictions | `predictions`: List[np.ndarray] K×(T, 2/3) | `float`: Average pairwise distance | 1 |
| 40 | `calculate_scene_compliance` | Scene compliance score (comfort, progress, etc.) | `trajectory`: np.ndarray (T, 2/3)<br>`map_data`: Dict<br>`speed_limit`: float | `dict`: {'comfort', 'progress', 'speed_compliance'} | 3 |

---

### Localization Metrics (8)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 41 | `calculate_absolute_trajectory_error` | Absolute Trajectory Error (ATE) | `pred_poses`: np.ndarray (N, 4, 4) or (N, 7)<br>`gt_poses`: np.ndarray (N, 4, 4) or (N, 7) | `dict`: {'ate_rmse', 'ate_mean', 'ate_std', 'ate_median', 'ate_min', 'ate_max'} | 6 |
| 42 | `calculate_relative_pose_error` | Relative Pose Error (RPE) | `pred_poses`: np.ndarray (N, 4, 4) or (N, 7)<br>`gt_poses`: np.ndarray (N, 4, 4) or (N, 7)<br>`delta`: int = 1 | `dict`: {'rpe_trans_rmse', 'rpe_rot_rmse', 'rpe_trans_mean', 'rpe_rot_mean'} | 4 |
| 43 | `calculate_translation_error` | Translation error statistics | `pred_poses`: np.ndarray<br>`gt_poses`: np.ndarray | `dict`: {'mean', 'std', 'rmse', 'median', 'min', 'max'} | 6 |
| 44 | `calculate_rotation_error` | Rotation error statistics | `pred_poses`: np.ndarray<br>`gt_poses`: np.ndarray<br>`metric`: str = "angle" | `dict`: {'mean', 'std', 'rmse', 'median', 'min', 'max'} | 6 |
| 45 | `calculate_lateral_error` | Lateral (cross-track) error | `pred_trajectory`: np.ndarray (N, 2/3)<br>`gt_trajectory`: np.ndarray (N, 2/3) | `dict`: {'mean', 'std', 'max', 'rmse'} | 4 |
| 46 | `calculate_longitudinal_error` | Longitudinal (along-track) error | `pred_trajectory`: np.ndarray (N, 2/3)<br>`gt_trajectory`: np.ndarray (N, 2/3) | `dict`: {'mean', 'std', 'max', 'rmse'} | 4 |
| 47 | `calculate_heading_error` | Heading/yaw angle error | `pred_headings`: np.ndarray (N,)<br>`gt_headings`: np.ndarray (N,) | `dict`: {'mean', 'std', 'rmse', 'median', 'min', 'max'} | 6 |
| 48 | `calculate_drift` | Cumulative position drift over time | `pred_poses`: np.ndarray<br>`gt_poses`: np.ndarray | `dict`: {'total_drift', 'drift_rate', 'drift_per_meter'} | 3 |

---

### Occupancy Metrics (6)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 49 | `calculate_mean_iou` | Mean Intersection over Union for occupancy grids | `pred_occupancy`: np.ndarray (H, W, D)<br>`gt_occupancy`: np.ndarray (H, W, D)<br>`num_classes`: int = 3 | `dict`: {'miou', 'iou_per_class': dict} | 1 + per class |
| 50 | `calculate_occupancy_recall` | Recall for occupied cells | `pred_occupancy`: np.ndarray<br>`gt_occupancy`: np.ndarray<br>`class_id`: int = 1 | `float`: Recall [0, 1] | 1 |
| 51 | `calculate_occupancy_precision` | Precision for occupied cells | `pred_occupancy`: np.ndarray<br>`gt_occupancy`: np.ndarray<br>`class_id`: int = 1 | `float`: Precision [0, 1] | 1 |
| 52 | `calculate_voxel_accuracy` | Per-voxel classification accuracy | `pred_occupancy`: np.ndarray<br>`gt_occupancy`: np.ndarray | `float`: Accuracy [0, 1] | 1 |
| 53 | `calculate_occupancy_f1` | F1-score for occupancy classification | `pred_occupancy`: np.ndarray<br>`gt_occupancy`: np.ndarray<br>`class_id`: int = 1 | `float`: F1 score [0, 1] | 1 |
| 54 | `calculate_occupancy_flow_error` | Flow error for dynamic occupancy | `pred_flow`: np.ndarray (H, W, D, 2/3)<br>`gt_flow`: np.ndarray (H, W, D, 2/3)<br>`mask`: np.ndarray | `dict`: {'mean_epe', 'outlier_rate'} | 2 |

---

### Planning Metrics (11)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 55 | `calculate_l2_distance` | L2 distance between planned and expert trajectory | `pred_trajectory`: np.ndarray (T, 2/3)<br>`gt_trajectory`: np.ndarray (T, 2/3) | `dict`: {'mean', 'max', 'final'} | 3 |
| 56 | `calculate_collision_time` | Time to first collision | `trajectory`: np.ndarray (T, 2/3)<br>`obstacles`: List[np.ndarray]<br>`ego_size`: Tuple = (4.5, 2.0) | `float`: Time in seconds (inf if no collision) | 1 |
| 57 | `calculate_comfort_metrics` | Jerk, acceleration, steering smoothness | `trajectory`: np.ndarray (T, 2/3)<br>`dt`: float = 0.1 | `dict`: {'jerk', 'accel_max', 'steering_rate'} | 3 |
| 58 | `calculate_progress_towards_goal` | Progress made towards goal | `trajectory`: np.ndarray (T, 2/3)<br>`goal_position`: np.ndarray (2/3,)<br>`start_position`: np.ndarray (2/3,) | `float`: Progress [0, 1] | 1 |
| 59 | `calculate_imitation_loss` | L1/L2 loss for imitation learning | `pred_trajectory`: np.ndarray<br>`expert_trajectory`: np.ndarray<br>`loss_type`: str = "l2" | `float`: Imitation loss | 1 |
| 60 | `calculate_rule_violation_rate` | Traffic rule violation rate | `trajectory`: np.ndarray<br>`map_data`: Dict<br>`traffic_rules`: List[str] | `dict`: {'red_light', 'speed_limit', 'lane_invasion', 'total_violations'} | 4 |
| 61 | `calculate_time_to_collision_min` | Minimum time-to-collision with any obstacle | `trajectory`: np.ndarray (T, 2/3)<br>`obstacles`: List[np.ndarray]<br>`ego_size`: Tuple | `float`: Min TTC in seconds | 1 |
| 62 | `calculate_lane_center_offset` | Distance from lane center over trajectory | `trajectory`: np.ndarray (T, 2/3)<br>`lane_centerline`: np.ndarray (M, 2/3) | `dict`: {'mean', 'std', 'max'} | 3 |
| 63 | `calculate_curvature` | Trajectory curvature statistics | `trajectory`: np.ndarray (T, 2/3) | `dict`: {'mean', 'std', 'max'} | 3 |
| 64 | `calculate_smoothness` | Trajectory smoothness (via jerk) | `trajectory`: np.ndarray (T, 2/3)<br>`dt`: float = 0.1 | `float`: Smoothness score | 1 |
| 65 | `calculate_goal_reaching_rate` | Rate of reaching goal within threshold | `trajectories`: List[np.ndarray]<br>`goals`: List[np.ndarray]<br>`threshold`: float = 2.0 | `float`: Success rate [0, 1] | 1 |

---

### Vector Map Metrics (8)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 66 | `calculate_chamfer_distance_polyline` | Chamfer distance between two polylines | `pred_polyline`: np.ndarray (N, 2)<br>`gt_polyline`: np.ndarray (M, 2) | `dict`: {'chamfer', 'chamfer_pred_to_gt', 'chamfer_gt_to_pred', 'precision_50cm', 'recall_50cm'} | 5 |
| 67 | `calculate_frechet_distance` | Fréchet distance between polylines | `pred_polyline`: np.ndarray (N, 2)<br>`gt_polyline`: np.ndarray (M, 2) | `float`: Fréchet distance | 1 |
| 68 | `calculate_iou_polygon` | IoU between two polygons | `pred_polygon`: np.ndarray (N, 2)<br>`gt_polygon`: np.ndarray (M, 2) | `float`: IoU [0, 1] | 1 |
| 69 | `calculate_centerline_accuracy` | Accuracy of lane centerline extraction | `pred_centerlines`: List[np.ndarray]<br>`gt_centerlines`: List[np.ndarray]<br>`distance_threshold`: float = 0.5 | `dict`: {'precision', 'recall', 'f1'} | 3 |
| 70 | `calculate_topology_accuracy` | Lane connectivity/topology accuracy | `pred_graph`: Dict<br>`gt_graph`: Dict | `dict`: {'edge_precision', 'edge_recall', 'node_precision', 'node_recall'} | 4 |
| 71 | `calculate_boundary_iou` | IoU for road/lane boundaries | `pred_boundaries`: List[np.ndarray]<br>`gt_boundaries`: List[np.ndarray]<br>`threshold`: float = 0.5 | `dict`: {'iou', 'precision', 'recall'} | 3 |
| 72 | `calculate_semantic_segmentation_iou` | Semantic IoU for map elements | `pred_map`: np.ndarray (H, W, C)<br>`gt_map`: np.ndarray (H, W, C)<br>`num_classes`: int | `dict`: {'miou', 'iou_per_class': dict} | 1 + per class |
| 73 | `calculate_rasterized_map_iou` | IoU for rasterized map representation | `pred_raster`: np.ndarray (H, W)<br>`gt_raster`: np.ndarray (H, W) | `float`: IoU [0, 1] | 1 |

---

### Simulation Quality Metrics (7)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 74 | `calculate_sensor_noise_metrics` | Sensor noise characterization | `sensor_data`: np.ndarray<br>`ground_truth`: np.ndarray<br>`sensor_type`: str | `dict`: {'mean_error', 'std_error', 'bias', 'snr', 'outlier_rate'} | 5 |
| 75 | `calculate_lidar_accuracy` | LiDAR simulation accuracy | `sim_points`: np.ndarray (N, 3/4)<br>`real_points`: np.ndarray (M, 3/4)<br>`voxel_size`: float = 0.1 | `dict`: {'chamfer_distance', 'coverage', 'density_error'} | 3 |
| 76 | `calculate_camera_realism` | Camera image realism metrics | `sim_image`: np.ndarray (H, W, 3)<br>`real_image`: np.ndarray (H, W, 3) | `dict`: {'ssim', 'psnr', 'lpips', 'fid'} | 4 |
| 77 | `calculate_motion_realism` | Motion/dynamics realism | `sim_trajectory`: np.ndarray (T, 7)<br>`real_trajectory`: np.ndarray (T, 7) | `dict`: {'velocity_error', 'acceleration_error', 'jerk_error'} | 3 |
| 78 | `calculate_scenario_diversity` | Diversity of simulated scenarios | `scenarios`: List[Dict]<br>`feature_extractor`: Callable | `dict`: {'diversity_score', 'coverage', 'uniqueness'} | 3 |
| 79 | `calculate_sim2real_gap` | Sim-to-real performance gap | `sim_metrics`: Dict<br>`real_metrics`: Dict | `dict`: {metric_name: gap for each metric} | N (per metric) |
| 80 | `calculate_temporal_consistency` | Temporal consistency across frames | `sim_sequence`: List[np.ndarray]<br>`real_sequence`: List[np.ndarray] | `dict`: {'optical_flow_error', 'frame_consistency'} | 2 |

---

### Utility Functions (9)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 81 | `calculate_iou_matrix` | Compute IoU matrix between two sets of boxes | `boxes1`: List[np.ndarray]<br>`boxes2`: List[np.ndarray]<br>`metric_type`: str = "3d" | `np.ndarray`: (N, M) IoU matrix | N/A |
| 82 | `greedy_matching` | Greedy matching based on IoU/distance | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `List[Tuple]`: Matched indices | N/A |
| 83 | `hungarian_matching` | Hungarian (optimal) matching | `cost_matrix`: np.ndarray | `Tuple`: (row_indices, col_indices) | N/A |
| 84 | `transform_boxes` | Transform boxes to different coordinate frame | `boxes`: np.ndarray<br>`transform`: np.ndarray (4, 4) | `np.ndarray`: Transformed boxes | N/A |
| 85 | `convert_box_format` | Convert between box formats | `boxes`: np.ndarray<br>`input_format`: str<br>`output_format`: str | `np.ndarray`: Converted boxes | N/A |
| 86 | `filter_boxes_by_range` | Filter boxes by distance range | `boxes`: List[Dict]<br>`min_distance`: float = 0<br>`max_distance`: float = 100 | `List[Dict]`: Filtered boxes | N/A |
| 87 | `visualize_boxes_bev` | Visualize boxes in bird's-eye view | `boxes`: List[np.ndarray]<br>`image_size`: Tuple = (800, 800)<br>`colors`: List = None | `np.ndarray`: BEV image | N/A |
| 88 | `plot_trajectory` | Plot 2D/3D trajectory | `trajectory`: np.ndarray (T, 2/3)<br>`gt_trajectory`: np.ndarray = None<br>`title`: str = "" | `matplotlib.figure.Figure` | N/A |
| 89 | `nms` | Non-maximum suppression for boxes | `boxes`: List[np.ndarray]<br>`scores`: List[float]<br>`iou_threshold`: float = 0.5 | `List[int]`: Indices to keep | N/A |


