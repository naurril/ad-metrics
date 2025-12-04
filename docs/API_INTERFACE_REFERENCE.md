# API Interface Reference

Complete reference of all 105 exported metric calculation interfaces in the `admetrics` library.

---

## Summary

| Category | Functions | Metric Outputs | Description |
|----------|-----------|----------------|-------------|
| **Detection** | 24 | 40+ metrics | IoU, AP, NDS, AOS, Confusion, Distance |
| **Tracking** | 21 | 50+ metrics | MOTA, MOTP, HOTA, IDF1, AMOTA, TID/LGD, MOTAL, OWTA |
| **Trajectory Prediction** | 10 | 10+ metrics | ADE, FDE, Miss Rate, NLL |
| **Localization** | 8 | 25+ metrics | ATE, RTE, ARE, Lateral/Longitudinal Error |
| **Occupancy** | 9 | 25+ metrics | IoU, mIoU, Chamfer, Scene Completion, Panoptic Quality, VPQ |
| **Planning** | 20 | 50+ metrics | L2, Collision, Progress, Comfort, Driving Score, Safety |
| **Vector Map** | 12 | 20+ metrics | Chamfer, Fréchet, Lane Detection, Topology, 3D Metrics, OLS, Per-Category |
| **Simulation Quality** | 11 | 40+ metrics | Camera, LiDAR, Radar, Noise, Alignment, Temporal, Sim2Real, Weather, Dynamics, Semantic, Occlusion |
| **Utility** | 9 | N/A | Transform, Matching, NMS, Visualization |
| **TOTAL** | **124** | **260+** | Comprehensive AD evaluation metrics |


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

## Tracking Metrics (21)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 25 | `calculate_mota` | Multiple Object Tracking Accuracy (single frame) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `dict`: {mota, tp, fp, fn, num_gt} | 5 |
| 26 | `calculate_motp` | Multiple Object Tracking Precision (single frame) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `dict`: {motp, num_matches, total_distance} | 3 |
| 27 | `calculate_clearmot_metrics` | CLEAR MOT metrics (MOTA + MOTP combined) | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `dict`: {mota, motp, tp, fp, fn} | 5 |
| 28 | `calculate_multi_frame_mota` | MOTA across multiple frames with ID tracking | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {mota, motp, num_matches, num_false_positives, num_misses, num_switches, num_fragmentations, mostly_tracked, partially_tracked, mostly_lost, total_gt, precision, recall} | 12 |
| 29 | `calculate_hota` | Higher Order Tracking Accuracy | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {hota, deta, assa, detection_re, detection_pr, association_re, association_pr, localization} | 8 |
| 30 | `calculate_id_f1` | ID F1 Score (identity preservation) | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {idf1, idp, idr, idtp, idfp, idfn} | 6 |
| 31 | `calculate_amota` | Average MOTA (nuScenes primary metric) | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`recall_thresholds`: List[float]<br>`iou_threshold`: float = 0.5 | `dict`: {amota, mota_at_recalls, recall_thresholds} | 1 + arrays |
| 32 | `calculate_motar` | MOTA at Recall threshold | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`recall_threshold`: float = 0.5<br>`iou_threshold`: float = 0.5 | `dict`: {motar, achieved_recall, threshold_used} | 3 |
| 33 | `calculate_false_alarm_rate` | False Alarms per Frame (FAF) and False Alarm Rate | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {faf, far, total_fp, total_frames, total_gt} | 5 |
| 34 | `calculate_track_metrics` | Track-level recall and precision | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {track_recall, track_precision, num_matched_gt_tracks, num_matched_pred_tracks, num_gt_tracks, num_pred_tracks} | 6 |
| 35 | `calculate_moda` | Multiple Object Detection Accuracy (MOTA without ID switches) | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {moda, tp, fp, fn, total_gt} | 5 |
| 36 | `calculate_hota_components` | Full HOTA decomposition with all sub-metrics | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {hota, det_a, det_re, det_pr, ass_a, ass_re, ass_pr, loc_a, tp, fp, fn} | 11 |
| 37 | `calculate_trajectory_metrics` | Trajectory-level metrics (MT/ML/PT) | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5<br>`mt_threshold`: float = 0.8<br>`ml_threshold`: float = 0.2 | `dict`: {mt_ratio, ml_ratio, pt_ratio, mt_count, ml_count, pt_count, total_tracks, avg_coverage, avg_track_length} | 9 |
| 38 | `calculate_detection_metrics` | Frame-level detection P/R/F1 (no tracking) | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {precision, recall, f1, tp, fp, fn} | 6 |
| 39 | `calculate_smota` | Soft MOTA for segmentation tracking | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5<br>`use_soft_matching`: bool = True | `dict`: {smota, soft_tp_error, num_matches, num_false_positives, num_switches} | 5 |
| 40 | `calculate_completeness` | GT coverage and detection density | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {gt_covered_ratio, avg_gt_coverage, frame_coverage, detection_density, num_gt_objects, num_detected_objects} | 6 |
| 41 | `calculate_identity_metrics` | Detailed identity preservation metrics | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {id_switches, id_switch_rate, avg_track_purity, avg_track_completeness, num_fragmentations, fragmentation_rate} | 6 |
| 42 | `calculate_tid_lgd` | Track Initialization Duration and Longest Gap Duration (nuScenes) | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {tid, lgd, avg_initialization_frames, avg_longest_gap, num_tracks, num_detected_tracks} | 6 |
| 43 | `calculate_motal` | MOTA with Logarithmic ID Switches | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {motal, mota, tp, fp, fn, id_switches, log_id_switches, num_gt} | 8 |
| 44 | `calculate_clr_metrics` | CLEAR MOT Recall, Precision, F1 | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {clr_re, clr_pr, clr_f1, tp, fp, fn, num_frames} | 7 |
| 45 | `calculate_owta` | Open World Tracking Accuracy | `frame_predictions`: Dict[int, List[Dict]]<br>`frame_ground_truth`: Dict[int, List[Dict]]<br>`iou_threshold`: float = 0.5 | `dict`: {owta, det_re, ass_a} | 3 |

**Benchmark Coverage:**
- **MOTChallenge**: MOTA, MOTP, HOTA, IDF1, MT/ML/PT, Fragmentations, ID Switches ✅
- **nuScenes**: AMOTA, AMOTP, MOTAR, FAF, TID, LGD ✅
- **KITTI**: MOTA, MOTP, MT/ML, ID Switches ✅
- **HOTA Framework**: Full HOTA decomposition (DetA, AssA, LocA) ✅
- **Open World Tracking**: OWTA ✅

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
| 41 | `calculate_ate` | Absolute Trajectory Error (ATE) | `pred_poses`: np.ndarray (N, 4, 4) or (N, 7)<br>`gt_poses`: np.ndarray (N, 4, 4) or (N, 7) | `dict`: {'ate_rmse', 'ate_mean', 'ate_std', 'ate_median', 'ate_min', 'ate_max'} | 6 |
| 42 | `calculate_relative_pose_error` | Relative Pose Error (RPE) | `pred_poses`: np.ndarray (N, 4, 4) or (N, 7)<br>`gt_poses`: np.ndarray (N, 4, 4) or (N, 7)<br>`delta`: int = 1 | `dict`: {'rpe_trans_rmse', 'rpe_rot_rmse', 'rpe_trans_mean', 'rpe_rot_mean'} | 4 |
| 43 | `calculate_translation_error` | Translation error statistics | `pred_poses`: np.ndarray<br>`gt_poses`: np.ndarray | `dict`: {'mean', 'std', 'rmse', 'median', 'min', 'max'} | 6 |
| 44 | `calculate_rotation_error` | Rotation error statistics | `pred_poses`: np.ndarray<br>`gt_poses`: np.ndarray<br>`metric`: str = "angle" | `dict`: {'mean', 'std', 'rmse', 'median', 'min', 'max'} | 6 |
| 45 | `calculate_lateral_error` | Lateral (cross-track) error | `pred_trajectory`: np.ndarray (N, 2/3)<br>`gt_trajectory`: np.ndarray (N, 2/3) | `dict`: {'mean', 'std', 'max', 'rmse'} | 4 |
| 46 | `calculate_longitudinal_error` | Longitudinal (along-track) error | `pred_trajectory`: np.ndarray (N, 2/3)<br>`gt_trajectory`: np.ndarray (N, 2/3) | `dict`: {'mean', 'std', 'max', 'rmse'} | 4 |
| 47 | `calculate_heading_error` | Heading/yaw angle error | `pred_headings`: np.ndarray (N,)<br>`gt_headings`: np.ndarray (N,) | `dict`: {'mean', 'std', 'rmse', 'median', 'min', 'max'} | 6 |
| 48 | `calculate_drift` | Cumulative position drift over time | `pred_poses`: np.ndarray<br>`gt_poses`: np.ndarray | `dict`: {'total_drift', 'drift_rate', 'drift_per_meter'} | 3 |

---

### Occupancy Metrics (9)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 49 | `calculate_occupancy_iou` | IoU for specific class or binary occupancy | `pred_occupancy`: np.ndarray (X, Y, Z)<br>`gt_occupancy`: np.ndarray (X, Y, Z)<br>`class_id`: int optional<br>`ignore_index`: int = 255 | `float`: IoU [0, 1] | 1 |
| 50 | `calculate_mean_iou` | Mean IoU across all semantic classes | `pred_occupancy`: np.ndarray (X, Y, Z)<br>`gt_occupancy`: np.ndarray (X, Y, Z)<br>`num_classes`: int<br>`ignore_index`: int = 255<br>`ignore_classes`: List[int] optional | `dict`: {'mIoU', 'class_iou': dict, 'valid_classes'} | 1 + per class |
| 51 | `calculate_occupancy_precision_recall` | Precision, recall, F1 for occupancy | `pred_occupancy`: np.ndarray (X, Y, Z)<br>`gt_occupancy`: np.ndarray (X, Y, Z)<br>`class_id`: int optional<br>`ignore_index`: int = 255 | `dict`: {'precision', 'recall', 'f1', 'true_positives', 'false_positives', 'false_negatives'} | 6 |
| 52 | `calculate_scene_completion` | Scene completion and SSC metrics | `pred_occupancy`: np.ndarray (X, Y, Z)<br>`gt_occupancy`: np.ndarray (X, Y, Z)<br>`free_class`: int = 0<br>`ignore_index`: int = 255 | `dict`: {'SC_IoU', 'SC_Precision', 'SC_Recall', 'SSC_mIoU', 'completion_ratio'} | 5 |
| 53 | `calculate_chamfer_distance` | Chamfer distance between point clouds | `pred_points`: np.ndarray (N, 3)<br>`gt_points`: np.ndarray (M, 3)<br>`bidirectional`: bool = True | `dict`: {'chamfer_distance', 'chamfer_pred_to_gt', 'chamfer_gt_to_pred'} | 3 |
| 54 | `calculate_surface_distance` | Surface distance metrics for boundaries | `pred_occupancy`: np.ndarray (X, Y, Z)<br>`gt_occupancy`: np.ndarray (X, Y, Z)<br>`voxel_size`: float = 1.0<br>`percentile`: int optional | `dict`: {'mean_surface_distance', 'median_surface_distance', 'std_surface_distance', 'max_surface_distance', 'percentile_distance'} | 5 |
| 55 | `calculate_visibility_weighted_iou` | Visibility-weighted IoU for sensor-based evaluation | `pred_occupancy`: np.ndarray (X, Y, Z)<br>`gt_occupancy`: np.ndarray (X, Y, Z)<br>`visibility_mask`: np.ndarray optional<br>`num_classes`: int = 16<br>`ignore_index`: int = 255 | `dict`: {'visibility_weighted_mIoU', 'class_iou': dict, 'visible_voxel_ratio'} | 2 + per class |
| 56 | `calculate_panoptic_quality` | Panoptic Quality for instance-aware occupancy | `pred_occupancy`: np.ndarray (X, Y, Z)<br>`pred_instances`: np.ndarray (X, Y, Z)<br>`gt_occupancy`: np.ndarray (X, Y, Z)<br>`gt_instances`: np.ndarray (X, Y, Z)<br>`num_classes`: int = 16<br>`ignore_index`: int = 255<br>`stuff_classes`: List[int] optional | `dict`: {'PQ', 'SQ', 'RQ', 'PQ_stuff', 'PQ_thing', 'per_class_pq', 'per_class_sq', 'per_class_rq'} | 5 + per class |
| 57 | `calculate_video_panoptic_quality` | Video Panoptic Quality for temporal sequences | `pred_occupancy_seq`: List[np.ndarray]<br>`pred_instances_seq`: List[np.ndarray]<br>`gt_occupancy_seq`: List[np.ndarray]<br>`gt_instances_seq`: List[np.ndarray]<br>`num_classes`: int = 16<br>`ignore_index`: int = 255<br>`stuff_classes`: List[int] optional | `dict`: {'VPQ', 'STQ', 'AQ', 'avg_PQ', 'avg_SQ', 'avg_RQ', 'per_frame_pq'} | 7 + per frame |

---

### Planning Metrics (20)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 58 | `calculate_l2_distance` | L2 distance between planned and expert trajectory | `pred_trajectory`: np.ndarray (T, 2/3)<br>`gt_trajectory`: np.ndarray (T, 2/3)<br>`weights`: np.ndarray optional | `float`: Weighted average L2 distance | 1 |
| 59 | `calculate_collision_rate` | Collision rate and statistics | `trajectory`: np.ndarray (T, 2)<br>`obstacles`: List[np.ndarray]<br>`vehicle_size`: Tuple = (4.5, 2.0)<br>`safety_margin`: float = 0.0 | `dict`: {'collision_rate', 'num_collisions', 'first_collision'} | 3 |
| 60 | **`calculate_collision_with_fault_classification`** | **NEW**: Collision with at-fault classification (nuPlan) | `ego_trajectory`: np.ndarray (T, 2)<br>`ego_velocities`: np.ndarray (T,)<br>`ego_headings`: np.ndarray (T,)<br>`other_vehicles`: List[np.ndarray]<br>`vehicle_size`: Tuple = (4.5, 2.0) | `dict`: {'total_collisions', 'at_fault_collisions', 'collision_rate', 'collision_types': Counter} | 5+ (types) |
| 61 | `calculate_progress_score` | Progress along planned route | `trajectory`: np.ndarray (T, 2)<br>`route`: np.ndarray (N, 2)<br>`distance_threshold`: float = 2.0 | `dict`: {'progress_score', 'distance_traveled', 'progress_ratio'} | 3 |
| 62 | `calculate_route_completion` | Route completion percentage | `trajectory`: np.ndarray (T, 2)<br>`route_waypoints`: List[np.ndarray]<br>`completion_radius`: float = 2.0 | `dict`: {'completion_rate', 'num_waypoints_reached', 'waypoint_status', 'total_waypoints'} | 3 |
| 63 | `average_displacement_error_planning` | ADE/FDE with multi-horizon support | `pred_trajectory`: np.ndarray (T, 2)<br>`expert_trajectory`: np.ndarray (T, 2)<br>`horizons`: List[int] optional | `dict`: {'ADE', 'FDE', 'ADE_H', 'FDE_H'} per horizon | 2+ (per horizon) |
| 64 | `calculate_lateral_deviation` | Lateral deviation from reference path | `trajectory`: np.ndarray (T, 2)<br>`reference_path`: np.ndarray (N, 2) | `dict`: {'mean_deviation', 'std_deviation', 'max_deviation'} | 3 |
| 65 | `calculate_heading_error` | Heading angle error with wrapping | `pred_headings`: np.ndarray (T,)<br>`reference_headings`: np.ndarray (T,) | `dict`: {'mean_error', 'std_error', 'max_error'} | 3 |
| 66 | `calculate_velocity_error` | Velocity error statistics | `pred_velocities`: np.ndarray (T,)<br>`reference_velocities`: np.ndarray (T,) | `dict`: {'mean_error', 'std_error', 'max_error', 'rmse'} | 4 |
| 67 | **`calculate_comfort_metrics`** | **UPDATED**: Comprehensive comfort with smoothing | `trajectory`: np.ndarray (T, 2)<br>`timestamps`: np.ndarray (T,)<br>`max_longitudinal_accel`: float = 4.0<br>`max_lateral_accel`: float = 4.0<br>`max_jerk`: float = 4.0<br>`max_yaw_rate`: float = 1.0<br>`max_yaw_accel`: float = 1.0<br>`include_lateral`: bool = True<br>`use_smoothing`: bool = False<br>`smoothing_window`: int = 15<br>`smoothing_order`: int = 2 | `dict`: {'mean_longitudinal_accel', 'max_longitudinal_accel', 'mean_lateral_accel', 'max_lateral_accel', 'mean_jerk', 'max_jerk', 'mean_yaw_rate', 'max_yaw_rate', 'mean_yaw_accel', 'max_yaw_accel', 'comfort_violations', 'comfort_rate'} | 12 |
| 68 | **`calculate_driving_score`** | **UPDATED**: Composite score with nuPlan mode | `pred_trajectory`: np.ndarray (T, 2)<br>`expert_trajectory`: np.ndarray (T, 2)<br>`timestamps`: np.ndarray (T,)<br>`obstacles`: List[np.ndarray] optional<br>`route`: np.ndarray optional<br>`weights`: Dict optional<br>`mode`: str = 'default' or 'nuplan' | `dict`: {'driving_score', 'planning_accuracy', 'safety_score', 'progress_score', 'comfort_score'} + nuPlan specifics | 5+ |
| 69 | `calculate_planning_kl_divergence` | KL divergence between trajectory distributions | `pred_distribution`: np.ndarray<br>`expert_distribution`: np.ndarray | `float`: KL divergence value | 1 |
| 70 | `calculate_time_to_collision` | Basic time-to-collision (TTC) | `ego_trajectory`: np.ndarray (T, 2)<br>`ego_velocity`: np.ndarray (T,)<br>`obstacles`: List[np.ndarray]<br>`vehicle_size`: Tuple = (4.5, 2.0) | `dict`: {'min_ttc', 'ttc_violations'} | 2 |
| 71 | **`calculate_time_to_collision_enhanced`** | **NEW**: Enhanced TTC with forward projection (nuPlan) | `ego_trajectory`: np.ndarray (T, 2)<br>`ego_velocities`: np.ndarray (T,)<br>`ego_headings`: np.ndarray (T,)<br>`timestamps`: np.ndarray (T,)<br>`other_vehicles`: List[np.ndarray]<br>`projection_horizon`: float = 1.0<br>`projection_dt`: float = 0.3<br>`ttc_threshold`: float = 3.0<br>`vehicle_size`: Tuple = (4.5, 2.0) | `dict`: {'min_ttc', 'mean_ttc', 'ttc_violations', 'ttc_profile'} | 4 |
| 72 | `calculate_lane_invasion_rate` | Lane invasion/departure rate | `trajectory`: np.ndarray (T, 2)<br>`lane_boundaries`: Tuple[np.ndarray, np.ndarray]<br>`vehicle_width`: float = 2.0 | `dict`: {'invasion_rate', 'num_invasions', 'max_invasion_distance'} | 3 |
| 73 | `calculate_collision_severity` | Collision severity based on impact | `trajectory`: np.ndarray (T, 2)<br>`obstacles`: List[np.ndarray]<br>`vehicle_size`: Tuple = (4.5, 2.0)<br>`velocities`: np.ndarray optional | `dict`: {'max_severity', 'severities': List} | 2 |
| 74 | `check_kinematic_feasibility` | Check trajectory kinematic feasibility (lateral accel & yaw rate) | `trajectory`: np.ndarray (T, 2)<br>`timestamps`: np.ndarray (T,)<br>`max_lateral_accel`: float = 3.0<br>`max_yaw_rate`: float = 1.0 | `dict`: {'feasible': bool, 'max_lateral_accel', 'max_yaw_rate'} | 3 |
| 75 | **`calculate_distance_to_road_edge`** | **NEW**: Signed distance to drivable area (Waymo) | `trajectory`: np.ndarray (T, 2)<br>`drivable_area`: Polygon optional<br>`lane_centerline`: np.ndarray optional<br>`lane_width`: float optional<br>`vehicle_width`: float = 2.0 | `dict`: {'mean_distance', 'min_distance', 'max_violation', 'violation_rate', 'distances'} | 5 |
| 76 | **`calculate_driving_direction_compliance`** | **NEW**: Wrong-way detection (nuPlan) | `trajectory`: np.ndarray (T, 2)<br>`headings`: np.ndarray (T,)<br>`lane_centerline`: np.ndarray (N, 2)<br>`angle_threshold`: float = π/2 | `dict`: {'compliance_score', 'wrong_way_distance', 'wrong_way_rate', 'heading_errors'} | 4 |
| 77 | **`calculate_interaction_metrics`** | **NEW**: Multi-agent proximity analysis (Waymo) | `ego_trajectory`: np.ndarray (T, 2)<br>`other_objects`: List[np.ndarray]<br>`vehicle_size`: Tuple = (4.5, 2.0)<br>`close_distance_threshold`: float = 5.0 | `dict`: {'min_distance', 'mean_distance_to_nearest', 'distance_to_nearest_per_timestep', 'closest_object_id', 'closest_approach_timestep', 'num_close_interactions'} | 6 |


### Vector Map Metrics (12)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 78 | `calculate_chamfer_distance_polyline` | Chamfer distance between two polylines | `pred_polyline`: np.ndarray (N, 2)<br>`gt_polyline`: np.ndarray (M, 2) | `dict`: {'chamfer', 'chamfer_pred_to_gt', 'chamfer_gt_to_pred', 'precision_50cm', 'recall_50cm'} | 5 |
| 79 | `calculate_frechet_distance` | Fréchet distance between polylines | `pred_polyline`: np.ndarray (N, 2)<br>`gt_polyline`: np.ndarray (M, 2) | `float`: Fréchet distance | 1 |
| 80 | `calculate_polyline_iou` | IoU between two polylines/polygons (approximated via sampling) | `pred_polyline`: np.ndarray (N, 2)<br>`gt_polyline`: np.ndarray (M, 2) | `float`: IoU [0, 1] | 1 |
| 81 | `calculate_lane_detection_metrics` | Accuracy of lane centerline extraction | `pred_lanes`: List[np.ndarray]<br>`gt_lanes`: List[np.ndarray]<br>`distance_threshold`: float = 0.5 | `dict`: {'precision', 'recall', 'f1', 'chamfer_distance', 'tp', 'fp', 'fn'} | 3 |
| 82 | `calculate_topology_accuracy` | Lane connectivity/topology accuracy | `pred_graph`: Dict<br>`gt_graph`: Dict | `dict`: {'edge_precision', 'edge_recall', 'node_precision', 'node_recall'} | 4 |
| 83 | `calculate_boundary_iou` | IoU for road/lane boundaries | `pred_boundaries`: List[np.ndarray]<br>`gt_boundaries`: List[np.ndarray]<br>`threshold`: float = 0.5 | `dict`: {'iou', 'precision', 'recall'} | 3 |
| 84 | `calculate_semantic_segmentation_iou` | Semantic IoU for map elements | `pred_map`: np.ndarray (H, W, C)<br>`gt_map`: np.ndarray (H, W, C)<br>`num_classes`: int | `dict`: {'miou', 'iou_per_class': dict} | 1 + per class |
| 85 | `calculate_rasterized_map_iou` | IoU for rasterized map representation | `pred_raster`: np.ndarray (H, W)<br>`gt_raster`: np.ndarray (H, W) | `float`: IoU [0, 1] | 1 |
| 86 | `calculate_chamfer_distance_3d` | 3D Chamfer distance with elevation awareness | `pred_polyline`: np.ndarray (N, 3)<br>`gt_polyline`: np.ndarray (M, 3)<br>`weight_z`: float = 1.0<br>`visibility_mask`: np.ndarray = None | `dict`: {'chamfer_distance_3d', 'chamfer_distance_xy', 'elevation_error', 'forward_distances', 'backward_distances'} | 5 |
| 87 | `calculate_frechet_distance_3d` | 3D Fréchet distance for curved lanes | `pred_polyline`: np.ndarray (N, 3)<br>`gt_polyline`: np.ndarray (M, 3)<br>`weight_z`: float = 1.0 | `dict`: {'frechet_distance_3d', 'frechet_distance_xy'} | 2 |
| 88 | `calculate_online_lane_segment_metric` | Temporal consistency for streaming evaluation | `detections_sequence`: List[List[Dict]]<br>`ground_truth_sequence`: List[List[Dict]]<br>`iou_threshold`: float = 0.5<br>`consistency_weight`: float = 0.5 | `dict`: {'ols', 'detection_score', 'consistency_score', 'avg_precision', 'avg_recall', 'id_switches'} | 6 |
| 89 | `calculate_per_category_metrics` | Per-category lane element evaluation | `predictions`: List[Dict] (with 'category')<br>`ground_truth`: List[Dict] (with 'category')<br>`iou_threshold`: float = 0.5 | `dict`: {'per_category': {category: {'precision', 'recall', 'f1', 'ap'}}, 'overall': {'macro_precision', 'macro_recall', 'macro_f1', 'macro_ap'}} | 4 per category + 4 overall |

---

### Simulation Quality Metrics (11)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 90 | `calculate_camera_image_quality` | Camera visual realism validation | `sim_images`: np.ndarray (N, H, W, C)<br>`real_images`: np.ndarray (N, H, W, C)<br>`metrics`: List[str] = ['psnr', 'color_distribution', 'brightness', 'contrast'] | `dict`: {'psnr', 'ssim', 'lpips', 'fid', 'color_distribution', 'brightness', 'contrast'} | 7 |
| 91 | `calculate_lidar_point_cloud_quality` | LiDAR geometric fidelity | `sim_points`: np.ndarray (N, 3/4)<br>`real_points`: np.ndarray (M, 3/4)<br>`max_range`: float = 100.0 | `dict`: {'chamfer_distance', 'point_density', 'range_distribution', 'intensity_correlation', 'vertical_distribution'} | 5 |
| 92 | `calculate_radar_quality` | Radar detection realism | `sim_detections`: np.ndarray (N, 5) [x,y,z,vel,rcs]<br>`real_detections`: np.ndarray (M, 5) | `dict`: {'detection_density_ratio', 'velocity_distribution', 'rcs_distribution', 'spatial_accuracy'} | 4 |
| 93 | `calculate_sensor_noise_characteristics` | Sensor noise pattern matching | `sim_measurements`: np.ndarray (N, D)<br>`real_measurements`: np.ndarray (M, D)<br>`ground_truth`: np.ndarray optional | `dict`: {'noise_std_ratio', 'noise_distribution_ks', 'bias_sim', 'bias_real', 'snr_ratio'} | 5 |
| 94 | `calculate_multimodal_sensor_alignment` | Multi-sensor fusion quality | `camera_detections`: np.ndarray (N, 7)<br>`lidar_detections`: np.ndarray (M, 7)<br>`camera_to_lidar_transform`: np.ndarray optional | `dict`: {'spatial_alignment_error', 'detection_agreement_rate', 'size_consistency', 'temporal_sync_error'} | 4 |
| 95 | `calculate_temporal_consistency` | Frame-to-frame coherence | `detections_sequence`: List[np.ndarray]<br>`fps`: float = 10.0 | `dict`: {'detection_count_variance', 'motion_smoothness', 'appearance_disappearance_rate', 'frame_to_frame_consistency'} | 4 |
| 96 | `calculate_perception_sim2real_gap` | Detection performance gap | `sim_detections`: List[Dict]<br>`real_detections`: List[Dict]<br>`metrics`: List[str] = ['recall', 'precision'] | `dict`: {'ap_gap', 'recall_gap', 'precision_gap', 'performance_drop'} | 4 |
| 97 | `calculate_weather_simulation_quality` | Weather/environment realism | `sim_data`: Dict[str, np.ndarray]<br>`real_data`: Dict[str, np.ndarray]<br>`weather_type`: str = 'rain'<br>`metrics`: List[str] optional | `dict`: {'intensity_distribution', 'visibility_range', 'temporal_consistency', 'spatial_distribution', 'particle_density', 'lighting_histogram', 'shadow_realism'} | 7+ |
| 98 | `calculate_vehicle_dynamics_quality` | Physics/dynamics realism | `sim_trajectories`: np.ndarray (N, T, D)<br>`real_trajectories`: np.ndarray (N, T, D)<br>`maneuver_type`: str = 'general'<br>`metrics`: List[str] optional | `dict`: {'acceleration_profile', 'braking_distance', 'lateral_dynamics', 'yaw_rate', 'trajectory_smoothness', 'speed_distribution', 'reaction_time'} | 7+ |
| 99 | `calculate_semantic_consistency` | Scene composition realism | `sim_scene_data`: Dict[str, np.ndarray]<br>`real_scene_data`: Dict[str, np.ndarray]<br>`scene_type`: str = 'mixed'<br>`metrics`: List[str] optional | `dict`: {'object_distribution_kl', 'vehicle_count_ratio', 'pedestrian_count_ratio', 'vehicle_speed_distribution', 'inter_vehicle_spacing', 'pedestrian_speed_distribution', 'traffic_density'} | 7+ |
| 100 | `calculate_occlusion_visibility_quality` | Occlusion pattern realism | `sim_detections`: Dict[str, np.ndarray]<br>`real_detections`: Dict[str, np.ndarray]<br>`metrics`: List[str] optional | `dict`: {'occlusion_kl_divergence', 'occlusion_mean_error', 'truncation_kl_divergence', 'visibility_correlation', 'range_visibility_correlation'} | 5+ |

---

### Utility Functions (9)

| # | Function | Description | Input | Output | Metric Count |
|---|----------|-------------|-------|--------|--------------|
| 101 | `calculate_iou_matrix` | Compute IoU matrix between two sets of boxes | `boxes1`: List[np.ndarray]<br>`boxes2`: List[np.ndarray]<br>`metric_type`: str = "3d" | `np.ndarray`: (N, M) IoU matrix | N/A |
| 102 | `greedy_matching` | Greedy matching based on IoU/distance | `predictions`: List[Dict]<br>`ground_truth`: List[Dict]<br>`iou_threshold`: float = 0.5 | `List[Tuple]`: Matched indices | N/A |
| 103 | `hungarian_matching` | Hungarian (optimal) matching | `cost_matrix`: np.ndarray | `Tuple`: (row_indices, col_indices) | N/A |
| 104 | `transform_box` | Transform a single box or apply translation/rotation/scale to box | `box`: np.ndarray<br>`translation`: np.ndarray = None<br>`rotation`: float = None<br>`scale`: float = None | `np.ndarray`: Transformed box | N/A |
| 105 | `convert_box_format` | Convert between box formats | `boxes`: np.ndarray<br>`input_format`: str<br>`output_format`: str | `np.ndarray`: Converted boxes | N/A |
| 106 | `filter_boxes_by_range` | Filter boxes by distance range | `boxes`: List[Dict]<br>`min_distance`: float = 0<br>`max_distance`: float = 100 | `List[Dict]`: Filtered boxes | N/A |
| 107 | `plot_boxes_bev` | Visualize boxes in bird's-eye view | `boxes`: List[np.ndarray]<br>`labels`: List[str] = None<br>`colors`: List[str] = None | `matplotlib.axes.Axes` | N/A |
| 108 | `plot_boxes_3d` | Plot 3D bounding boxes | `boxes`: List[np.ndarray]<br>`labels`: List[str] = None<br>`colors`: List[str] = None | `matplotlib.axes.Axes` | N/A |
| 109 | `nms_3d`, `nms_bev`, `nms_per_class` | Non-maximum suppression variants (3D, BEV, per-class) | `boxes`: List[np.ndarray]<br>`scores`: List[float]<br>`iou_threshold`: float = 0.5 | `List[int]`: Indices to keep | N/A |


