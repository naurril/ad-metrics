"""
Metrics for Autonomous Driving

This package provides comprehensive metrics for evaluating autonomous driving systems,
including 3D object detection, multi-object tracking, trajectory prediction, 
ego vehicle localization, 3D occupancy prediction, end-to-end planning,
simulation quality assessment, and HD map vector detection.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Detection metrics
from admetrics.detection import (
    calculate_iou_3d,
    calculate_iou_bev,
    calculate_iou_batch,
    calculate_ap,
    calculate_map,
    calculate_nds,
    calculate_aos,
    calculate_confusion_metrics,
    calculate_tp_fp_fn,
    calculate_center_distance,
    calculate_orientation_error,
)

# Tracking metrics
from admetrics.tracking import (
    calculate_mota,
    calculate_motp,
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_id_f1,
)

# Trajectory prediction metrics
from admetrics.prediction import (
    calculate_ade,
    calculate_fde,
    calculate_miss_rate,
    calculate_multimodal_ade,
    calculate_multimodal_fde,
    calculate_brier_fde,
    calculate_nll,
    calculate_collision_rate,
    calculate_drivable_area_compliance,
    calculate_trajectory_metrics,
)

# Localization metrics
from admetrics.localization import (
    calculate_ate,
    calculate_rte,
    calculate_are,
    calculate_lateral_error,
    calculate_longitudinal_error,
    calculate_convergence_rate,
    calculate_localization_metrics,
    calculate_map_alignment_score,
)

# Occupancy metrics
from admetrics.occupancy import (
    occupancy_iou,
    mean_iou,
    occupancy_precision_recall,
    scene_completion,
    chamfer_distance,
    surface_distance,
)

# End-to-end planning metrics
from admetrics.planning import (
    l2_distance,
    collision_rate,
    progress_score,
    route_completion,
    average_displacement_error_planning,
    lateral_deviation,
    heading_error,
    velocity_error,
    comfort_metrics,
    driving_score,
    planning_kl_divergence,
)

# Simulation quality metrics
from admetrics.simulation import (
    camera_image_quality,
    lidar_point_cloud_quality,
    radar_quality,
    sensor_noise_characteristics,
    multimodal_sensor_alignment,
    temporal_consistency,
    perception_sim2real_gap,
)

# Vector map detection metrics
from admetrics.vectormap import (
    chamfer_distance_polyline,
    frechet_distance,
    polyline_iou,
    lane_detection_metrics,
    topology_metrics,
    endpoint_error,
    direction_accuracy,
    vectormap_ap,
)

__all__ = [
    # Detection metrics
    "calculate_iou_3d",
    "calculate_iou_bev",
    "calculate_iou_batch",
    "calculate_ap",
    "calculate_map",
    "calculate_nds",
    "calculate_aos",
    "calculate_confusion_metrics",
    "calculate_tp_fp_fn",
    "calculate_center_distance",
    "calculate_orientation_error",
    # Tracking metrics
    "calculate_mota",
    "calculate_motp",
    "calculate_multi_frame_mota",
    "calculate_hota",
    "calculate_id_f1",
    # Trajectory prediction metrics
    "calculate_ade",
    "calculate_fde",
    "calculate_miss_rate",
    "calculate_multimodal_ade",
    "calculate_multimodal_fde",
    "calculate_brier_fde",
    "calculate_nll",
    "calculate_collision_rate",
    "calculate_drivable_area_compliance",
    "calculate_trajectory_metrics",
    # Localization metrics
    "calculate_ate",
    "calculate_rte",
    "calculate_are",
    "calculate_lateral_error",
    "calculate_longitudinal_error",
    "calculate_convergence_rate",
    "calculate_localization_metrics",
    "calculate_map_alignment_score",
    # Occupancy metrics
    "occupancy_iou",
    "mean_iou",
    "occupancy_precision_recall",
    "scene_completion",
    "chamfer_distance",
    "surface_distance",
    # End-to-end planning metrics
    "l2_distance",
    "collision_rate",
    "progress_score",
    "route_completion",
    "average_displacement_error_planning",
    "lateral_deviation",
    "heading_error",
    "velocity_error",
    "comfort_metrics",
    "driving_score",
    "planning_kl_divergence",
    # Simulation quality metrics
    "camera_image_quality",
    "lidar_point_cloud_quality",
    "radar_quality",
    "sensor_noise_characteristics",
    "multimodal_sensor_alignment",
    "temporal_consistency",
    "perception_sim2real_gap",
    # Vector map detection metrics
    "chamfer_distance_polyline",
    "frechet_distance",
    "polyline_iou",
    "lane_detection_metrics",
    "topology_metrics",
    "endpoint_error",
    "direction_accuracy",
    "vectormap_ap",
]
