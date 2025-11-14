"""
Simulation Quality Metrics for Autonomous Driving.

This module provides metrics for evaluating the quality and realism of simulated
sensor data in autonomous vehicle simulators (CARLA, LGSVL, AirSim, etc.).
"""

from .sensor_quality import (
    camera_image_quality,
    lidar_point_cloud_quality,
    radar_quality,
    sensor_noise_characteristics,
    multimodal_sensor_alignment,
    temporal_consistency,
    perception_sim2real_gap,
)

__all__ = [
    'camera_image_quality',
    'lidar_point_cloud_quality',
    'radar_quality',
    'sensor_noise_characteristics',
    'multimodal_sensor_alignment',
    'temporal_consistency',
    'perception_sim2real_gap',
]
