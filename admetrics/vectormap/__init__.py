"""
Vector Map Detection Metrics for HD Mapping.

This module provides metrics for evaluating road vector map detection/extraction,
including lane line detection, road boundary detection, and topology estimation.
Used in HD mapping, online mapping, and map-based localization tasks.
"""

from .vectormap import (
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
    'chamfer_distance_polyline',
    'frechet_distance',
    'polyline_iou',
    'lane_detection_metrics',
    'topology_metrics',
    'endpoint_error',
    'direction_accuracy',
    'vectormap_ap',
]
