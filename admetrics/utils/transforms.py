"""
Coordinate transformation utilities for 3D bounding boxes.
"""

import numpy as np
from typing import Union, List, Tuple


def transform_box(
    box: np.ndarray,
    translation: np.ndarray = None,
    rotation: float = None,
    scale: float = None
) -> np.ndarray:
    """
    Apply transformation to a 3D bounding box.
    
    Args:
        box: 3D box [x, y, z, w, h, l, yaw]
        translation: Translation vector [tx, ty, tz]
        rotation: Rotation angle in radians (around z-axis)
        scale: Scaling factor
    
    Returns:
        Transformed box
    """
    box = box.copy()
    
    # Apply translation
    if translation is not None:
        box[:3] += translation
    
    # Apply rotation
    if rotation is not None:
        # Rotate center
        x, y, z = box[:3]
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        box[0] = cos_r * x - sin_r * y
        box[1] = sin_r * x + cos_r * y
        
        # Update yaw
        box[6] += rotation
    
    # Apply scale
    if scale is not None:
        box[:3] *= scale  # Scale center
        box[3:6] *= scale  # Scale dimensions
    
    return box


def rotate_box(
    box: np.ndarray,
    rotation: float,
    origin: np.ndarray = None
) -> np.ndarray:
    """
    Rotate a 3D bounding box around a point.
    
    Args:
        box: 3D box [x, y, z, w, h, l, yaw]
        rotation: Rotation angle in radians (around z-axis)
        origin: Point to rotate around [x, y, z]. Default is [0, 0, 0]
    
    Returns:
        Rotated box
    """
    if origin is None:
        origin = np.array([0, 0, 0])
    
    box = box.copy()
    
    # Translate to origin
    box[:3] -= origin
    
    # Rotate around z-axis
    x, y, z = box[:3]
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    
    box[0] = cos_r * x - sin_r * y
    box[1] = sin_r * x + cos_r * y
    
    # Update yaw
    box[6] += rotation
    
    # Translate back
    box[:3] += origin
    
    return box


def translate_box(
    box: np.ndarray,
    translation: np.ndarray
) -> np.ndarray:
    """
    Translate a 3D bounding box.
    
    Args:
        box: 3D box [x, y, z, w, h, l, yaw]
        translation: Translation vector [tx, ty, tz]
    
    Returns:
        Translated box
    """
    box = box.copy()
    box[:3] += translation
    return box


def convert_box_format(
    box: np.ndarray,
    src_format: str,
    dst_format: str
) -> np.ndarray:
    """
    Convert between different 3D bounding box formats.
    
    Args:
        box: Input box
        src_format: Source format ('xyzwhlr', 'xyzhwlr', 'corners')
        dst_format: Destination format
    
    Returns:
        Box in destination format
        
    Formats:
        - 'xyzwhlr': [x, y, z, width, height, length, rotation]
        - 'xyzhwlr': [x, y, z, height, width, length, rotation]
        - 'corners': 8 corner points (8, 3)
    """
    if src_format == dst_format:
        return box.copy()
    
    # Convert to xyzwhlr as intermediate format
    if src_format == "xyzhwlr":
        # Swap width and height
        intermediate = box.copy()
        intermediate[3], intermediate[4] = box[4], box[3]
    elif src_format == "corners":
        # Convert corners to center format
        intermediate = corners_to_center(box)
    else:
        intermediate = box.copy()
    
    # Convert from xyzwhlr to destination format
    if dst_format == "xyzwhlr":
        return intermediate
    elif dst_format == "xyzhwlr":
        result = intermediate.copy()
        result[3], result[4] = intermediate[4], intermediate[3]
        return result
    elif dst_format == "corners":
        return center_to_corners(intermediate)
    else:
        raise ValueError(f"Unknown format: {dst_format}")


def center_to_corners(box: np.ndarray) -> np.ndarray:
    """
    Convert center-based box to 8 corner points.
    
    Args:
        box: [x, y, z, w, h, l, yaw]
    
    Returns:
        (8, 3) array of corner coordinates
    """
    x, y, z, w, h, l, yaw = box
    
    # Create template corners (box aligned with axes)
    corners = np.array([
        [l/2, w/2, -h/2],   # front-right-bottom
        [l/2, -w/2, -h/2],  # front-left-bottom
        [-l/2, -w/2, -h/2], # rear-left-bottom
        [-l/2, w/2, -h/2],  # rear-right-bottom
        [l/2, w/2, h/2],    # front-right-top
        [l/2, -w/2, h/2],   # front-left-top
        [-l/2, -w/2, h/2],  # rear-left-top
        [-l/2, w/2, h/2],   # rear-right-top
    ])
    
    # Rotation matrix around z-axis
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    # Rotate corners
    corners = corners @ rotation_matrix.T
    
    # Translate to center
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    
    return corners


def corners_to_center(corners: np.ndarray) -> np.ndarray:
    """
    Convert 8 corner points to center-based representation.
    
    Args:
        corners: (8, 3) array of corner coordinates
    
    Returns:
        Box in center format [x, y, z, w, h, l, yaw]
    """
    # Calculate center
    center = np.mean(corners, axis=0)
    
    # Calculate dimensions
    min_coords = np.min(corners, axis=0)
    max_coords = np.max(corners, axis=0)
    
    # This is an approximation; actual rotation needs to be computed
    # from the corner orientations
    dims = max_coords - min_coords
    
    # Estimate yaw from corner arrangement
    # Use front face corners (assuming first 2 corners are front)
    front_center = (corners[0] + corners[1]) / 2
    direction = front_center - center
    yaw = np.arctan2(direction[1], direction[0])
    
    return np.array([
        center[0], center[1], center[2],
        dims[1], dims[2], dims[0],  # w, h, l
        yaw
    ])


def lidar_to_camera(
    box: np.ndarray,
    calibration: np.ndarray
) -> np.ndarray:
    """
    Transform box from LiDAR coordinates to camera coordinates.
    
    Args:
        box: Box in LiDAR coordinates [x, y, z, w, h, l, yaw]
        calibration: 4x4 transformation matrix from LiDAR to camera
    
    Returns:
        Box in camera coordinates
    """
    # Transform center point
    center_lidar = np.array([box[0], box[1], box[2], 1.0])
    center_camera = calibration @ center_lidar
    center_camera = center_camera[:3] / center_camera[3]
    
    # Rotation needs to be adjusted based on coordinate system
    # This is a simplified version
    yaw_camera = box[6]  # May need adjustment based on calibration
    
    result = box.copy()
    result[:3] = center_camera
    result[6] = yaw_camera
    
    return result


def camera_to_lidar(
    box: np.ndarray,
    calibration: np.ndarray
) -> np.ndarray:
    """
    Transform box from camera coordinates to LiDAR coordinates.
    
    Args:
        box: Box in camera coordinates
        calibration: 4x4 transformation matrix from LiDAR to camera
    
    Returns:
        Box in LiDAR coordinates
    """
    # Inverse transformation
    inv_calibration = np.linalg.inv(calibration)
    return lidar_to_camera(box, inv_calibration)


def boxes_to_bev(
    boxes: np.ndarray,
    resolution: float = 0.1,
    x_range: Tuple[float, float] = (-50, 50),
    y_range: Tuple[float, float] = (-50, 50)
) -> np.ndarray:
    """
    Project 3D boxes to bird's eye view grid.
    
    Args:
        boxes: (N, 7) array of boxes
        resolution: Grid resolution in meters
        x_range: (min, max) range for x-axis
        y_range: (min, max) range for y-axis
    
    Returns:
        BEV occupancy grid
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)
    
    bev_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    for box in boxes:
        # Get corners in BEV
        corners_3d = center_to_corners(box)
        corners_bev = corners_3d[:4, :2]  # Bottom 4 corners, x-y only
        
        # Convert to grid coordinates
        corners_grid = np.zeros_like(corners_bev, dtype=np.int32)
        corners_grid[:, 0] = ((corners_bev[:, 0] - x_min) / resolution).astype(np.int32)
        corners_grid[:, 1] = ((corners_bev[:, 1] - y_min) / resolution).astype(np.int32)
        
        # Fill polygon (simplified - just mark corners)
        for corner in corners_grid:
            if 0 <= corner[0] < grid_width and 0 <= corner[1] < grid_height:
                bev_grid[corner[1], corner[0]] = 1
    
    return bev_grid


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi].
    
    Args:
        angle: Angle in radians
    
    Returns:
        Normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest difference between two angles.
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
    
    Returns:
        Angle difference in [-pi, pi]
    """
    diff = angle1 - angle2
    return normalize_angle(diff)
