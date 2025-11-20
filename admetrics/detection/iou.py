"""
3D and BEV IoU (Intersection over Union) calculations for 3D bounding boxes.
"""

import numpy as np
from typing import Union, List, Tuple


def calculate_iou_3d(
    box1: Union[np.ndarray, List[float]],
    box2: Union[np.ndarray, List[float]],
    box_format: str = "xyzwhlr"
) -> float:
    """
    Calculate 3D IoU between two 3D bounding boxes.
    
    Args:
        box1: First 3D bounding box [x, y, z, w, h, l, rotation]
              x, y, z: center coordinates
              w: width, h: height, l: length
              rotation: yaw angle in radians
        box2: Second 3D bounding box in same format
        box_format: Format of boxes ('xyzwhlr' or 'xyzhwlr')
    
    Returns:
        IoU value between 0 and 1
        
    Example:
        >>> box1 = [0, 0, 0, 4, 2, 1.5, 0]
        >>> box2 = [1, 0, 0, 4, 2, 1.5, 0]
        >>> iou = calculate_iou_3d(box1, box2)
    """
    box1 = np.array(box1, dtype=np.float64)
    box2 = np.array(box2, dtype=np.float64)
    
    # Calculate BEV IoU first (birds-eye view)
    bev_iou = calculate_iou_bev(box1, box2, box_format)
    
    if bev_iou == 0:
        return 0.0
    
    # Calculate height overlap
    # Extract z (height center) and h (height dimension)
    if box_format == "xyzwhlr":
        z1, h1 = box1[2], box1[4]
        z2, h2 = box2[2], box2[4]
    else:  # xyzhwlr
        z1, h1 = box1[2], box1[3]
        z2, h2 = box2[2], box2[3]
    
    # Calculate bottom and top of each box
    bottom1, top1 = z1 - h1 / 2, z1 + h1 / 2
    bottom2, top2 = z2 - h2 / 2, z2 + h2 / 2
    
    # Calculate height intersection
    height_intersection = max(0, min(top1, top2) - max(bottom1, bottom2))
    
    if height_intersection == 0:
        return 0.0
    
    # Calculate 3D intersection and union
    bev_intersection = _calculate_bev_intersection(box1, box2, box_format)
    intersection_3d = bev_intersection * height_intersection
    
    # Calculate volumes
    if box_format == "xyzwhlr":
        w1, h1, l1 = box1[3], box1[4], box1[5]
        w2, h2, l2 = box2[3], box2[4], box2[5]
    else:  # xyzhwlr
        h1, w1, l1 = box1[3], box1[4], box1[5]
        h2, w2, l2 = box2[3], box2[4], box2[5]
    
    volume1 = w1 * h1 * l1
    volume2 = w2 * h2 * l2
    
    union_3d = volume1 + volume2 - intersection_3d
    
    if union_3d == 0:
        return 0.0
    
    return float(intersection_3d / union_3d)


def calculate_iou_bev(
    box1: Union[np.ndarray, List[float]],
    box2: Union[np.ndarray, List[float]],
    box_format: str = "xyzwhlr"
) -> float:
    """
    Calculate Bird's Eye View (BEV) IoU between two 3D bounding boxes.
    
    This projects the boxes to 2D (x-y plane) and calculates IoU.
    
    Args:
        box1: First 3D bounding box [x, y, z, w, h, l, rotation]
        box2: Second 3D bounding box
        box_format: Format of boxes
    
    Returns:
        BEV IoU value between 0 and 1
        
    Example:
        >>> box1 = [0, 0, 0, 4, 2, 1.5, 0]
        >>> box2 = [1, 0, 0, 4, 2, 1.5, 0]
        >>> iou = calculate_iou_bev(box1, box2)
    """
    box1 = np.array(box1, dtype=np.float64)
    box2 = np.array(box2, dtype=np.float64)
    
    intersection = _calculate_bev_intersection(box1, box2, box_format)
    
    if intersection == 0:
        return 0.0
    
    # Calculate areas
    if box_format == "xyzwhlr":
        area1 = box1[3] * box1[5]  # w * l
        area2 = box2[3] * box2[5]
    else:  # xyzhwlr
        area1 = box1[4] * box1[5]  # w * l
        area2 = box2[4] * box2[5]
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def _calculate_bev_intersection(
    box1: np.ndarray,
    box2: np.ndarray,
    box_format: str = "xyzwhlr"
) -> float:
    """
    Calculate the BEV intersection area using rotated rectangle intersection.
    
    This is a simplified but robust implementation using convex hull after transformation.
    """
    import itertools
    
    # Get corners in BEV (x-y plane)
    corners1 = _get_bev_corners(box1, box_format)
    corners2 = _get_bev_corners(box2, box_format)
    
    # Use polygon intersection - try with cv2-style algorithm
    intersection_polygon = _polygon_intersection(corners1, corners2)
    
    if len(intersection_polygon) < 3:
        return 0.0
    
    return _polygon_area(intersection_polygon)


def _get_bev_corners(box: np.ndarray, box_format: str = "xyzwhlr") -> np.ndarray:
    """
    Get the 4 corners of a box in BEV (bird's eye view).
    
    Returns: (4, 2) array of corner coordinates in x-y plane
    """
    x, y = box[0], box[1]
    
    if box_format == "xyzwhlr":
        w, l, r = box[3], box[5], box[6]
    else:  # xyzhwlr
        w, l, r = box[4], box[5], box[6]
    
    # Create corners in local coordinate system (counter-clockwise order)
    # Front-right, rear-right, rear-left, front-left
    corners_local = np.array([
        [l/2, w/2],
        [-l/2, w/2],
        [-l/2, -w/2],
        [l/2, -w/2]
    ])
    
    # Rotation matrix
    cos_r, sin_r = np.cos(r), np.sin(r)
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    # Rotate and translate
    corners_global = corners_local @ rotation_matrix.T
    corners_global[:, 0] += x
    corners_global[:, 1] += y
    
    return corners_global


def _polygon_intersection(poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    """
    Calculate intersection of two convex polygons using Sutherland-Hodgman algorithm.
    
    Args:
        poly1: (N, 2) array of vertices 
        poly2: (M, 2) array of vertices (the clipping polygon)
    
    Returns:
        (K, 2) array of intersection polygon vertices
    """
    def _is_inside(point, edge_start, edge_end):
        """Test if point is on the left side (inside) of the edge."""
        return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - 
                (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= -1e-9
    
    def _line_intersect(p1, p2, p3, p4):
        """Find intersection of line p1-p2 with line p3-p4."""
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return p1
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    
    output = list(poly1)
    
    for i in range(len(poly2)):
        if len(output) == 0:
            break
        
        edge_start = poly2[i]
        edge_end = poly2[(i + 1) % len(poly2)]
        
        input_list = output
        output = []
        
        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]
            
            if _is_inside(current, edge_start, edge_end):
                if not _is_inside(previous, edge_start, edge_end):
                    intersection = _line_intersect(previous, current, edge_start, edge_end)
                    output.append(intersection)
                output.append(current)
            elif _is_inside(previous, edge_start, edge_end):
                intersection = _line_intersect(previous, current, edge_start, edge_end)
                output.append(intersection)
    
    if len(output) < 3:
        return np.array([])
    
    return np.array(output)


def _polygon_area(vertices: np.ndarray) -> float:
    """
    Calculate area of a polygon using the shoelace formula.
    
    Args:
        vertices: (N, 2) array of polygon vertices
    
    Returns:
        Area of the polygon
    """
    if len(vertices) < 3:
        return 0.0
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    return float(area)


def calculate_iou_batch(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    box_format: str = "xyzwhlr",
    mode: str = "3d"
) -> np.ndarray:
    """
    Calculate IoU for batches of boxes.
    
    Args:
        boxes1: (N, 7) array of N bounding boxes
        boxes2: (M, 7) array of M bounding boxes
        box_format: Format of boxes
        mode: '3d' for 3D IoU, 'bev' for BEV IoU
    
    Returns:
        (N, M) array of IoU values
        
    Example:
        >>> boxes1 = np.array([[0, 0, 0, 4, 2, 1.5, 0],
        ...                    [5, 5, 0, 3, 2, 1.5, 0]])
        >>> boxes2 = np.array([[1, 0, 0, 4, 2, 1.5, 0]])
        >>> ious = calculate_iou_batch(boxes1, boxes2)
        >>> print(ious.shape)  # (2, 1)
    """
    N = len(boxes1)
    M = len(boxes2)
    ious = np.zeros((N, M), dtype=np.float32)
    
    calc_func = calculate_iou_3d if mode == "3d" else calculate_iou_bev
    
    for i in range(N):
        for j in range(M):
            ious[i, j] = calc_func(boxes1[i], boxes2[j], box_format)
    
    return ious


def _get_3d_corners(box: np.ndarray, box_format: str = "xyzwhlr") -> np.ndarray:
    """Get the 8 corners of a 3D bounding box."""
    x, y, z = box[0], box[1], box[2]
    
    if box_format == "xyzwhlr":
        w, h, l, r = box[3], box[4], box[5], box[6]
    else:  # xyzhwlr
        h, w, l, r = box[3], box[4], box[5], box[6]

    # 3D corners in local frame
    corners_local = np.array([
        [l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
        [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
    ])

    # Rotation matrix for yaw
    cos_r, sin_r = np.cos(r), np.sin(r)
    rotation_matrix = np.array([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ])

    # Rotate and translate
    corners_global = corners_local @ rotation_matrix.T
    corners_global += box[:3]
    
    return corners_global


def calculate_giou_3d(
    box1: Union[np.ndarray, List[float]],
    box2: Union[np.ndarray, List[float]],
    box_format: str = "xyzwhlr"
) -> float:
    """
    Calculate Generalized IoU (GIoU) for 3D boxes.
    
    GIoU = IoU - (volume(C) - volume(A ∪ B)) / volume(C), where C is the smallest
    enclosing box that contains both A and B.
    
    Args:
        box1: First 3D bounding box
        box2: Second 3D bounding box
        box_format: Format of boxes
    
    Returns:
        GIoU value between -1 and 1
    """
    box1 = np.array(box1, dtype=np.float64)
    box2 = np.array(box2, dtype=np.float64)
    
    iou = calculate_iou_3d(box1, box2, box_format)
    
    # To compute the enclosing box, we need the corners of the boxes
    corners1 = _get_3d_corners(box1, box_format)
    corners2 = _get_3d_corners(box2, box_format)
    
    all_corners = np.vstack((corners1, corners2))
    
    min_coords = np.min(all_corners, axis=0)
    max_coords = np.max(all_corners, axis=0)
    
    enclosing_volume = np.prod(max_coords - min_coords)
    
    # Calculate volumes
    if box_format == "xyzwhlr":
        w1, h1, l1 = box1[3:6]
        w2, h2, l2 = box2[3:6]
        volume1 = w1 * h1 * l1
        volume2 = w2 * h2 * l2
    else:  # xyzhwlr
        h1, w1, l1 = box1[3:6]
        h2, w2, l2 = box2[3:6]
        volume1 = w1 * h1 * l1
        volume2 = w2 * h2 * l2
    
    union = volume1 + volume2 - (iou * (volume1 + volume2) / (1 + iou) if iou > 0 else volume1 + volume2)
    
    giou = iou - (enclosing_volume - union) / (enclosing_volume + 1e-8)
    
    return float(giou)