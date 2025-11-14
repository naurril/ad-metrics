"""
Visualization utilities for 3D object detection.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_boxes_3d(
    boxes: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    ax: Optional['Axes3D'] = None,
    show: bool = True
) -> 'Axes3D':
    """
    Plot 3D bounding boxes.
    
    Args:
        boxes: List of boxes [x, y, z, w, h, l, yaw]
        labels: Optional labels for each box
        colors: Optional colors for each box
        ax: Matplotlib 3D axis (created if None)
        show: Whether to show the plot
    
    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization")
    
    from admetrics.utils.transforms import center_to_corners
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    if colors is None:
        colors = ['red'] * len(boxes)
    
    for i, box in enumerate(boxes):
        corners = center_to_corners(np.array(box))
        color = colors[i]
        
        # Define the 12 edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            points = corners[edge]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2)
        
        # Add label if provided
        if labels is not None and i < len(labels):
            ax.text(box[0], box[1], box[2] + box[4]/2, labels[i], fontsize=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Bounding Boxes')
    
    if show:
        plt.show()
    
    return ax


def plot_boxes_bev(
    boxes: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None
) -> plt.Axes:
    """
    Plot Bird's Eye View of 3D boxes.
    
    Args:
        boxes: List of boxes
        labels: Optional labels
        colors: Optional colors
        ax: Matplotlib axis
        show: Whether to show
        x_range: X-axis range
        y_range: Y-axis range
    
    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization")
    
    from admetrics.utils.transforms import center_to_corners
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    if colors is None:
        colors = ['red'] * len(boxes)
    
    for i, box in enumerate(boxes):
        corners = center_to_corners(np.array(box))
        # Take bottom 4 corners
        bev_corners = corners[:4, :2]
        
        # Close the polygon
        bev_corners = np.vstack([bev_corners, bev_corners[0]])
        
        ax.plot(bev_corners[:, 0], bev_corners[:, 1], color=colors[i], linewidth=2)
        
        # Draw heading direction
        center = np.array([box[0], box[1]])
        heading_length = box[5] / 2  # Half of length
        heading_end = center + heading_length * np.array([np.cos(box[6]), np.sin(box[6])])
        ax.arrow(center[0], center[1], 
                heading_end[0] - center[0], heading_end[1] - center[1],
                head_width=0.3, head_length=0.5, fc=colors[i], ec=colors[i])
        
        if labels is not None and i < len(labels):
            ax.text(box[0], box[1], labels[i], fontsize=10)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Bird\'s Eye View')
    ax.set_aspect('equal')
    ax.grid(True)
    
    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    
    if show:
        plt.show()
    
    return ax


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    title: str = "Precision-Recall Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        ap: Average Precision value
        title: Plot title
        ax: Matplotlib axis
        show: Whether to show
    
    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, linewidth=2, label=f'AP = {ap:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
    
    if show:
        plt.show()
    
    return ax


def visualize_detection_results(
    predictions: List[Dict],
    ground_truth: List[Dict],
    matches: List[Tuple[int, int]],
    mode: str = "bev",
    show: bool = True
):
    """
    Visualize detection results with TP, FP, FN.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        matches: List of (pred_idx, gt_idx) matches
        mode: 'bev' or '3d'
        show: Whether to show
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization")
    
    matched_pred_idx = {m[0] for m in matches}
    matched_gt_idx = {m[1] for m in matches}
    
    # Prepare boxes and colors
    all_boxes = []
    all_labels = []
    all_colors = []
    
    # True Positives (green)
    for pred_idx, gt_idx in matches:
        all_boxes.append(predictions[pred_idx]['box'])
        all_labels.append(f"TP: {predictions[pred_idx].get('class', 'unknown')}")
        all_colors.append('green')
    
    # False Positives (red)
    for i, pred in enumerate(predictions):
        if i not in matched_pred_idx:
            all_boxes.append(pred['box'])
            all_labels.append(f"FP: {pred.get('class', 'unknown')}")
            all_colors.append('red')
    
    # False Negatives (blue)
    for i, gt in enumerate(ground_truth):
        if i not in matched_gt_idx:
            all_boxes.append(gt['box'])
            all_labels.append(f"FN: {gt.get('class', 'unknown')}")
            all_colors.append('blue')
    
    if mode == "bev":
        plot_boxes_bev(all_boxes, all_labels, all_colors, show=show)
    else:
        plot_boxes_3d(all_boxes, all_labels, all_colors, show=show)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    cmap: str = 'Blues',
    show: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: Class names
        normalize: Whether to normalize
        cmap: Colormap
        show: Whether to show
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization")
    
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if show:
        plt.show()
    
    return ax
