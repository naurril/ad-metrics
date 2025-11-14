"""
Tests for IoU calculations.
"""

import pytest
import numpy as np
from admetrics.detection.iou import (
    calculate_iou_3d,
    calculate_iou_bev,
    calculate_iou_batch,
    calculate_giou_3d
)


class TestIoU3D:
    """Test 3D IoU calculations."""
    
    def test_identical_boxes(self):
        """Test IoU of identical boxes should be 1.0"""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 4, 2, 1.5, 0]
        
        iou = calculate_iou_3d(box1, box2)
        assert np.isclose(iou, 1.0), f"Expected 1.0, got {iou}"
    
    def test_non_overlapping_boxes(self):
        """Test IoU of non-overlapping boxes should be 0.0"""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [10, 10, 0, 4, 2, 1.5, 0]
        
        iou = calculate_iou_3d(box1, box2)
        assert np.isclose(iou, 0.0), f"Expected 0.0, got {iou}"
    
    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes"""
        box1 = [0, 0, 0, 4, 2, 2, 0]
        box2 = [1, 0, 0, 4, 2, 2, 0]  # Moved closer to create overlap
        
        iou = calculate_iou_3d(box1, box2)
        
        # Box1: x=[-1,1], y=[-2,2], z=[-1,1] -> volume = 2*4*2 = 16
        # Box2: x=[0,2], y=[-2,2], z=[-1,1] -> volume = 2*4*2 = 16
        # Overlap: x=[0,1] (width=1), y=[-2,2] (width=4), z=[-1,1] (height=2)
        # Overlap volume = 1*4*2 = 8
        # Union = 16 + 16 - 8 = 24
        # IoU = 8/24 = 0.333...
        assert 0.3 < iou < 0.4, f"Expected ~0.333, got {iou}"
    
    def test_rotated_boxes(self):
        """Test IoU with rotated boxes"""
        box1 = [0, 0, 0, 4, 2, 2, 0]
        box2 = [0, 0, 0, 4, 2, 2, np.pi/4]
        
        iou = calculate_iou_3d(box1, box2)
        
        # Rotated boxes should have some overlap but not complete
        assert 0 < iou < 1, f"Expected 0 < IoU < 1, got {iou}"
    
    def test_vertical_offset(self):
        """Test boxes with vertical offset"""
        box1 = [0, 0, 0, 4, 2, 2, 0]
        box2 = [0, 0, 3, 4, 2, 2, 0]  # Offset in z
        
        iou = calculate_iou_3d(box1, box2)
        assert np.isclose(iou, 0.0), f"Expected 0.0 (no overlap), got {iou}"
    
    def test_numpy_array_input(self):
        """Test with numpy array input"""
        box1 = np.array([0, 0, 0, 4, 2, 1.5, 0])
        box2 = np.array([0, 0, 0, 4, 2, 1.5, 0])
        
        iou = calculate_iou_3d(box1, box2)
        assert np.isclose(iou, 1.0)


class TestIoUBEV:
    """Test Bird's Eye View IoU."""
    
    def test_identical_boxes_bev(self):
        """Test BEV IoU of identical boxes"""
        box1 = [0, 0, 0, 4, 2, 2, 0]
        box2 = [0, 0, 5, 4, 2, 2, 0]  # Different height but same BEV
        
        iou = calculate_iou_bev(box1, box2)
        assert np.isclose(iou, 1.0), f"Expected 1.0, got {iou}"
    
    def test_bev_partial_overlap(self):
        """Test BEV IoU with partial overlap"""
        box1 = [0, 0, 0, 4, 2, 2, 0]
        box2 = [1, 0, 0, 4, 2, 2, 0]  # Moved closer to create overlap
        
        iou = calculate_iou_bev(box1, box2)
        
        # Box1 BEV: x=[-1,1], y=[-2,2] -> area = 2*4 = 8
        # Box2 BEV: x=[0,2], y=[-2,2] -> area = 2*4 = 8
        # Overlap BEV: x=[0,1] (width=1), y=[-2,2] (width=4) -> area = 1*4 = 4
        # Union = 8 + 8 - 4 = 12
        # IoU = 4/12 = 0.333...
        assert 0.3 < iou < 0.4, f"Expected ~0.333, got {iou}"
    
    def test_bev_no_overlap(self):
        """Test BEV IoU with no overlap"""
        box1 = [0, 0, 0, 4, 2, 2, 0]
        box2 = [10, 10, 0, 4, 2, 2, 0]
        
        iou = calculate_iou_bev(box1, box2)
        assert np.isclose(iou, 0.0)


class TestIoUBatch:
    """Test batch IoU calculations."""
    
    def test_batch_iou_3d(self):
        """Test batch 3D IoU"""
        boxes1 = np.array([
            [0, 0, 0, 4, 2, 1.5, 0],
            [5, 5, 0, 3, 2, 1.5, 0]
        ])
        boxes2 = np.array([
            [0, 0, 0, 4, 2, 1.5, 0],
            [1, 0, 0, 4, 2, 1.5, 0]
        ])
        
        ious = calculate_iou_batch(boxes1, boxes2, mode="3d")
        
        assert ious.shape == (2, 2), f"Expected shape (2, 2), got {ious.shape}"
        assert np.isclose(ious[0, 0], 1.0), "Diagonal should be 1.0"
    
    def test_batch_iou_bev(self):
        """Test batch BEV IoU"""
        boxes1 = np.array([
            [0, 0, 0, 4, 2, 1.5, 0]
        ])
        boxes2 = np.array([
            [0, 0, 0, 4, 2, 1.5, 0],
            [1, 0, 0, 4, 2, 1.5, 0]  # Moved closer to create overlap
        ])
        
        ious = calculate_iou_batch(boxes1, boxes2, mode="bev")
        
        assert ious.shape == (1, 2)
        assert np.isclose(ious[0, 0], 1.0)
        assert 0 < ious[0, 1] < 1  # Should have partial overlap


class TestGIoU:
    """Test Generalized IoU."""
    
    def test_giou_identical(self):
        """Test GIoU of identical boxes"""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 4, 2, 1.5, 0]
        
        giou = calculate_giou_3d(box1, box2)
        assert np.isclose(giou, 1.0)
    
    def test_giou_non_overlapping(self):
        """Test GIoU of non-overlapping boxes"""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [10, 10, 10, 4, 2, 1.5, 0]
        
        giou = calculate_giou_3d(box1, box2)
        
        # GIoU should be negative when boxes don't overlap
        assert giou < 0, f"Expected negative GIoU, got {giou}"
    
    def test_giou_range(self):
        """Test GIoU is in valid range [-1, 1]"""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [2, 0, 0, 4, 2, 1.5, 0]
        
        giou = calculate_giou_3d(box1, box2)
        assert -1 <= giou <= 1, f"GIoU should be in [-1, 1], got {giou}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
