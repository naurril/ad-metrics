"""
Tests for utility functions.
"""

import pytest
import numpy as np
from admetrics.utils.transforms import (
    transform_box,
    rotate_box,
    translate_box,
    convert_box_format,
    center_to_corners,
    corners_to_center,
    normalize_angle
)
from admetrics.utils.matching import (
    match_detections,
    greedy_matching,
    hungarian_matching
)
from admetrics.utils.nms import (
    nms_3d,
    nms_bev,
    nms_per_class
)


class TestTransforms:
    """Test transformation utilities."""
    
    def test_translate_box(self):
        """Test box translation."""
        box = np.array([0, 0, 0, 4, 2, 1.5, 0])
        translation = np.array([1, 2, 3])
        
        translated = translate_box(box, translation)
        
        assert np.allclose(translated[:3], [1, 2, 3])
        assert np.allclose(translated[3:], box[3:])
    
    def test_rotate_box(self):
        """Test box rotation."""
        box = np.array([1, 0, 0, 4, 2, 1.5, 0])
        rotation = np.pi / 2  # 90 degrees
        
        rotated = rotate_box(box, rotation)
        
        # After 90° rotation around origin, (1, 0) should become ~(0, 1)
        assert np.isclose(rotated[0], 0, atol=1e-10)
        assert np.isclose(rotated[1], 1, atol=1e-10)
        assert np.isclose(rotated[6], np.pi / 2)
    
    def test_center_to_corners(self):
        """Test converting center format to corners."""
        box = np.array([0, 0, 0, 4, 2, 2, 0])
        
        corners = center_to_corners(box)
        
        assert corners.shape == (8, 3)
        
        # Check that corners are at expected distances
        center = np.array([0, 0, 0])
        for corner in corners:
            dist = np.linalg.norm(corner[:2] - center[:2])
            # Distance should be sqrt((l/2)^2 + (w/2)^2) for BEV
            expected_dist = np.sqrt((2/2)**2 + (4/2)**2)
            assert np.isclose(dist, expected_dist, atol=0.1)
    
    def test_normalize_angle(self):
        """Test angle normalization."""
        angle1 = 3 * np.pi
        normalized = normalize_angle(angle1)
        
        assert -np.pi <= normalized <= np.pi
        # π and -π are equivalent, so check if it's close to either
        assert np.isclose(np.abs(normalized), np.pi, atol=1e-10)
    
    def test_convert_box_format(self):
        """Test box format conversion."""
        box = np.array([0, 0, 0, 4, 2, 1.5, 0])
        
        # Convert xyzwhlr to xyzhwlr and back
        converted = convert_box_format(box, 'xyzwhlr', 'xyzhwlr')
        back = convert_box_format(converted, 'xyzhwlr', 'xyzwhlr')
        
        assert np.allclose(box, back)


class TestMatching:
    """Test detection matching algorithms."""
    
    @pytest.fixture
    def sample_detections(self):
        """Sample detections for testing."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [5.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        return predictions, ground_truth
    
    def test_greedy_matching(self, sample_detections):
        """Test greedy matching algorithm."""
        predictions, ground_truth = sample_detections
        
        matches, unmatched_preds, unmatched_gts = greedy_matching(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        assert len(matches) == 2
        assert len(unmatched_preds) == 0
        assert len(unmatched_gts) == 0
    
    def test_hungarian_matching(self, sample_detections):
        """Test Hungarian matching algorithm."""
        predictions, ground_truth = sample_detections
        
        matches, unmatched_preds, unmatched_gts = hungarian_matching(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        assert len(matches) == 2
        assert len(unmatched_preds) == 0
        assert len(unmatched_gts) == 0
    
    def test_match_detections_wrapper(self, sample_detections):
        """Test match_detections wrapper function."""
        predictions, ground_truth = sample_detections
        
        # Test greedy
        matches_greedy, _, _ = match_detections(
            predictions, ground_truth, method="greedy"
        )
        
        # Test hungarian
        matches_hungarian, _, _ = match_detections(
            predictions, ground_truth, method="hungarian"
        )
        
        # Both should find 2 matches
        assert len(matches_greedy) == 2
        assert len(matches_hungarian) == 2


class TestNMS:
    """Test Non-Maximum Suppression."""
    
    def test_nms_3d_basic(self):
        """Test basic 3D NMS."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8},  # High overlap
            {'box': [10, 10, 0, 4, 2, 1.5, 0], 'score': 0.7},  # No overlap
        ]
        
        keep_indices = nms_3d(boxes, iou_threshold=0.5)
        
        # Should keep first and third boxes (highest scores with no overlap)
        assert 0 in keep_indices
        assert 2 in keep_indices
        assert 1 not in keep_indices
    
    def test_nms_bev(self):
        """Test BEV NMS."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [0.5, 0, 5, 4, 2, 1.5, 0], 'score': 0.8},  # High overlap in BEV
        ]
        
        keep_indices = nms_bev(boxes, iou_threshold=0.5)
        
        # Should keep only the highest score (boxes have high BEV overlap)
        assert len(keep_indices) == 1
        assert 0 in keep_indices
    
    def test_nms_per_class(self):
        """Test NMS applied per class."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
            {'box': [0, 0, 0, 2, 1, 1, 0], 'score': 0.7, 'class': 'pedestrian'},
        ]
        
        keep_indices = nms_per_class(boxes, iou_threshold=0.5)
        
        # Should keep highest car and the pedestrian
        assert 0 in keep_indices  # Highest car
        assert 2 in keep_indices  # Pedestrian (different class)
    
    def test_nms_with_score_threshold(self):
        """Test NMS with score threshold."""
        boxes = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
            {'box': [10, 10, 0, 4, 2, 1.5, 0], 'score': 0.3},  # Below threshold
        ]
        
        keep_indices = nms_3d(boxes, score_threshold=0.5)
        
        # Should only keep first box
        assert len(keep_indices) == 1
        assert 0 in keep_indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
