"""
Tests for Average Precision calculations.
"""

import pytest
import numpy as np
from admetrics.detection.ap import (
    calculate_ap,
    calculate_map,
    calculate_ap_coco_style,
    calculate_precision_recall_curve
)


class TestAP:
    """Test Average Precision calculations."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions for testing."""
        return [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'score': 0.7, 'class': 'car'},
        ]
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Sample ground truth for testing."""
        return [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [5.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
    
    def test_ap_calculation(self, sample_predictions, sample_ground_truth):
        """Test basic AP calculation."""
        result = calculate_ap(
            predictions=sample_predictions,
            ground_truth=sample_ground_truth,
            iou_threshold=0.5
        )
        
        assert 'ap' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 0 <= result['ap'] <= 1
    
    def test_perfect_detection(self):
        """Test AP with perfect detections."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        
        result = calculate_ap(predictions, ground_truth, iou_threshold=0.5)
        
        assert np.isclose(result['ap'], 1.0), f"Expected AP=1.0, got {result['ap']}"
        assert result['num_tp'] == 1
        assert result['num_fp'] == 0
    
    def test_no_detections(self):
        """Test AP with no detections."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        
        result = calculate_ap(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['ap'] == 0.0
        assert result['num_tp'] == 0
    
    def test_no_ground_truth(self):
        """Test AP with no ground truth."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
        ]
        ground_truth = []
        
        result = calculate_ap(predictions, ground_truth, iou_threshold=0.5)
        
        # With no GT, all predictions are FP
        assert result['num_fp'] == 1
        assert result['num_gt'] == 0
    
    def test_ap_with_different_iou_thresholds(self, sample_predictions, sample_ground_truth):
        """Test AP at different IoU thresholds."""
        ap_05 = calculate_ap(sample_predictions, sample_ground_truth, iou_threshold=0.5)
        ap_07 = calculate_ap(sample_predictions, sample_ground_truth, iou_threshold=0.7)
        
        # Higher IoU threshold should give same or lower AP
        assert ap_05['ap'] >= ap_07['ap']


class TestMAP:
    """Test Mean Average Precision."""
    
    @pytest.fixture
    def multi_class_data(self):
        """Multi-class prediction and GT data."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 2, 1, 1, 0], 'score': 0.8, 'class': 'pedestrian'},
            {'box': [10, 0, 0, 2, 1, 1.5, 0], 'score': 0.7, 'class': 'cyclist'},
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [5.2, 0, 0, 2, 1, 1, 0], 'class': 'pedestrian'},
            {'box': [10.5, 0, 0, 2, 1, 1.5, 0], 'class': 'cyclist'},
        ]
        return predictions, ground_truth
    
    def test_map_calculation(self, multi_class_data):
        """Test mAP calculation."""
        predictions, ground_truth = multi_class_data
        
        result = calculate_map(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car', 'pedestrian', 'cyclist'],
            iou_thresholds=0.5
        )
        
        assert 'mAP' in result
        assert 'AP_per_class' in result
        assert 0 <= result['mAP'] <= 1
        assert len(result['AP_per_class']) == 3
    
    def test_map_multiple_thresholds(self, multi_class_data):
        """Test mAP with multiple IoU thresholds."""
        predictions, ground_truth = multi_class_data
        
        result = calculate_map(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car', 'pedestrian', 'cyclist'],
            iou_thresholds=[0.5, 0.7]
        )
        
        assert 'AP_per_threshold' in result
        assert len(result['AP_per_threshold']) == 2


class TestCOCOStyle:
    """Test COCO-style AP."""
    
    def test_coco_ap(self):
        """Test COCO-style AP calculation."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        
        result = calculate_ap_coco_style(predictions, ground_truth)
        
        assert 'AP' in result
        assert 'AP50' in result
        assert 'AP75' in result
        assert 0 <= result['AP'] <= 1


class TestPrecisionRecallCurve:
    """Test precision-recall curve generation."""
    
    def test_pr_curve(self):
        """Test PR curve generation."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        
        precision, recall, scores = calculate_precision_recall_curve(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        assert len(precision) == len(recall)
        assert len(recall) == len(scores)
        assert all(0 <= p <= 1 for p in precision)
        assert all(0 <= r <= 1 for r in recall)


class TestMAPEdgeCases:
    """Test mAP edge cases."""

    def test_no_predictions(self):
        """Test mAP with no predictions."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        result = calculate_map(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car'],
            iou_thresholds=0.5
        )
        assert result['mAP'] == 0.0

    def test_no_ground_truth(self):
        """Test mAP with no ground truth."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
        ]
        ground_truth = []
        result = calculate_map(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car'],
            iou_thresholds=0.5
        )
        assert result['mAP'] == 0.0

    def test_no_predictions_no_ground_truth(self):
        """Test mAP with no predictions and no ground truth."""
        predictions = []
        ground_truth = []
        result = calculate_map(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car'],
            iou_thresholds=0.5
        )
        assert result['mAP'] == 0.0

    def test_missing_score(self):
        """Test mAP with missing score in predictions."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        with pytest.raises(ValueError):
            calculate_map(
                predictions=predictions,
                ground_truth=ground_truth,
                class_names=['car'],
                iou_thresholds=0.5
            )

    def test_missing_box(self):
        """Test mAP with missing box in predictions."""
        predictions = [
            {'score': 0.9, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        with pytest.raises(ValueError):
            calculate_map(
                predictions=predictions,
                ground_truth=ground_truth,
                class_names=['car'],
                iou_thresholds=0.5
            )

    def test_missing_class(self):
        """Test mAP with missing class in predictions."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9},
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
        ]
        with pytest.raises(ValueError):
            calculate_map(
                predictions=predictions,
                ground_truth=ground_truth,
                class_names=['car'],
                iou_thresholds=0.5
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
