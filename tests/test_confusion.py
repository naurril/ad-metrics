"""
Tests for confusion matrix metrics.
"""

import pytest
import numpy as np
from admetrics.detection.confusion import (
    calculate_tp_fp_fn,
    calculate_confusion_metrics,
    calculate_confusion_matrix_multiclass,
    calculate_specificity
)


class TestConfusionMetrics:
    """Test confusion matrix calculations."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample detection data."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
            {'box': [100, 100, 0, 4, 2, 1.5, 0], 'score': 0.7, 'class': 'car'},  # FP
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [5.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},  # FN
        ]
        return predictions, ground_truth
    
    def test_tp_fp_fn_calculation(self, sample_data):
        """Test TP/FP/FN counting."""
        predictions, ground_truth = sample_data
        
        result = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold=0.5)
        
        assert 'tp' in result
        assert 'fp' in result
        assert 'fn' in result
        
        # Should have 2 TP, 1 FP, 1 FN
        assert result['tp'] == 2
        assert result['fp'] == 1
        assert result['fn'] == 1
    
    def test_perfect_detection(self):
        """Test perfect detection scenario."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_tp_fp_fn(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['tp'] == 1
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_confusion_metrics(self, sample_data):
        """Test precision, recall, F1 calculation."""
        predictions, ground_truth = sample_data
        
        metrics = calculate_confusion_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check ranges
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        
        # TP=2, FP=1, FN=1
        # Precision = 2/3 = 0.667
        # Recall = 2/3 = 0.667
        expected_precision = 2/3
        expected_recall = 2/3
        
        assert np.isclose(metrics['precision'], expected_precision, atol=0.01)
        assert np.isclose(metrics['recall'], expected_recall, atol=0.01)
    
    def test_no_predictions(self):
        """Test with no predictions."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        metrics = calculate_confusion_metrics(predictions, ground_truth)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0


class TestMultiClassConfusion:
    """Test multi-class confusion matrix."""
    
    @pytest.fixture
    def multiclass_data(self):
        """Multi-class detection data."""
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
    
    def test_multiclass_confusion_matrix(self, multiclass_data):
        """Test multi-class confusion matrix."""
        predictions, ground_truth = multiclass_data
        
        result = calculate_confusion_matrix_multiclass(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car', 'pedestrian', 'cyclist'],
            iou_threshold=0.5
        )
        
        assert 'confusion_matrix' in result
        assert 'class_names' in result
        assert 'per_class_metrics' in result
        
        cm = result['confusion_matrix']
        assert cm.shape == (3, 3)
        
        # Check per-class metrics
        assert len(result['per_class_metrics']) == 3
        for cls_name, metrics in result['per_class_metrics'].items():
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics


class TestSpecificity:
    """Test specificity metric."""
    
    def test_specificity_basic(self):
        """Test basic specificity calculation."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [100, 100, 0, 4, 2, 1.5, 0], 'score': 0.7, 'class': 'car'},  # FP
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        # Assume 1000 total negative samples
        total_negatives = 1000
        
        specificity = calculate_specificity(
            predictions, ground_truth, 
            total_negatives=total_negatives,
            iou_threshold=0.5
        )
        
        # FP = 1, TN = 1000 - 1 = 999
        # Specificity = TN / (TN + FP) = 999 / 1000 = 0.999
        assert isinstance(specificity, float)
        assert 0 <= specificity <= 1
        assert np.isclose(specificity, 0.999, atol=0.001)
    
    def test_specificity_perfect(self):
        """Test perfect specificity (no false positives)."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        total_negatives = 1000
        
        specificity = calculate_specificity(
            predictions, ground_truth,
            total_negatives=total_negatives
        )
        
        # FP = 0, TN = 1000
        # Specificity = 1.0
        assert np.isclose(specificity, 1.0, atol=0.001)
    
    def test_specificity_all_false_positives(self):
        """Test specificity when all predictions are false positives."""
        predictions = [
            {'box': [100, 100, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [200, 200, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        total_negatives = 100
        
        specificity = calculate_specificity(
            predictions, ground_truth,
            total_negatives=total_negatives
        )
        
        # FP = 2, TN = 100 - 2 = 98
        # Specificity = 98 / 100 = 0.98
        assert np.isclose(specificity, 0.98, atol=0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
