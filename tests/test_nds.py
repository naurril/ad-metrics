"""
Tests for NuScenes Detection Score.
"""

import pytest
import numpy as np
from admetrics.detection.nds import calculate_nds, calculate_tp_metrics, calculate_nds_detailed


class TestNDS:
    """Test NuScenes Detection Score."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for NDS calculation."""
        predictions = [
            {
                'box': [0, 0, 0, 4, 2, 1.5, 0],
                'score': 0.9,
                'class': 'car',
                'velocity': [1.0, 0.0]
            },
            {
                'box': [5, 0, 0, 2, 1, 1, 0],
                'score': 0.8,
                'class': 'pedestrian',
                'velocity': [0.5, 0.0]
            }
        ]
        ground_truth = [
            {
                'box': [0.2, 0, 0, 4, 2, 1.5, 0],
                'class': 'car',
                'velocity': [1.1, 0.0]
            },
            {
                'box': [5.1, 0, 0, 2, 1, 1, 0],
                'class': 'pedestrian',
                'velocity': [0.6, 0.0]
            }
        ]
        return predictions, ground_truth
    
    def test_nds_calculation(self, sample_data):
        """Test basic NDS calculation."""
        predictions, ground_truth = sample_data
        
        nds = calculate_nds(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car', 'pedestrian'],
            iou_threshold=0.5
        )
        
        assert 0 <= nds <= 1, f"NDS should be in [0, 1], got {nds}"
    
    def test_perfect_nds(self):
        """Test NDS with perfect predictions."""
        predictions = [
            {
                'box': [0, 0, 0, 4, 2, 1.5, 0],
                'score': 1.0,
                'class': 'car',
                'velocity': [1.0, 0.0]
            }
        ]
        ground_truth = [
            {
                'box': [0, 0, 0, 4, 2, 1.5, 0],
                'class': 'car',
                'velocity': [1.0, 0.0]
            }
        ]
        
        nds = calculate_nds(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car']
        )
        
        # Perfect match should give high NDS (allowing for floating point precision)
        assert nds > 0.89, f"Perfect match should give high NDS, got {nds}"


class TestTPMetrics:
    """Test True Positive metrics."""
    
    def test_tp_metrics_calculation(self):
        """Test TP metrics calculation."""
        predictions = [
            {
                'box': [0, 0, 0, 4, 2, 1.5, 0],
                'score': 0.9,
                'class': 'car',
                'velocity': [1.0, 0.0]
            }
        ]
        ground_truth = [
            {
                'box': [0.5, 0, 0, 4, 2, 1.5, 0.1],
                'class': 'car',
                'velocity': [1.2, 0.0]
            }
        ]
        
        metrics = calculate_tp_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        assert 'ate' in metrics  # Average Translation Error
        assert 'ase' in metrics  # Average Scale Error
        assert 'aoe' in metrics  # Average Orientation Error
        assert 'ave' in metrics  # Average Velocity Error
        assert 'aae' in metrics  # Average Attribute Error
        
        # All errors should be non-negative
        assert all(v >= 0 for k, v in metrics.items() if k != 'num_tp')
    
    def test_tp_metrics_no_matches(self):
        """Test TP metrics with no matches."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [100, 100, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        metrics = calculate_tp_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        # With no matches, errors should be at maximum (1.0)
        assert metrics['ate'] == 1.0
        assert metrics['num_tp'] == 0


class TestNDSDetailed:
    """Test detailed NDS calculation."""
    
    def test_nds_detailed(self):
        """Test detailed NDS calculation."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_nds_detailed(
            predictions=predictions,
            ground_truth=ground_truth,
            class_names=['car']
        )
        
        assert 'nds' in result
        assert 'mAP' in result
        assert 'tp_metrics' in result
        assert 'per_class_nds' in result
        assert 'AP_per_class' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
