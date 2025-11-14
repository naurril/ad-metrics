"""
Tests for trajectory prediction metrics.
"""

import pytest
import numpy as np
from admetrics.prediction.trajectory import (
    calculate_ade,
    calculate_fde,
    calculate_miss_rate,
    calculate_multimodal_ade,
    calculate_multimodal_fde,
    calculate_brier_fde,
    calculate_nll,
    calculate_trajectory_metrics,
    calculate_collision_rate,
    calculate_drivable_area_compliance,
)


class TestADE:
    """Test Average Displacement Error."""
    
    def test_perfect_prediction(self):
        """Test ADE with perfect prediction."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        ade = calculate_ade(pred, gt)
        assert ade == 0.0
    
    def test_constant_error(self):
        """Test ADE with constant displacement."""
        pred = np.array([[1, 0], [2, 0], [3, 0]])
        gt = np.array([[0, 0], [1, 0], [2, 0]])
        
        ade = calculate_ade(pred, gt)
        assert abs(ade - 1.0) < 1e-6
    
    def test_3d_trajectory(self):
        """Test ADE with 3D trajectory."""
        pred = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        gt = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
        
        ade = calculate_ade(pred, gt)
        # Error at t=1: 1.0, t=2: 2.0, mean = 1.0
        assert abs(ade - 1.0) < 1e-6
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        pred = np.array([[0, 0], [1, 1]])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_ade(pred, gt)


class TestFDE:
    """Test Final Displacement Error."""
    
    def test_perfect_final_position(self):
        """Test FDE with perfect final position."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        gt = np.array([[0, 0], [1.5, 1.5], [2, 2]])
        
        fde = calculate_fde(pred, gt)
        assert fde == 0.0
    
    def test_final_error_only(self):
        """Test FDE ignores intermediate errors."""
        pred = np.array([[1, 1], [2, 2], [5, 5]])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        fde = calculate_fde(pred, gt)
        # Final error: sqrt((5-2)^2 + (5-2)^2) = sqrt(18)
        expected = np.sqrt(18)
        assert abs(fde - expected) < 1e-6


class TestMissRate:
    """Test Miss Rate calculation."""
    
    def test_no_miss(self):
        """Test trajectory within threshold."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        gt = np.array([[0, 0], [1, 1], [2.5, 2.5]])
        
        result = calculate_miss_rate(pred, gt, threshold=2.0)
        assert result['is_miss'] is False
        assert result['miss_rate'] == 0.0
    
    def test_miss(self):
        """Test trajectory exceeding threshold."""
        pred = np.array([[0, 0], [1, 1], [5, 5]])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        result = calculate_miss_rate(pred, gt, threshold=2.0)
        assert result['is_miss'] is True
        assert result['miss_rate'] == 1.0
        assert result['fde'] > 2.0


class TestMultimodalADE:
    """Test multi-modal ADE."""
    
    def test_best_mode_selection(self):
        """Test that best mode is selected."""
        # 3 modes: bad, perfect, bad
        preds = np.array([
            [[5, 5], [6, 6], [7, 7]],      # Mode 0: bad
            [[0, 0], [1, 1], [2, 2]],      # Mode 1: perfect
            [[10, 10], [11, 11], [12, 12]] # Mode 2: bad
        ])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        result = calculate_multimodal_ade(preds, gt)
        
        assert result['min_ade'] == 0.0
        assert result['best_mode'] == 1
        assert result['mean_ade'] > result['min_ade']
    
    def test_all_modes(self):
        """Test returning all mode ADEs."""
        preds = np.array([
            [[0, 0], [1, 0], [2, 0]],  # Mode 0
            [[0, 0], [0, 1], [0, 2]]   # Mode 1
        ])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        result = calculate_multimodal_ade(preds, gt, mode="all")
        
        assert 'all_ades' in result
        assert len(result['all_ades']) == 2


class TestMultimodalFDE:
    """Test multi-modal FDE."""
    
    def test_best_mode_fde(self):
        """Test best mode FDE selection."""
        preds = np.array([
            [[0, 0], [1, 1], [5, 5]],   # Mode 0: bad final
            [[0, 0], [1, 1], [2, 2]],   # Mode 1: perfect final
        ])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        result = calculate_multimodal_fde(preds, gt)
        
        assert result['min_fde'] == 0.0
        assert result['best_mode'] == 1


class TestBrierFDE:
    """Test Brier-FDE (probability-weighted)."""
    
    def test_uniform_probabilities(self):
        """Test with uniform probabilities."""
        preds = np.array([
            [[0, 0], [1, 1], [2, 2]],  # FDE = 0
            [[0, 0], [1, 1], [4, 4]]   # FDE = sqrt(8)
        ])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        result = calculate_brier_fde(preds, gt)
        
        # Uniform: (0 + sqrt(8)) / 2
        expected = np.sqrt(8) / 2
        assert abs(result['brier_fde'] - expected) < 1e-6
    
    def test_weighted_probabilities(self):
        """Test with custom probabilities."""
        preds = np.array([
            [[0, 0], [1, 1], [2, 2]],  # FDE = 0
            [[0, 0], [1, 1], [4, 4]]   # FDE = sqrt(8)
        ])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        probs = np.array([0.9, 0.1])  # Heavily weight mode 0
        
        result = calculate_brier_fde(preds, gt, probs)
        
        # Weighted: 0.9 * 0 + 0.1 * sqrt(8)
        expected = 0.1 * np.sqrt(8)
        assert abs(result['brier_fde'] - expected) < 1e-6


class TestNLL:
    """Test Negative Log-Likelihood."""
    
    def test_perfect_prediction_low_nll(self):
        """Test that perfect predictions have low NLL."""
        preds = np.array([[[0, 0], [1, 1], [2, 2]]])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        # Small covariance for confident prediction
        covs = np.tile(np.eye(2) * 0.01, (1, 3, 1, 1))
        
        result = calculate_nll(preds, gt, covs)
        
        # Perfect prediction should have high likelihood (low NLL)
        assert result['nll'] < 10.0
        assert result['log_likelihood'] > -10.0
    
    def test_diagonal_covariance(self):
        """Test with diagonal covariance format."""
        preds = np.array([[[0, 0], [1, 1], [2, 2]]])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        # Diagonal covariances (K, T)
        covs = np.ones((1, 3)) * 0.1
        
        result = calculate_nll(preds, gt, covs)
        
        assert 'nll' in result
        assert result['nll'] < 100.0  # Reasonable NLL


class TestTrajectoryMetrics:
    """Test comprehensive trajectory metrics."""
    
    def test_single_modal(self):
        """Test single-modal trajectory metrics."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        gt = np.array([[0, 0], [1.1, 0.9], [2.1, 1.9]])
        
        result = calculate_trajectory_metrics(pred, gt, multimodal=False)
        
        assert 'ade' in result
        assert 'fde' in result
        assert 'miss_rate' in result
        assert result['ade'] > 0
        assert result['is_miss'] is False
    
    def test_multimodal(self):
        """Test multi-modal trajectory metrics."""
        preds = np.array([
            [[0, 0], [1, 1], [2, 2]],
            [[0, 0], [1, 1], [5, 5]]
        ])
        gt = np.array([[0, 0], [1, 1], [2, 2]])
        
        result = calculate_trajectory_metrics(preds, gt, multimodal=True)
        
        assert 'ade' in result
        assert 'mean_ade' in result
        assert 'num_modes' in result
        assert result['num_modes'] == 2
        assert result['ade'] < result['mean_ade']


class TestCollisionRate:
    """Test collision rate calculation."""
    
    def test_no_collision(self):
        """Test trajectory with no collisions."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [{'center': [5, 5], 'radius': 1.0}]
        
        result = calculate_collision_rate(pred, obstacles, safety_margin=0.5)
        
        assert result['collision_rate'] == 0.0
        assert result['num_collisions'] == 0
    
    def test_collision(self):
        """Test trajectory with collision."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [{'center': [1, 0], 'radius': 0.3}]
        
        result = calculate_collision_rate(pred, obstacles, safety_margin=0.2)
        
        assert result['collision_rate'] > 0.0
        assert result['num_collisions'] > 0
        assert 1 in result['collision_timesteps']
    
    def test_multiple_obstacles(self):
        """Test with multiple obstacles."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [
            {'center': [1, 0], 'radius': 0.3},
            {'center': [2, 0], 'radius': 0.3}
        ]
        
        result = calculate_collision_rate(pred, obstacles, safety_margin=0.2)
        
        # Should hit both obstacles
        assert result['num_collisions'] >= 2


class TestDrivableAreaCompliance:
    """Test drivable area compliance."""
    
    def test_full_compliance_rectangle(self):
        """Test trajectory fully inside rectangle."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        area = {
            'type': 'rectangle',
            'x_min': -1, 'x_max': 3,
            'y_min': -1, 'y_max': 3
        }
        
        result = calculate_drivable_area_compliance(pred, area)
        
        assert result['compliance_rate'] == 1.0
        assert result['num_violations'] == 0
    
    def test_violation_rectangle(self):
        """Test trajectory violating rectangle bounds."""
        pred = np.array([[0, 0], [1, 1], [5, 5]])
        area = {
            'type': 'rectangle',
            'x_min': -1, 'x_max': 3,
            'y_min': -1, 'y_max': 3
        }
        
        result = calculate_drivable_area_compliance(pred, area)
        
        assert result['compliance_rate'] < 1.0
        assert result['num_violations'] > 0
        assert 2 in result['violation_timesteps']
    
    def test_polygon_area(self):
        """Test polygon drivable area."""
        pred = np.array([[0, 0], [1, 1], [2, 2]])
        area = {
            'type': 'polygon',
            'vertices': [[-1, -1], [3, -1], [3, 3], [-1, 3]]
        }
        
        result = calculate_drivable_area_compliance(pred, area)
        
        assert result['compliance_rate'] == 1.0
    
    def test_polygon_violation(self):
        """Test polygon boundary violation."""
        pred = np.array([[0, 0], [1, 1], [5, 5]])
        area = {
            'type': 'polygon',
            'vertices': [[0, 0], [2, 0], [2, 2], [0, 2]]
        }
        
        result = calculate_drivable_area_compliance(pred, area)
        
        assert result['num_violations'] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        pred = np.array([]).reshape(0, 2)
        gt = np.array([]).reshape(0, 2)
        
        # Should handle gracefully or raise appropriate error
        # Depending on desired behavior
        try:
            ade = calculate_ade(pred, gt)
            # If it succeeds, should return NaN or 0
            assert np.isnan(ade) or ade == 0.0
        except (ValueError, IndexError):
            # Or it might raise an error, which is also acceptable
            pass
    
    def test_single_timestep(self):
        """Test with single timestep."""
        pred = np.array([[1, 1]])
        gt = np.array([[0, 0]])
        
        ade = calculate_ade(pred, gt)
        fde = calculate_fde(pred, gt)
        
        # ADE and FDE should be the same for single timestep
        assert abs(ade - fde) < 1e-6
        assert abs(ade - np.sqrt(2)) < 1e-6
