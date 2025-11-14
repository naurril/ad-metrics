"""
Tests for localization metrics.
"""

import pytest
import numpy as np
from admetrics.localization.localization import (
    calculate_ate,
    calculate_rte,
    calculate_are,
    calculate_lateral_error,
    calculate_longitudinal_error,
    calculate_convergence_rate,
    calculate_localization_metrics,
    calculate_map_alignment_score,
)


class TestATE:
    """Test Absolute Trajectory Error."""
    
    def test_perfect_localization(self):
        """Test ATE with perfect localization."""
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_ate(pred, gt)
        
        assert result['mean'] == 0.0
        assert result['rmse'] == 0.0
        assert result['max'] == 0.0
    
    def test_constant_offset(self):
        """Test ATE with constant position offset."""
        pred = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_ate(pred, gt)
        
        assert abs(result['mean'] - 1.0) < 1e-6
        assert abs(result['std'] - 0.0) < 1e-6
    
    def test_varying_error(self):
        """Test ATE with varying error."""
        pred = np.array([[0, 0, 0], [1.5, 0, 0], [2, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_ate(pred, gt)
        
        assert result['mean'] > 0
        assert result['std'] > 0
        assert result['max'] == 0.5
    
    def test_with_7d_poses(self):
        """Test ATE with quaternion poses."""
        # [x, y, z, qw, qx, qy, qz]
        pred = np.array([[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0]])
        gt = np.array([[0, 0, 0, 1, 0, 0, 0], [1.1, 0, 0, 1, 0, 0, 0]])
        
        result = calculate_ate(pred, gt)
        
        assert abs(result['mean'] - 0.05) < 1e-6


class TestRTE:
    """Test Relative Trajectory Error."""
    
    def test_perfect_relative_trajectory(self):
        """Test RTE with perfect relative motion."""
        # Straight line, constant velocity
        pred = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
        gt = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
        
        result = calculate_rte(pred, gt, distances=[10])
        
        assert 'rte_10' in result
        # Perfect tracking should have near-zero RTE
        if 'rte_10' in result:
            assert result['rte_10'] < 0.01
    
    def test_accumulating_drift(self):
        """Test RTE with accumulating drift."""
        # Drift increases over distance
        pred = np.array([[0, 0, 0], [10, 0.1, 0], [20, 0.3, 0], [30, 0.6, 0]])
        gt = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
        
        result = calculate_rte(pred, gt, distances=[10, 20])
        
        # Should detect drift
        assert len(result) > 0


class TestARE:
    """Test Absolute Rotation Error."""
    
    def test_perfect_orientation(self):
        """Test ARE with perfect orientation."""
        pred = np.array([[0, 0, 0, 0], [1, 0, 0, 0.5]])
        gt = np.array([[0, 0, 0, 0], [1, 0, 0, 0.5]])
        
        result = calculate_are(pred, gt)
        
        assert result['mean'] == 0.0
        assert result['rmse'] == 0.0
    
    def test_constant_heading_error(self):
        """Test ARE with constant heading error."""
        pred = np.array([[0, 0, 0, 0.1], [1, 0, 0, 0.1]])
        gt = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
        
        result = calculate_are(pred, gt)
        
        # ~5.7 degrees error
        expected = np.rad2deg(0.1)
        assert abs(result['mean'] - expected) < 0.1
    
    def test_with_quaternions(self):
        """Test ARE with quaternion poses."""
        # [x, y, z, qw, qx, qy, qz]
        # Identity quaternion: [1, 0, 0, 0]
        pred = np.array([[0, 0, 0, 1, 0, 0, 0]])
        gt = np.array([[0, 0, 0, 1, 0, 0, 0]])
        
        result = calculate_are(pred, gt)
        
        assert result['mean'] == 0.0


class TestLateralError:
    """Test Lateral (Cross-Track) Error."""
    
    def test_no_lateral_deviation(self):
        """Test lateral error with no cross-track deviation."""
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_lateral_error(pred, gt)
        
        assert result['mean'] == 0.0
        assert result['lane_violation_rate'] == 0.0
    
    def test_constant_lateral_offset(self):
        """Test lateral error with constant offset."""
        pred = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_lateral_error(pred, gt, lane_width=3.5)
        
        assert abs(result['mean'] - 1.0) < 0.1
        assert result['lane_violation_rate'] == 0.0  # Within lane
    
    def test_lane_violation(self):
        """Test lane violation detection."""
        # Large lateral offset exceeding lane width
        pred = np.array([[0, 3, 0], [1, 3, 0], [2, 3, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_lateral_error(pred, gt, lane_width=3.5)
        
        assert result['lane_violations'] > 0
        assert result['lane_violation_rate'] > 0


class TestLongitudinalError:
    """Test Longitudinal (Along-Track) Error."""
    
    def test_no_longitudinal_deviation(self):
        """Test longitudinal error with perfect alignment."""
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_longitudinal_error(pred, gt)
        
        assert abs(result['mean']) < 1e-6
    
    def test_ahead_of_gt(self):
        """Test when prediction is ahead of ground truth."""
        pred = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_longitudinal_error(pred, gt)
        
        # Prediction is ahead, positive longitudinal error
        assert result['mean'] > 0


class TestConvergenceRate:
    """Test convergence rate calculation."""
    
    def test_fast_convergence(self):
        """Test fast convergence to accurate localization."""
        # Error decreases quickly
        errors = np.array([1.0, 0.5, 0.2, 0.05, 0.03, 0.02])
        timestamps = np.array([0, 1, 2, 3, 4, 5])
        
        result = calculate_convergence_rate(errors, timestamps, threshold=0.1)
        
        assert result['converged'] is True
        assert result['time_to_convergence'] <= 3.0
        assert result['convergence_rate'] > 0
    
    def test_no_convergence(self):
        """Test when system doesn't converge."""
        errors = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        
        result = calculate_convergence_rate(errors, threshold=0.1)
        
        assert result['converged'] is False
        assert result['time_to_convergence'] is None
    
    def test_immediate_convergence(self):
        """Test when already converged at start."""
        errors = np.array([0.05, 0.04, 0.03])
        
        result = calculate_convergence_rate(errors, threshold=0.1)
        
        assert result['converged'] is True


class TestLocalizationMetrics:
    """Test comprehensive localization metrics."""
    
    def test_comprehensive_3d_poses(self):
        """Test with 3D position-only poses."""
        pred = np.array([[0, 0, 0], [1, 0.1, 0], [2, 0.2, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = calculate_localization_metrics(pred, gt)
        
        assert 'ate_mean' in result
        assert 'lateral_mean' in result
        assert 'longitudinal_mean' in result
        assert 'are_mean' not in result  # No orientation
    
    def test_comprehensive_with_orientation(self):
        """Test with poses including orientation."""
        pred = np.array([[0, 0, 0, 0], [1, 0.1, 0, 0.1]])
        gt = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
        
        result = calculate_localization_metrics(pred, gt)
        
        assert 'ate_mean' in result
        assert 'lateral_mean' in result
        assert 'are_mean' in result  # Has orientation
    
    def test_with_timestamps(self):
        """Test with timestamps for convergence analysis."""
        pred = np.array([[0, 0, 0], [1, 0.5, 0], [2, 0.1, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        timestamps = np.array([0, 1, 2])
        
        result = calculate_localization_metrics(pred, gt, timestamps=timestamps)
        
        assert 'convergence_converged' in result


class TestMapAlignment:
    """Test HD map alignment scoring."""
    
    def test_perfect_alignment(self):
        """Test when poses perfectly align with lane."""
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        # Lane centerline matching the trajectory
        lanes = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])]
        
        result = calculate_map_alignment_score(pred, lanes, max_distance=2.0)
        
        assert result['mean_distance_to_lane'] < 0.01
        assert result['alignment_rate'] == 1.0
    
    def test_offset_from_lane(self):
        """Test with offset from lane centerline."""
        pred = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]])
        
        # Lane at y=0
        lanes = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])]
        
        result = calculate_map_alignment_score(pred, lanes, max_distance=2.0)
        
        assert abs(result['mean_distance_to_lane'] - 1.0) < 0.1
        assert result['alignment_rate'] == 1.0  # Within 2m
    
    def test_poor_alignment(self):
        """Test with poor map alignment."""
        pred = np.array([[0, 5, 0], [1, 5, 0], [2, 5, 0]])
        
        # Lane far from trajectory
        lanes = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])]
        
        result = calculate_map_alignment_score(pred, lanes, max_distance=2.0)
        
        assert result['mean_distance_to_lane'] > 4.0
        assert result['alignment_rate'] == 0.0  # Outside 2m threshold
    
    def test_multiple_lanes(self):
        """Test with multiple lane options."""
        pred = np.array([[0, 0, 0], [1, 0, 0]])
        
        # Multiple parallel lanes
        lanes = [
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            np.array([[0, 3.5, 0], [1, 3.5, 0], [2, 3.5, 0]])
        ]
        
        result = calculate_map_alignment_score(pred, lanes, max_distance=2.0)
        
        # Should align with closest lane
        assert result['mean_distance_to_lane'] < 0.1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_mismatched_lengths(self):
        """Test error on mismatched trajectory lengths."""
        pred = np.array([[0, 0, 0], [1, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        with pytest.raises(ValueError, match="Number of poses must match"):
            calculate_ate(pred, gt)
    
    def test_single_pose(self):
        """Test with single pose."""
        pred = np.array([[1, 1, 0]])
        gt = np.array([[0, 0, 0]])
        
        result = calculate_ate(pred, gt)
        
        expected_error = np.sqrt(2)
        assert abs(result['mean'] - expected_error) < 0.01
    
    def test_2d_trajectory(self):
        """Test with 2D positions (z implicit)."""
        # Should still work with (N, 3) but z=0
        pred = np.array([[0, 0, 0], [1, 0, 0]])
        gt = np.array([[0, 0, 0], [1, 0.1, 0]])
        
        result = calculate_lateral_error(pred, gt)
        
        # Lateral error should be close to 0.1m but may vary due to direction calculation
        assert result['mean'] < 0.15
