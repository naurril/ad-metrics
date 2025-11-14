"""
Tests for distance-based detection metrics.
"""

import pytest
import numpy as np
from admetrics.detection.distance import (
    calculate_center_distance,
    calculate_orientation_error,
    calculate_size_error,
    calculate_velocity_error,
    calculate_average_distance_error,
    calculate_translation_error_bins
)


class TestCenterDistance:
    """Test center distance calculations."""
    
    def test_identical_boxes(self):
        """Test distance between identical boxes is zero."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 4, 2, 1.5, 0]
        
        dist = calculate_center_distance(box1, box2)
        assert np.isclose(dist, 0.0), f"Expected 0.0, got {dist}"
    
    def test_euclidean_distance(self):
        """Test 3D Euclidean distance."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [3, 4, 0, 4, 2, 1.5, 0]
        
        dist = calculate_center_distance(box1, box2, distance_type="euclidean")
        expected = np.sqrt(3**2 + 4**2)  # 5.0
        assert np.isclose(dist, expected), f"Expected {expected}, got {dist}"
    
    def test_bev_distance(self):
        """Test bird's eye view (2D) distance."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [3, 4, 10, 4, 2, 1.5, 0]  # Different z
        
        dist = calculate_center_distance(box1, box2, distance_type="bev")
        expected = np.sqrt(3**2 + 4**2)  # 5.0, ignores z
        assert np.isclose(dist, expected), f"Expected {expected}, got {dist}"
    
    def test_vertical_distance(self):
        """Test vertical (z-axis) distance."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [10, 10, 5, 4, 2, 1.5, 0]
        
        dist = calculate_center_distance(box1, box2, distance_type="vertical")
        assert np.isclose(dist, 5.0), f"Expected 5.0, got {dist}"
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        box1 = np.array([0, 0, 0, 4, 2, 1.5, 0])
        box2 = np.array([1, 0, 0, 4, 2, 1.5, 0])
        
        dist = calculate_center_distance(box1, box2)
        assert np.isclose(dist, 1.0)
    
    def test_invalid_distance_type(self):
        """Test with invalid distance type."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [1, 0, 0, 4, 2, 1.5, 0]
        
        with pytest.raises(ValueError):
            calculate_center_distance(box1, box2, distance_type="invalid")


class TestOrientationError:
    """Test orientation error calculations."""
    
    def test_zero_error(self):
        """Test zero orientation error."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 4, 2, 1.5, 0]
        
        error = calculate_orientation_error(box1, box2)
        assert np.isclose(error, 0.0), f"Expected 0.0, got {error}"
    
    def test_small_error_radians(self):
        """Test small orientation error in radians."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0.1]
        box2 = [0, 0, 0, 4, 2, 1.5, 0.0]
        
        error = calculate_orientation_error(box1, box2, error_type="absolute")
        assert np.isclose(error, 0.1, atol=0.01), f"Expected 0.1, got {error}"
    
    def test_error_degrees(self):
        """Test orientation error in degrees."""
        box1 = [0, 0, 0, 4, 2, 1.5, np.pi/2]  # 90 degrees
        box2 = [0, 0, 0, 4, 2, 1.5, 0]
        
        error = calculate_orientation_error(box1, box2, error_type="degrees")
        assert np.isclose(error, 90.0, atol=0.1), f"Expected 90, got {error}"
    
    def test_angle_normalization(self):
        """Test angle normalization to [-pi, pi]."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 4, 2, 1.5, 2*np.pi]
        
        error = calculate_orientation_error(box1, box2)
        assert np.isclose(error, 0.0, atol=0.01), "2π and 0 should be equivalent"
    
    def test_opposite_orientations(self):
        """Test opposite orientations."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 4, 2, 1.5, np.pi]
        
        error = calculate_orientation_error(box1, box2)
        assert np.isclose(error, np.pi, atol=0.01), f"Expected π, got {error}"


class TestSizeError:
    """Test size/dimension error calculations."""
    
    def test_identical_sizes(self):
        """Test zero size error for identical boxes."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 4, 2, 1.5, 0]
        
        errors = calculate_size_error(box1, box2)
        
        assert np.isclose(errors['width_error'], 0.0)
        assert np.isclose(errors['height_error'], 0.0)
        assert np.isclose(errors['length_error'], 0.0)
        assert np.isclose(errors['volume_error'], 0.0)
    
    def test_absolute_error(self):
        """Test absolute size errors."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [0, 0, 0, 5, 3, 2.0, 0]
        
        errors = calculate_size_error(box1, box2, error_type="absolute")
        
        assert np.isclose(errors['width_error'], 1.0)
        assert np.isclose(errors['height_error'], 1.0)
        assert np.isclose(errors['length_error'], 0.5)
    
    def test_relative_error(self):
        """Test relative size errors."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]  # Volume = 12
        box2 = [0, 0, 0, 2, 2, 1.5, 0]  # Volume = 6
        
        errors = calculate_size_error(box1, box2, error_type="relative")
        
        # Width: (4-2)/2 = 1.0 (100% larger)
        assert np.isclose(errors['width_error'], 1.0)
        # Height: (2-2)/2 = 0
        assert np.isclose(errors['height_error'], 0.0)
    
    def test_percentage_error(self):
        """Test percentage size errors."""
        box1 = [0, 0, 0, 3, 2, 1.5, 0]
        box2 = [0, 0, 0, 2, 2, 1.5, 0]
        
        errors = calculate_size_error(box1, box2, error_type="percentage")
        
        # Width: (3-2)/2 * 100 = 50%
        assert np.isclose(errors['width_error'], 50.0)
    
    def test_volume_error(self):
        """Test volume error calculation."""
        box1 = [0, 0, 0, 4, 2, 2, 0]  # Volume = 16
        box2 = [0, 0, 0, 2, 2, 2, 0]  # Volume = 8
        
        errors = calculate_size_error(box1, box2, error_type="absolute")
        
        assert np.isclose(errors['volume_error'], 8.0)


class TestVelocityError:
    """Test velocity error calculations."""
    
    def test_zero_velocity_error(self):
        """Test zero velocity error."""
        vel1 = [1.0, 2.0]
        vel2 = [1.0, 2.0]
        
        error = calculate_velocity_error(vel1, vel2)
        assert np.isclose(error, 0.0), f"Expected 0.0, got {error}"
    
    def test_euclidean_velocity_error(self):
        """Test Euclidean velocity error."""
        vel1 = [3.0, 4.0]
        vel2 = [0.0, 0.0]
        
        error = calculate_velocity_error(vel1, vel2, error_type="euclidean")
        expected = np.sqrt(3**2 + 4**2)  # 5.0
        assert np.isclose(error, expected), f"Expected {expected}, got {error}"
    
    def test_manhattan_velocity_error(self):
        """Test Manhattan velocity error."""
        vel1 = [3.0, 4.0]
        vel2 = [0.0, 0.0]
        
        error = calculate_velocity_error(vel1, vel2, error_type="manhattan")
        expected = 7.0  # |3| + |4|
        assert np.isclose(error, expected), f"Expected {expected}, got {error}"
    
    def test_angular_velocity_error(self):
        """Test angular velocity error."""
        vel1 = [1.0, 0.0]
        vel2 = [0.0, 1.0]
        
        error = calculate_velocity_error(vel1, vel2, error_type="angular")
        expected = np.pi / 2  # 90 degrees
        assert np.isclose(error, expected, atol=0.01), f"Expected {expected}, got {error}"
    
    def test_3d_velocity(self):
        """Test 3D velocity error."""
        vel1 = [1.0, 2.0, 3.0]
        vel2 = [1.0, 2.0, 0.0]
        
        error = calculate_velocity_error(vel1, vel2)
        assert np.isclose(error, 3.0), f"Expected 3.0, got {error}"
    
    def test_zero_velocity_angular(self):
        """Test angular error with zero velocity."""
        vel1 = [0.0, 0.0]
        vel2 = [1.0, 0.0]
        
        error = calculate_velocity_error(vel1, vel2, error_type="angular")
        assert np.isclose(error, 0.0), "Should return 0 for zero velocity"


class TestAverageDistanceError:
    """Test average distance error for matched detections."""
    
    @pytest.fixture
    def matched_data(self):
        """Sample data with good matches."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [10.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},  # 0.5m away
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}   # 0.5m away
        ]
        return predictions, ground_truth
    
    def test_average_distance_error(self, matched_data):
        """Test average distance error calculation."""
        predictions, ground_truth = matched_data
        
        result = calculate_average_distance_error(
            predictions, ground_truth, 
            iou_threshold=0.5
        )
        
        assert 'mean_distance_error' in result
        assert 'median_distance_error' in result
        assert 'std_distance_error' in result
        assert 'num_matched' in result
        
        # Both boxes are 0.5m away
        assert result['num_matched'] == 2
        assert np.isclose(result['mean_distance_error'], 0.5, atol=0.1)
    
    def test_no_matches(self):
        """Test with no matches."""
        predictions = [
            {'box': [100, 100, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_average_distance_error(predictions, ground_truth)
        
        assert result['num_matched'] == 0
        assert result['mean_distance_error'] == 0.0
    
    def test_bev_distance_type(self, matched_data):
        """Test with BEV distance type."""
        predictions, ground_truth = matched_data
        
        result = calculate_average_distance_error(
            predictions, ground_truth,
            distance_type="bev"
        )
        
        assert result['num_matched'] == 2
        assert result['mean_distance_error'] >= 0


class TestTranslationErrorBins:
    """Test translation error binning by distance ranges."""
    
    @pytest.fixture
    def range_data(self):
        """Sample data at different distances."""
        predictions = [
            # Near (5m)
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            # Medium (25m)
            {'box': [25, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'},
            # Far (45m)
            {'box': [45, 0, 0, 4, 2, 1.5, 0], 'score': 0.7, 'class': 'car'},
            # Very far - false positive (100m)
            {'box': [100, 0, 0, 4, 2, 1.5, 0], 'score': 0.6, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [25, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [45, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        return predictions, ground_truth
    
    def test_default_bins(self, range_data):
        """Test with default distance bins."""
        predictions, ground_truth = range_data
        
        result = calculate_translation_error_bins(
            predictions, ground_truth,
            iou_threshold=0.5
        )
        
        # Default bins: [0, 10, 30, 50, 100]
        assert '0-10m' in result
        assert '10-30m' in result
        assert '30-50m' in result
        assert '50-100m' in result
        
        # Check TP counts
        assert result['0-10m']['tp'] == 1  # 5m detection
        assert result['10-30m']['tp'] == 1  # 25m detection
        assert result['30-50m']['tp'] == 1  # 45m detection
        assert result['50-100m']['tp'] == 0  # No GT in this range
        # Note: 100m prediction is AT the bin boundary, so it's not counted in this bin
        assert result['50-100m']['fp'] == 0
    
    def test_custom_bins(self, range_data):
        """Test with custom distance bins."""
        predictions, ground_truth = range_data
        
        result = calculate_translation_error_bins(
            predictions, ground_truth,
            bins=[0, 20, 40, 100]
        )
        
        assert '0-20m' in result
        assert '20-40m' in result
        assert '40-100m' in result
        
        # 5m and 25m in different bins now
        assert result['0-20m']['tp'] >= 1
        assert result['20-40m']['tp'] >= 1
    
    def test_empty_predictions(self):
        """Test with empty predictions."""
        predictions = []
        ground_truth = [
            {'box': [5, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_translation_error_bins(predictions, ground_truth)
        
        # Should have FN in 0-10m bin
        assert result['0-10m']['fn'] == 1
        assert result['0-10m']['tp'] == 0
        assert result['0-10m']['fp'] == 0


class TestEdgeCases:
    """Test edge cases for distance metrics."""
    
    def test_very_large_distance(self):
        """Test with very large distances."""
        box1 = [0, 0, 0, 4, 2, 1.5, 0]
        box2 = [1000, 1000, 1000, 4, 2, 1.5, 0]
        
        dist = calculate_center_distance(box1, box2)
        assert dist > 1000
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        box1 = [-10, -10, -5, 4, 2, 1.5, 0]
        box2 = [10, 10, 5, 4, 2, 1.5, 0]
        
        dist = calculate_center_distance(box1, box2)
        expected = np.sqrt(20**2 + 20**2 + 10**2)
        assert np.isclose(dist, expected)
    
    def test_zero_size_boxes(self):
        """Test size error with very small boxes."""
        box1 = [0, 0, 0, 0.001, 0.001, 0.001, 0]
        box2 = [0, 0, 0, 1, 1, 1, 0]
        
        errors = calculate_size_error(box1, box2, error_type="absolute")
        assert errors['width_error'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
