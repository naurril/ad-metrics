"""
Tests for Average Orientation Similarity (AOS) metrics.
"""

import pytest
import numpy as np
from admetrics.detection.aos import (
    calculate_aos,
    calculate_orientation_similarity,
    calculate_aos_per_difficulty,
    calculate_orientation_error
)


class TestOrientationSimilarity:
    """Test orientation similarity calculation."""
    
    def test_identical_orientation(self):
        """Test similarity with identical orientations."""
        sim = calculate_orientation_similarity(0.0, 0.0)
        assert np.isclose(sim, 1.0), f"Expected 1.0, got {sim}"
    
    def test_opposite_orientation(self):
        """Test similarity with opposite orientations."""
        sim = calculate_orientation_similarity(0.0, np.pi)
        assert np.isclose(sim, 0.0, atol=0.01), f"Expected 0.0, got {sim}"
    
    def test_perpendicular_orientation(self):
        """Test similarity with perpendicular orientations."""
        sim = calculate_orientation_similarity(0.0, np.pi/2)
        # cos(90°) = 0, so similarity = (1 + 0) / 2 = 0.5
        assert np.isclose(sim, 0.5, atol=0.01), f"Expected 0.5, got {sim}"
    
    def test_small_difference(self):
        """Test similarity with small orientation difference."""
        sim = calculate_orientation_similarity(0.1, 0.0)
        # cos(0.1) ≈ 0.995, so similarity ≈ 0.9975
        assert sim > 0.99, f"Small difference should give high similarity, got {sim}"
    
    def test_angle_wrapping(self):
        """Test similarity handles angle wrapping."""
        sim1 = calculate_orientation_similarity(0.0, 2*np.pi)
        sim2 = calculate_orientation_similarity(-np.pi, np.pi)
        
        # 0 and 2π are the same angle
        assert np.isclose(sim1, 1.0, atol=0.01)
        # -π and π are the same angle
        assert np.isclose(sim2, 1.0, atol=0.01)


class TestOrientationError:
    """Test orientation error calculation."""
    
    def test_zero_error(self):
        """Test zero orientation error."""
        error = calculate_orientation_error(0.0, 0.0)
        assert np.isclose(error, 0.0), f"Expected 0.0, got {error}"
    
    def test_small_error(self):
        """Test small orientation error."""
        error = calculate_orientation_error(0.1, 0.0)
        assert np.isclose(error, 0.1, atol=0.01), f"Expected 0.1, got {error}"
    
    def test_pi_error(self):
        """Test maximum orientation error."""
        error = calculate_orientation_error(0.0, np.pi)
        assert np.isclose(error, np.pi, atol=0.01), f"Expected π, got {error}"
    
    def test_angle_normalization(self):
        """Test error handles angle normalization."""
        # 2π and 0 are the same
        error1 = calculate_orientation_error(0.0, 2*np.pi)
        assert np.isclose(error1, 0.0, atol=0.01)
        
        # -π/2 and 3π/2 are the same
        error2 = calculate_orientation_error(-np.pi/2, 3*np.pi/2)
        assert np.isclose(error2, 0.0, atol=0.01)


class TestAOS:
    """Test Average Orientation Similarity."""
    
    @pytest.fixture
    def perfect_data(self):
        """Perfect predictions with correct orientations."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.1], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, -0.2], 'score': 0.8, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.1], 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, -0.2], 'class': 'car'}
        ]
        return predictions, ground_truth
    
    @pytest.fixture
    def wrong_orientation_data(self):
        """Good detections but wrong orientations."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, np.pi], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'class': 'car'}
        ]
        return predictions, ground_truth
    
    def test_perfect_aos(self, perfect_data):
        """Test AOS with perfect orientations."""
        predictions, ground_truth = perfect_data
        
        result = calculate_aos(predictions, ground_truth, iou_threshold=0.5)
        
        # Perfect detection and orientation should give high AOS
        assert result['aos'] > 0.99, f"Expected AOS > 0.99, got {result['aos']}"
        assert result['ap'] > 0.99, f"Expected AP > 0.99, got {result['ap']}"
        assert result['orientation_similarity'] > 0.99
        assert result['num_tp'] == 2
        assert result['num_fp'] == 0
    
    def test_wrong_orientation(self, wrong_orientation_data):
        """Test AOS penalizes wrong orientations."""
        predictions, ground_truth = wrong_orientation_data
        
        result = calculate_aos(predictions, ground_truth, iou_threshold=0.5)
        
        # AP should be high (detection is correct)
        assert result['ap'] > 0.99
        
        # AOS should be low (orientation is wrong)
        # With opposite orientation, similarity ≈ 0
        assert result['aos'] < 0.1, f"Expected low AOS, got {result['aos']}"
        assert result['orientation_similarity'] < 0.1
    
    def test_no_detections(self):
        """Test AOS with no detections."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_aos(predictions, ground_truth)
        
        assert result['aos'] == 0.0
        assert result['ap'] == 0.0
        assert result['num_tp'] == 0
    
    def test_mixed_orientations(self):
        """Test AOS with mix of good and bad orientations."""
        predictions = [
            # Perfect orientation
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'score': 0.9, 'class': 'car'},
            # Wrong orientation (180 degrees off)
            {'box': [5, 0, 0, 4, 2, 1.5, np.pi], 'score': 0.8, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'class': 'car'},
            {'box': [5, 0, 0, 4, 2, 1.5, 0.0], 'class': 'car'}
        ]
        
        result = calculate_aos(predictions, ground_truth, iou_threshold=0.5)
        
        # AP should be high (both detections correct)
        assert result['ap'] > 0.99
        
        # AOS should be between 0 and AP (one good, one bad orientation)
        # Mean orientation similarity should be around 0.5
        assert 0.4 < result['orientation_similarity'] < 0.6
        assert 0 < result['aos'] < result['ap']


class TestAOSPerDifficulty:
    """Test AOS calculation by difficulty level."""
    
    def test_by_difficulty_easy(self):
        """Test AOS for easy difficulty."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.1], 'score': 0.9, 'class': 'Car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.1], 'class': 'Car', 'difficulty': 'easy'}
        ]
        
        result = calculate_aos_per_difficulty(
            predictions, ground_truth, 
            iou_threshold=0.7
        )
        
        assert 'easy' in result
        assert result['easy']['aos'] > 0.99
        assert result['easy']['ap'] > 0.99
    
    def test_multiple_difficulties(self):
        """Test AOS for multiple difficulty levels."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.1], 'score': 0.9, 'class': 'Car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0.2], 'score': 0.8, 'class': 'Car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.1], 'class': 'Car', 'difficulty': 'easy'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0.2], 'class': 'Car', 'difficulty': 'moderate'}
        ]
        
        result = calculate_aos_per_difficulty(predictions, ground_truth, iou_threshold=0.5)
        
        assert 'easy' in result
        assert 'moderate' in result
        assert 'hard' in result
        
        # Each difficulty level is evaluated independently
        # Easy should match with first prediction
        assert result['easy']['aos'] > 0.99
        # Moderate could match with either prediction (whichever has higher score)
        # But since predictions are sorted by score, the first one is used for moderate GT too
        # This gives an AP of 0.5 (1 TP, 1 FP for 1 GT)
        assert 0 < result['moderate']['aos'] <= 0.5
        
        # Hard should have 0 AOS (no GT in that category)
        assert result['hard']['aos'] == 0.0


class TestAOSEdgeCases:
    """Test edge cases for AOS."""
    
    def test_false_positives(self):
        """Test AOS with false positives."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'score': 0.9, 'class': 'car'},
            {'box': [100, 100, 0, 4, 2, 1.5, 0.0], 'score': 0.8, 'class': 'car'}  # FP
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'class': 'car'}
        ]
        
        result = calculate_aos(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['num_tp'] == 1
        assert result['num_fp'] == 1
        # FP should reduce AOS
        assert result['aos'] < 1.0
    
    def test_no_ground_truth(self):
        """Test AOS with no ground truth."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = []
        
        result = calculate_aos(predictions, ground_truth)
        
        assert result['aos'] == 0.0
        assert result['ap'] == 0.0
        assert result['num_gt'] == 0
    
    def test_class_filtering(self):
        """Test AOS filters by class."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'score': 0.9, 'class': 'car'},
            {'box': [5, 0, 0, 2, 1, 1, 0.0], 'score': 0.8, 'class': 'pedestrian'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0.0], 'class': 'car'},
            {'box': [5, 0, 0, 2, 1, 1, 0.0], 'class': 'car'}  # Different class
        ]
        
        result = calculate_aos(predictions, ground_truth, iou_threshold=0.5)
        
        # Only the car prediction should match car GT
        assert result['num_tp'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
