"""
Tests for multi-object tracking metrics.
"""

import pytest
import numpy as np
from admetrics.tracking.tracking import (
    calculate_mota,
    calculate_motp,
    calculate_clearmot_metrics,
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_id_f1
)


class TestMOTA:
    """Test MOTA calculation for single frame."""
    
    def test_perfect_tracking(self):
        """Test MOTA with perfect detections."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['mota'] == 1.0
        assert result['tp'] == 1
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_with_false_positives(self):
        """Test MOTA with false positives."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_mota(predictions, ground_truth, iou_threshold=0.5)
        
        # MOTA = 1 - (FN + FP + IDSW) / GT = 1 - (0 + 1 + 0) / 1 = 0.0
        assert result['mota'] == 0.0
        assert result['tp'] == 1
        assert result['fp'] == 1
        assert result['fn'] == 0
    
    def test_with_false_negatives(self):
        """Test MOTA with missed detections."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_mota(predictions, ground_truth, iou_threshold=0.5)
        
        # MOTA = 1 - (FN + FP + IDSW) / GT = 1 - (1 + 0 + 0) / 1 = 0.0
        assert result['mota'] == 0.0
        assert result['tp'] == 0
        assert result['fp'] == 0
        assert result['fn'] == 1


class TestMOTP:
    """Test MOTP calculation."""
    
    def test_perfect_localization(self):
        """Test MOTP with perfect localization."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_motp(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['motp'] == 0.0  # Perfect match = 0 distance
        assert result['num_tp'] == 1
    
    def test_with_distance(self):
        """Test MOTP with some distance error."""
        predictions = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_motp(predictions, ground_truth, iou_threshold=0.5)
        
        # Distance should be 0.5 (x offset)
        assert 0.4 < result['motp'] < 0.6
        assert result['num_tp'] == 1


class TestClearMOT:
    """Test CLEAR MOT metrics (combined MOTA and MOTP)."""
    
    def test_clearmot_perfect(self):
        """Test CLEAR MOT with perfect tracking."""
        predictions = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'}
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_clearmot_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        # Should have both MOTA and MOTP
        assert 'mota' in result
        assert 'motp' in result
        assert 'tp' in result
        assert 'fp' in result
        assert 'fn' in result
        assert 'num_tp' in result
        
        # Perfect tracking
        assert result['mota'] == 1.0
        assert result['motp'] == 0.0  # Perfect localization = 0 distance
        assert result['tp'] == 1
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_clearmot_with_errors(self):
        """Test CLEAR MOT with tracking errors."""
        predictions = [
            {'box': [0.5, 0, 0, 4, 2, 1.5, 0], 'score': 0.9, 'class': 'car'},
            {'box': [100, 100, 0, 4, 2, 1.5, 0], 'score': 0.8, 'class': 'car'}  # FP
        ]
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}  # FN
        ]
        
        result = calculate_clearmot_metrics(predictions, ground_truth, iou_threshold=0.5)
        
        # Should have reduced MOTA due to FP and FN
        assert result['mota'] < 1.0
        assert result['tp'] == 1
        assert result['fp'] == 1
        assert result['fn'] == 1
        
        # MOTP should reflect localization error of TP
        assert result['motp'] > 0
        assert result['num_tp'] == 1
    
    def test_clearmot_no_detections(self):
        """Test CLEAR MOT with no detections."""
        predictions = []
        ground_truth = [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'class': 'car'}
        ]
        
        result = calculate_clearmot_metrics(predictions, ground_truth)
        
        assert result['mota'] == 0.0
        assert result['motp'] == 0.0
        assert result['tp'] == 0
        assert result['fn'] == 1


class TestMultiFrameMOTA:
    """Test multi-frame MOTA calculation."""
    
    @pytest.fixture
    def simple_sequence(self):
        """Simple 2-frame sequence with one object."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        return predictions, ground_truth
    
    def test_perfect_tracking_sequence(self, simple_sequence):
        """Test perfect tracking across frames."""
        predictions, ground_truth = simple_sequence
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['mota'] == 1.0
        assert result['num_matches'] == 2
        assert result['num_false_positives'] == 0
        assert result['num_misses'] == 0
        assert result['num_switches'] == 0
        assert result['mostly_tracked'] == 1
    
    def test_id_switch(self):
        """Test ID switch detection."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}]  # ID changed!
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['num_switches'] == 1
        assert result['mota'] < 1.0  # MOTA penalizes ID switches
    
    def test_fragmentation(self):
        """Test fragmentation detection."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [],  # Track lost
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]  # Track recovered
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['num_fragmentations'] == 1
        assert result['num_misses'] == 1  # Frame 1 was missed
    
    def test_mostly_tracked_ratio(self):
        """Test mostly tracked trajectory classification."""
        # Track detected in 4 out of 5 frames (80%)
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            2: [],  # Missed
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            4: [{'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            2: [{'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            3: [{'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            4: [{'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_multi_frame_mota(predictions, ground_truth, iou_threshold=0.5)
        
        # 4/5 = 80% >= 80% threshold
        assert result['mostly_tracked'] == 1
        assert result['partially_tracked'] == 0
        assert result['mostly_lost'] == 0


class TestHOTA:
    """Test HOTA metric."""
    
    def test_perfect_hota(self):
        """Test HOTA with perfect tracking."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_hota(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['hota'] > 0
        assert result['det_a'] == 1.0  # Perfect detection
        assert result['tp'] == 2
        assert result['fp'] == 0
        assert result['fn'] == 0
    
    def test_hota_with_errors(self):
        """Test HOTA with detection errors."""
        predictions = {
            0: [
                {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
                {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}  # FP
            ],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [
                {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
                {'box': [11, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'}  # FN
            ]
        }
        
        result = calculate_hota(predictions, ground_truth, iou_threshold=0.5)
        
        assert 0 <= result['hota'] <= 1
        assert result['det_a'] < 1.0  # Has errors
        assert result['tp'] == 2
        assert result['fp'] == 1
        assert result['fn'] == 1


class TestIDMetrics:
    """Test ID-based metrics (IDF1)."""
    
    def test_perfect_id_f1(self):
        """Test IDF1 with perfect ID consistency."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}]
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_id_f1(predictions, ground_truth, iou_threshold=0.5)
        
        assert result['idf1'] == 1.0
        assert result['idp'] == 1.0
        assert result['idr'] == 1.0
    
    def test_id_switch_penalty(self):
        """Test IDF1 penalty for ID switches."""
        predictions = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'}]  # ID switched
        }
        ground_truth = {
            0: [{'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}],
            1: [{'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'}]
        }
        
        result = calculate_id_f1(predictions, ground_truth, iou_threshold=0.5)
        
        # ID switch should reduce IDF1
        assert result['idf1'] < 1.0
        assert result['idfp'] > 0
        assert result['idfn'] > 0
