"""
Tests for vector map detection metrics.
"""

import numpy as np
import pytest
from admetrics.vectormap import (
    chamfer_distance_polyline,
    frechet_distance,
    polyline_iou,
    lane_detection_metrics,
    topology_metrics,
    endpoint_error,
    direction_accuracy,
    vectormap_ap
)


class TestChamferDistancePolyline:
    """Test Chamfer distance for polylines."""
    
    def test_identical_polylines(self):
        """Test with identical polylines."""
        polyline = np.array([[0, 0], [1, 0], [2, 0]])
        result = chamfer_distance_polyline(polyline, polyline)
        
        assert result['chamfer_distance'] == pytest.approx(0.0, abs=1e-6)
        assert result['chamfer_pred_to_gt'] == pytest.approx(0.0, abs=1e-6)
        assert result['chamfer_gt_to_pred'] == pytest.approx(0.0, abs=1e-6)
    
    def test_parallel_polylines(self):
        """Test with parallel polylines."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[0, 1], [10, 1]])
        
        result = chamfer_distance_polyline(pred, gt)
        assert result['chamfer_distance'] == pytest.approx(1.0, abs=0.01)
    
    def test_offset_polylines(self):
        """Test with slightly offset polylines."""
        pred = np.array([[0, 0], [1, 0], [2, 0.1]])
        gt = np.array([[0, 0], [1, 0], [2, 0]])
        
        result = chamfer_distance_polyline(pred, gt)
        assert result['chamfer_distance'] < 0.1
    
    def test_empty_polylines(self):
        """Test with empty polylines."""
        pred = np.array([])
        gt = np.array([[0, 0], [1, 0]])
        
        result = chamfer_distance_polyline(pred.reshape(0, 2), gt)
        assert result['chamfer_distance'] == float('inf')
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
    
    def test_with_threshold(self):
        """Test precision/recall with distance threshold."""
        pred = np.array([[0, 0], [1, 0], [2, 0], [3, 5]])  # Last point is outlier
        gt = np.array([[0, 0], [1, 0], [2, 0]])
        
        result = chamfer_distance_polyline(pred, gt, max_distance=0.5)
        assert result['precision'] == pytest.approx(0.75, abs=0.01)  # 3/4 points match
        assert result['recall'] == pytest.approx(1.0, abs=0.01)  # All GT points match
    
    def test_3d_polylines(self):
        """Test with 3D polylines."""
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.1]])
        gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        
        result = chamfer_distance_polyline(pred, gt)
        assert result['chamfer_distance'] < 0.1


class TestFrechetDistance:
    """Test Fréchet distance."""
    
    def test_identical_polylines(self):
        """Test with identical polylines."""
        polyline = np.array([[0, 0], [1, 0], [2, 0]])
        dist = frechet_distance(polyline, polyline)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_parallel_polylines(self):
        """Test with parallel polylines."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[0, 1], [10, 1]])
        
        dist = frechet_distance(pred, gt)
        assert dist == pytest.approx(1.0, abs=0.01)
    
    def test_different_directions(self):
        """Test with polylines in different directions."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        gt = np.array([[2, 0], [1, 0], [0, 0]])  # Reversed
        
        dist = frechet_distance(pred, gt)
        assert dist >= 2.0  # Should be large due to reversed order
    
    def test_curved_polylines(self):
        """Test with curved polylines."""
        # Circle arc
        theta = np.linspace(0, np.pi/2, 10)
        pred = np.column_stack([np.cos(theta), np.sin(theta)])
        
        # Slightly larger circle arc
        gt = np.column_stack([1.1 * np.cos(theta), 1.1 * np.sin(theta)])
        
        dist = frechet_distance(pred, gt)
        assert 0.0 < dist < 0.2
    
    def test_empty_polylines(self):
        """Test with empty polylines."""
        pred = np.array([]).reshape(0, 2)
        gt = np.array([[0, 0], [1, 0]])
        
        dist = frechet_distance(pred, gt)
        assert dist == float('inf')


class TestPolylineIoU:
    """Test polyline IoU metric."""
    
    def test_identical_polylines(self):
        """Test with identical polylines."""
        polyline = np.array([[0, 0], [10, 0]])
        iou = polyline_iou(polyline, polyline, width=1.0)
        assert iou == pytest.approx(1.0, abs=0.1)
    
    def test_parallel_close_polylines(self):
        """Test with close parallel polylines."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[0, 0.5], [10, 0.5]])  # 0.5m apart
        
        iou = polyline_iou(pred, gt, width=1.0)
        assert iou > 0.5  # Should have good overlap
    
    def test_parallel_far_polylines(self):
        """Test with far parallel polylines."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[0, 5], [10, 5]])  # 5m apart
        
        iou = polyline_iou(pred, gt, width=1.0)
        assert iou < 0.1  # Should have minimal overlap
    
    def test_perpendicular_polylines(self):
        """Test with perpendicular polylines."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[5, -5], [5, 5]])
        
        iou = polyline_iou(pred, gt, width=1.0)
        assert 0.0 < iou < 0.5  # Some overlap at intersection
    
    def test_short_polylines(self):
        """Test with very short polylines."""
        pred = np.array([[0, 0], [0.1, 0]])
        gt = np.array([[0, 0], [0.1, 0]])
        
        iou = polyline_iou(pred, gt, width=1.0)
        assert iou > 0.5


class TestLaneDetectionMetrics:
    """Test lane detection metrics."""
    
    def test_perfect_detection(self):
        """Test with perfect lane detection."""
        pred = [
            np.array([[0, 0], [10, 0]]),
            np.array([[0, 3], [10, 3]])
        ]
        gt = [
            np.array([[0, 0.1], [10, 0.1]]),
            np.array([[0, 3.1], [10, 3.1]])
        ]
        
        metrics = lane_detection_metrics(pred, gt, distance_threshold=1.5)
        assert metrics['precision'] == pytest.approx(1.0, abs=0.01)
        assert metrics['recall'] == pytest.approx(1.0, abs=0.01)
        assert metrics['f1_score'] == pytest.approx(1.0, abs=0.01)
        assert metrics['tp'] == 2
        assert metrics['fp'] == 0
        assert metrics['fn'] == 0
    
    def test_missed_detection(self):
        """Test with missed lane detection."""
        pred = [
            np.array([[0, 0], [10, 0]])
        ]
        gt = [
            np.array([[0, 0.1], [10, 0.1]]),
            np.array([[0, 3], [10, 3]])
        ]
        
        metrics = lane_detection_metrics(pred, gt, distance_threshold=1.5)
        assert metrics['precision'] == pytest.approx(1.0, abs=0.01)
        assert metrics['recall'] == pytest.approx(0.5, abs=0.01)
        assert metrics['tp'] == 1
        assert metrics['fp'] == 0
        assert metrics['fn'] == 1
    
    def test_false_positive(self):
        """Test with false positive detection."""
        pred = [
            np.array([[0, 0], [10, 0]]),
            np.array([[0, 3], [10, 3]]),
            np.array([[0, 6], [10, 6]])  # False positive
        ]
        gt = [
            np.array([[0, 0.1], [10, 0.1]]),
            np.array([[0, 3.1], [10, 3.1]])
        ]
        
        metrics = lane_detection_metrics(pred, gt, distance_threshold=1.5)
        assert metrics['precision'] == pytest.approx(2/3, abs=0.01)
        assert metrics['recall'] == pytest.approx(1.0, abs=0.01)
        assert metrics['tp'] == 2
        assert metrics['fp'] == 1
        assert metrics['fn'] == 0
    
    def test_empty_predictions(self):
        """Test with no predictions."""
        pred = []
        gt = [np.array([[0, 0], [10, 0]])]
        
        metrics = lane_detection_metrics(pred, gt)
        assert metrics['precision'] == pytest.approx(1.0)
        assert metrics['recall'] == pytest.approx(0.0)
        assert metrics['tp'] == 0
        assert metrics['fn'] == 1
    
    def test_empty_ground_truth(self):
        """Test with no ground truth."""
        pred = [np.array([[0, 0], [10, 0]])]
        gt = []
        
        metrics = lane_detection_metrics(pred, gt)
        assert metrics['precision'] == pytest.approx(0.0)
        assert metrics['recall'] == pytest.approx(1.0)
        assert metrics['fp'] == 1
    
    def test_both_empty(self):
        """Test with both empty."""
        metrics = lane_detection_metrics([], [])
        assert metrics['precision'] == pytest.approx(1.0)
        assert metrics['recall'] == pytest.approx(1.0)
        assert metrics['f1_score'] == pytest.approx(1.0)


class TestTopologyMetrics:
    """Test topology metrics."""
    
    def test_perfect_topology(self):
        """Test with perfect topology prediction."""
        pred_topo = {
            'successors': [1, 2],
            'left_neighbor': [3]
        }
        gt_topo = {
            'successors': [1, 2],
            'left_neighbor': [3]
        }
        lane_matches = {0: 0, 1: 1, 2: 2, 3: 3}
        
        metrics = topology_metrics(pred_topo, gt_topo, lane_matches)
        assert metrics['topology_precision'] == pytest.approx(1.0)
        assert metrics['topology_recall'] == pytest.approx(1.0)
        assert metrics['topology_f1'] == pytest.approx(1.0)
    
    def test_partial_topology(self):
        """Test with partially correct topology."""
        pred_topo = {
            'successors': [1, 2],
            'left_neighbor': [3]
        }
        gt_topo = {
            'successors': [1, 2],
            'left_neighbor': [3, 4]  # Missing connection
        }
        lane_matches = {0: 0, 1: 1, 2: 2, 3: 3}
        
        metrics = topology_metrics(pred_topo, gt_topo, lane_matches)
        assert metrics['topology_precision'] == pytest.approx(1.0)  # All pred correct
        assert metrics['topology_recall'] == pytest.approx(3/4)  # 3/4 GT connections found
        assert metrics['correct_connections'] == 3
        assert metrics['pred_connections'] == 3
        assert metrics['gt_connections'] == 4
    
    def test_wrong_topology(self):
        """Test with incorrect topology."""
        pred_topo = {
            'successors': [5, 6]  # Wrong successors
        }
        gt_topo = {
            'successors': [1, 2]
        }
        lane_matches = {5: 99, 6: 98}  # Don't match GT
        
        metrics = topology_metrics(pred_topo, gt_topo, lane_matches)
        assert metrics['topology_precision'] == pytest.approx(0.0)
        assert metrics['topology_recall'] == pytest.approx(0.0)
    
    def test_empty_topology(self):
        """Test with empty topology."""
        metrics = topology_metrics({}, {}, {})
        assert metrics['topology_precision'] == pytest.approx(0.0)
        assert metrics['topology_recall'] == pytest.approx(0.0)


class TestEndpointError:
    """Test endpoint error metric."""
    
    def test_identical_endpoints(self):
        """Test with identical endpoints."""
        polyline = np.array([[0, 0], [5, 0], [10, 0]])
        errors = endpoint_error(polyline, polyline)
        
        assert errors['start_error'] == pytest.approx(0.0, abs=1e-6)
        assert errors['end_error'] == pytest.approx(0.0, abs=1e-6)
        assert errors['mean_endpoint_error'] == pytest.approx(0.0, abs=1e-6)
    
    def test_small_endpoint_errors(self):
        """Test with small endpoint errors."""
        pred = np.array([[0, 0], [5, 0], [10, 0]])
        gt = np.array([[0.1, 0], [5, 0], [10.1, 0]])
        
        errors = endpoint_error(pred, gt)
        assert errors['start_error'] == pytest.approx(0.1, abs=0.01)
        assert errors['end_error'] == pytest.approx(0.1, abs=0.01)
        assert errors['mean_endpoint_error'] == pytest.approx(0.1, abs=0.01)
    
    def test_large_endpoint_errors(self):
        """Test with large endpoint errors."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[0, 5], [10, 5]])
        
        errors = endpoint_error(pred, gt)
        assert errors['start_error'] == pytest.approx(5.0, abs=0.1)
        assert errors['end_error'] == pytest.approx(5.0, abs=0.1)
    
    def test_asymmetric_errors(self):
        """Test with different start and end errors."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[0, 1], [10, 3]])
        
        errors = endpoint_error(pred, gt)
        assert errors['start_error'] == pytest.approx(1.0, abs=0.1)
        assert errors['end_error'] == pytest.approx(3.0, abs=0.1)
        assert errors['mean_endpoint_error'] == pytest.approx(2.0, abs=0.1)
    
    def test_empty_polyline(self):
        """Test with empty polyline."""
        pred = np.array([]).reshape(0, 2)
        gt = np.array([[0, 0], [10, 0]])
        
        errors = endpoint_error(pred, gt)
        assert errors['start_error'] == float('inf')


class TestDirectionAccuracy:
    """Test direction accuracy metric."""
    
    def test_identical_directions(self):
        """Test with identical directions."""
        polyline = np.array([[0, 0], [10, 0]])
        acc = direction_accuracy(polyline, polyline)
        
        assert acc['mean_direction_error'] == pytest.approx(0.0, abs=0.01)
        assert acc['mean_direction_error_deg'] == pytest.approx(0.0, abs=0.01)
        assert acc['direction_accuracy'] == pytest.approx(1.0)
    
    def test_small_direction_error(self):
        """Test with small direction error."""
        pred = np.array([[0, 0], [10, 1]])  # Slight upward angle
        gt = np.array([[0, 0], [10, 0]])    # Horizontal
        
        acc = direction_accuracy(pred, gt)
        assert acc['mean_direction_error_deg'] < 10  # Should be small
        assert acc['direction_accuracy'] > 0.9
    
    def test_perpendicular_directions(self):
        """Test with perpendicular directions."""
        pred = np.array([[0, 0], [10, 0]])  # Horizontal
        gt = np.array([[0, 0], [0, 10]])    # Vertical
        
        acc = direction_accuracy(pred, gt)
        assert acc['mean_direction_error_deg'] == pytest.approx(90, abs=1)
        assert acc['direction_accuracy'] == pytest.approx(0.0)
    
    def test_opposite_directions(self):
        """Test with opposite directions."""
        pred = np.array([[0, 0], [10, 0]])
        gt = np.array([[10, 0], [0, 0]])  # Reversed
        
        acc = direction_accuracy(pred, gt)
        assert acc['mean_direction_error_deg'] == pytest.approx(180, abs=1)
        assert acc['direction_accuracy'] == pytest.approx(0.0)
    
    def test_curved_polyline(self):
        """Test with curved polyline."""
        # Quarter circle
        theta = np.linspace(0, np.pi/2, 20)
        pred = np.column_stack([np.cos(theta), np.sin(theta)])
        gt = np.column_stack([np.cos(theta), np.sin(theta)])
        
        acc = direction_accuracy(pred, gt)
        assert acc['mean_direction_error_deg'] < 5
        assert acc['direction_accuracy'] > 0.95
    
    def test_short_polyline(self):
        """Test with very short polyline."""
        pred = np.array([[0, 0]])  # Single point
        gt = np.array([[0, 0], [1, 0]])
        
        acc = direction_accuracy(pred, gt)
        assert acc['mean_direction_error'] == float('inf')


class TestVectormapAP:
    """Test vector map Average Precision."""
    
    def test_perfect_detection(self):
        """Test with perfect detection."""
        pred = [
            {'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9},
            {'polyline': np.array([[0, 3], [10, 3]]), 'score': 0.8}
        ]
        gt = [
            {'polyline': np.array([[0, 0.1], [10, 0.1]])},
            {'polyline': np.array([[0, 3.1], [10, 3.1]])}
        ]
        
        ap = vectormap_ap(pred, gt, distance_thresholds=[1.0])
        assert ap['ap_1.0'] > 0.9
        assert ap['map'] > 0.9
    
    def test_with_false_positives(self):
        """Test with false positives."""
        pred = [
            {'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9},
            {'polyline': np.array([[0, 3], [10, 3]]), 'score': 0.8},
            {'polyline': np.array([[0, 6], [10, 6]]), 'score': 0.7}  # FP
        ]
        gt = [
            {'polyline': np.array([[0, 0.1], [10, 0.1]])},
            {'polyline': np.array([[0, 3.1], [10, 3.1]])}
        ]
        
        ap = vectormap_ap(pred, gt, distance_thresholds=[1.0])
        assert 0.5 < ap['ap_1.0'] < 1.0  # Lower due to FP
    
    def test_with_false_negatives(self):
        """Test with false negatives."""
        pred = [
            {'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9}
        ]
        gt = [
            {'polyline': np.array([[0, 0.1], [10, 0.1]])},
            {'polyline': np.array([[0, 3], [10, 3]])}  # Missed
        ]
        
        ap = vectormap_ap(pred, gt, distance_thresholds=[1.0])
        assert 0.3 < ap['ap_1.0'] < 0.7  # Lower due to FN
    
    def test_multiple_thresholds(self):
        """Test with multiple distance thresholds."""
        pred = [
            {'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9}
        ]
        gt = [
            {'polyline': np.array([[0, 0.7], [10, 0.7]])}  # 0.7m offset
        ]
        
        ap = vectormap_ap(pred, gt, distance_thresholds=[0.5, 1.0, 1.5])
        assert ap['ap_0.5'] == pytest.approx(0.0)  # Too strict
        assert ap['ap_1.0'] > 0.9  # Should match
        assert ap['ap_1.5'] > 0.9  # Should match
        assert 'map' in ap
    
    def test_confidence_ranking(self):
        """Test that higher confidence predictions are prioritized."""
        pred = [
            {'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.5},  # Low conf
            {'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9}   # High conf, same location
        ]
        gt = [
            {'polyline': np.array([[0, 0.1], [10, 0.1]])}
        ]
        
        ap = vectormap_ap(pred, gt, distance_thresholds=[1.0])
        assert ap['ap_1.0'] > 0.5  # Should get partial credit
    
    def test_empty_predictions(self):
        """Test with no predictions."""
        pred = []
        gt = [{'polyline': np.array([[0, 0], [10, 0]])}]
        
        ap = vectormap_ap(pred, gt)
        assert ap['map'] == pytest.approx(0.0)
    
    def test_empty_ground_truth(self):
        """Test with no ground truth."""
        pred = [{'polyline': np.array([[0, 0], [10, 0]]), 'score': 0.9}]
        gt = []
        
        ap = vectormap_ap(pred, gt)
        # All predictions are false positives
        assert ap['map'] == pytest.approx(0.0)
    
    def test_external_scores(self):
        """Test with external confidence scores."""
        pred = [
            {'polyline': np.array([[0, 0], [10, 0]])},
            {'polyline': np.array([[0, 3], [10, 3]])}
        ]
        gt = [
            {'polyline': np.array([[0, 0.1], [10, 0.1]])},
            {'polyline': np.array([[0, 3.1], [10, 3.1]])}
        ]
        scores = [0.9, 0.8]
        
        ap = vectormap_ap(pred, gt, confidence_scores=scores)
        assert ap['map'] > 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
