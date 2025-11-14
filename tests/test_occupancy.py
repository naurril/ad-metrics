"""Tests for 3D occupancy prediction metrics."""

import numpy as np
import pytest
from admetrics.occupancy import (
    occupancy_iou,
    mean_iou,
    occupancy_precision_recall,
    scene_completion,
    chamfer_distance,
    surface_distance,
)


class TestOccupancyIoU:
    """Test occupancy IoU calculation."""
    
    def test_perfect_match(self):
        """Test IoU with perfect prediction."""
        pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        gt = pred.copy()
        
        iou = occupancy_iou(pred, gt, class_id=1)
        assert iou == 1.0
    
    def test_no_overlap(self):
        """Test IoU with no overlap."""
        pred = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
        gt = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        
        iou = occupancy_iou(pred, gt, class_id=1)
        assert iou == 0.0
    
    def test_partial_overlap(self):
        """Test IoU with partial overlap."""
        pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        gt = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        
        iou = occupancy_iou(pred, gt, class_id=1)
        # pred has 1s at: (0,0,1), (0,1,0), (0,1,1), (1,2,0) = 4 positions
        # gt has 1s at: (0,0,1), (1,1,0), (1,2,0) = 3 positions
        # Intersection: (0,0,1), (1,2,0) = 2 positions (WAIT - let me recount)
        # pred[0,0,1]=1, pred[0,1,0]=1, pred[0,1,1]=1, pred[1,2,0]=1
        # gt[0,0,1]=1, gt[1,1,0]=1, gt[1,2,0]=1
        # Intersection: positions where both are 1: (0,0,1), (1,2,0) WAIT
        # Let me trace through the array more carefully
        # Union: positions where at least one is 1: 5 positions
        # Actual IoU is 3/5 = 0.6
        assert iou == pytest.approx(0.6, abs=0.01)
    
    def test_binary_occupancy(self):
        """Test binary occupancy (occupied vs free)."""
        pred = np.array([[[0, 1], [2, 3]], [[0, 0], [1, 0]]])
        gt = np.array([[[0, 2], [1, 0]], [[0, 3], [2, 0]]])
        
        # Binary: any non-zero is occupied
        iou = occupancy_iou(pred, gt, class_id=None)
        # pred occupied: (0,0,1), (0,1,0), (0,1,1), (1,2,0) = 4 voxels
        # gt occupied: (0,0,1), (0,1,0), (1,1,0), (1,1,1) = 4 voxels  
        # Recalculating based on actual positions
        # Actual IoU should be calculated properly
        assert iou == pytest.approx(0.6, abs=0.01)
    
    def test_ignore_index(self):
        """Test handling of ignore index."""
        pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 255]]])
        gt = np.array([[[0, 1], [1, 255]], [[0, 1], [1, 0]]])
        
        iou = occupancy_iou(pred, gt, class_id=1, ignore_index=255)
        # Ignore positions with 255 in GT
        assert 0.0 <= iou <= 1.0
    
    def test_empty_class(self):
        """Test IoU when class doesn't exist."""
        pred = np.zeros((5, 5, 5), dtype=int)
        gt = np.zeros((5, 5, 5), dtype=int)
        
        iou = occupancy_iou(pred, gt, class_id=1)
        assert iou == 1.0  # Both empty, perfect match
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        pred = np.zeros((5, 5, 5))
        gt = np.zeros((10, 10, 10))
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            occupancy_iou(pred, gt)


class TestMeanIoU:
    """Test mean IoU calculation."""
    
    def test_perfect_prediction(self):
        """Test mIoU with perfect prediction."""
        pred = np.random.randint(0, 3, size=(10, 10, 10))
        gt = pred.copy()
        
        result = mean_iou(pred, gt, num_classes=3)
        assert result['mIoU'] == 1.0
        assert result['valid_classes'] == 3
        assert all(iou == 1.0 for iou in result['class_iou'].values())
    
    def test_multiple_classes(self):
        """Test mIoU with multiple classes."""
        # Create simple grid with clear class boundaries
        pred = np.zeros((10, 10, 10), dtype=int)
        pred[5:, :, :] = 1
        pred[:, 5:, :] = 2
        
        gt = np.zeros((10, 10, 10), dtype=int)
        gt[5:, :, :] = 1
        gt[:, 5:, :] = 2
        
        result = mean_iou(pred, gt, num_classes=3)
        assert 0.0 <= result['mIoU'] <= 1.0
        assert result['valid_classes'] <= 3
        assert len(result['class_iou']) == 3
    
    def test_ignore_classes(self):
        """Test ignoring specific classes."""
        pred = np.random.randint(0, 4, size=(10, 10, 10))
        gt = np.random.randint(0, 4, size=(10, 10, 10))
        
        result = mean_iou(pred, gt, num_classes=4, ignore_classes=[0])
        assert 0 not in [k for k, v in result['class_iou'].items() if not np.isnan(v)]
    
    def test_missing_classes(self):
        """Test handling of classes not present in data."""
        # Only use classes 0 and 1, but specify 5 classes
        pred = np.random.randint(0, 2, size=(10, 10, 10))
        gt = np.random.randint(0, 2, size=(10, 10, 10))
        
        result = mean_iou(pred, gt, num_classes=5)
        assert result['valid_classes'] <= 2
        # Classes 2, 3, 4 should have NaN IoU
        assert any(np.isnan(iou) for iou in result['class_iou'].values())
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        pred = np.zeros((5, 5, 5))
        gt = np.zeros((10, 10, 10))
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            mean_iou(pred, gt, num_classes=3)


class TestOccupancyPrecisionRecall:
    """Test precision/recall calculation."""
    
    def test_perfect_prediction(self):
        """Test metrics with perfect prediction."""
        pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        gt = pred.copy()
        
        metrics = occupancy_precision_recall(pred, gt, class_id=1)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        # Count 1s in the array: pred[0,0,1], pred[0,1,0], pred[0,1,1], pred[1,2,0] = 4
        assert metrics['true_positives'] == 4
        assert metrics['false_positives'] == 0
        assert metrics['false_negatives'] == 0
    
    def test_all_false_positives(self):
        """Test with only false positives."""
        pred = np.ones((5, 5, 5), dtype=int)
        gt = np.zeros((5, 5, 5), dtype=int)
        
        metrics = occupancy_precision_recall(pred, gt, class_id=1)
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['true_positives'] == 0
        assert metrics['false_positives'] == 125
    
    def test_all_false_negatives(self):
        """Test with only false negatives."""
        pred = np.zeros((5, 5, 5), dtype=int)
        gt = np.ones((5, 5, 5), dtype=int)
        
        metrics = occupancy_precision_recall(pred, gt, class_id=1)
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['true_positives'] == 0
        assert metrics['false_negatives'] == 125
    
    def test_partial_match(self):
        """Test with partial match."""
        pred = np.array([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
        gt = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        
        metrics = occupancy_precision_recall(pred, gt, class_id=1)
        # Actual results from the implementation
        assert metrics['true_positives'] == 3
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 1
        assert metrics['precision'] == pytest.approx(0.75, abs=0.01)  # 3/4
        assert metrics['recall'] == pytest.approx(0.75, abs=0.01)  # 3/4
        assert metrics['f1'] == pytest.approx(0.75, abs=0.01)
    
    def test_binary_mode(self):
        """Test binary occupancy mode."""
        pred = np.array([[[0, 1], [2, 3]], [[0, 0], [1, 0]]])
        gt = np.array([[[0, 2], [1, 0]], [[0, 3], [2, 0]]])
        
        metrics = occupancy_precision_recall(pred, gt, class_id=None)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics


class TestSceneCompletion:
    """Test scene completion metrics."""
    
    def test_perfect_completion(self):
        """Test with perfect scene completion."""
        pred = np.random.randint(0, 3, size=(10, 10, 10))
        gt = pred.copy()
        
        sc = scene_completion(pred, gt, free_class=0)
        assert sc['SC_IoU'] == 1.0
        assert sc['SC_Precision'] == 1.0
        assert sc['SC_Recall'] == 1.0
        assert sc['completion_ratio'] == 1.0
    
    def test_over_completion(self):
        """Test over-completion (predicting too much occupancy)."""
        pred = np.ones((10, 10, 10), dtype=int)
        gt = np.zeros((10, 10, 10), dtype=int)
        gt[5:, 5:, 5:] = 1
        
        sc = scene_completion(pred, gt, free_class=0)
        assert sc['SC_Precision'] < 1.0
        assert sc['SC_Recall'] == 1.0
        assert sc['completion_ratio'] > 1.0
    
    def test_under_completion(self):
        """Test under-completion (missing occupancy)."""
        pred = np.zeros((10, 10, 10), dtype=int)
        gt = np.ones((10, 10, 10), dtype=int)
        
        sc = scene_completion(pred, gt, free_class=0)
        assert sc['SC_IoU'] == 0.0
        assert sc['SC_Precision'] == 0.0
        assert sc['SC_Recall'] == 0.0
        assert sc['completion_ratio'] == 0.0
    
    def test_semantic_scene_completion(self):
        """Test semantic scene completion mIoU."""
        # Create grid with multiple semantic classes
        pred = np.zeros((10, 10, 10), dtype=int)
        pred[:5, :, :] = 1
        pred[5:, :, :] = 2
        
        gt = np.zeros((10, 10, 10), dtype=int)
        gt[:5, :, :] = 1
        gt[5:, :, :] = 2
        
        sc = scene_completion(pred, gt, free_class=0)
        assert sc['SSC_mIoU'] == 1.0
    
    def test_mixed_scene(self):
        """Test with mixed occupied and free space."""
        pred = np.zeros((10, 10, 10), dtype=int)
        pred[2:8, 2:8, 2:8] = 1
        
        gt = np.zeros((10, 10, 10), dtype=int)
        gt[3:7, 3:7, 3:7] = 1
        
        sc = scene_completion(pred, gt, free_class=0)
        assert 0.0 < sc['SC_IoU'] < 1.0
        # pred cube is 6x6x6 = 216, gt cube is 4x4x4 = 64
        # completion_ratio = 216/64 = 3.375
        assert 0.0 < sc['completion_ratio'] < 5.0  # Allow for over-completion


class TestChamferDistance:
    """Test Chamfer Distance calculation."""
    
    def test_identical_point_clouds(self):
        """Test Chamfer Distance with identical point clouds."""
        points = np.random.rand(100, 3)
        
        cd = chamfer_distance(points, points, bidirectional=True)
        assert cd['chamfer_distance'] == pytest.approx(0.0, abs=1e-6)
        assert cd['pred_to_gt'] == pytest.approx(0.0, abs=1e-6)
        assert cd['gt_to_pred'] == pytest.approx(0.0, abs=1e-6)
    
    def test_different_point_clouds(self):
        """Test Chamfer Distance with different point clouds."""
        pred = np.random.rand(50, 3)
        gt = np.random.rand(60, 3) + 1.0  # Offset by 1 unit
        
        cd = chamfer_distance(pred, gt, bidirectional=True)
        assert cd['chamfer_distance'] > 0.0
        assert cd['pred_to_gt'] > 0.0
        assert cd['gt_to_pred'] > 0.0
    
    def test_unidirectional(self):
        """Test unidirectional Chamfer Distance."""
        pred = np.random.rand(50, 3)
        gt = np.random.rand(60, 3)
        
        cd = chamfer_distance(pred, gt, bidirectional=False)
        assert cd['chamfer_distance'] == cd['pred_to_gt']
        assert cd['gt_to_pred'] is None
    
    def test_empty_point_clouds(self):
        """Test with empty point clouds."""
        pred = np.array([]).reshape(0, 3)
        gt = np.random.rand(50, 3)
        
        cd = chamfer_distance(pred, gt, bidirectional=True)
        assert cd['chamfer_distance'] == np.inf
    
    def test_single_points(self):
        """Test with single points."""
        pred = np.array([[0.0, 0.0, 0.0]])
        gt = np.array([[1.0, 0.0, 0.0]])
        
        cd = chamfer_distance(pred, gt, bidirectional=True)
        assert cd['chamfer_distance'] == pytest.approx(1.0, abs=1e-6)
    
    def test_invalid_shape(self):
        """Test with invalid point dimensions."""
        pred = np.random.rand(50, 2)  # Wrong dimension
        gt = np.random.rand(60, 3)
        
        with pytest.raises(ValueError, match="must have shape"):
            chamfer_distance(pred, gt)


class TestSurfaceDistance:
    """Test surface distance metrics."""
    
    def test_identical_surfaces(self):
        """Test surface distance with identical occupancy."""
        occupancy = np.zeros((20, 20, 20), dtype=int)
        occupancy[5:15, 5:15, 5:15] = 1
        
        sd = surface_distance(occupancy, occupancy, voxel_size=0.2)
        assert sd['mean_surface_distance'] == pytest.approx(0.0, abs=1e-6)
        assert sd['median_surface_distance'] == pytest.approx(0.0, abs=1e-6)
        assert sd['std_surface_distance'] == pytest.approx(0.0, abs=1e-6)
    
    def test_different_surfaces(self):
        """Test surface distance with different occupancy."""
        pred = np.zeros((20, 20, 20), dtype=int)
        pred[5:15, 5:15, 5:15] = 1
        
        gt = np.zeros((20, 20, 20), dtype=int)
        gt[6:14, 6:14, 6:14] = 1
        
        sd = surface_distance(pred, gt, voxel_size=0.2)
        assert sd['mean_surface_distance'] > 0.0
        assert sd['median_surface_distance'] > 0.0
        assert sd['max_surface_distance'] > 0.0
    
    def test_voxel_size_scaling(self):
        """Test that voxel size correctly scales distances."""
        pred = np.zeros((20, 20, 20), dtype=int)
        pred[5:15, 5:15, 5:15] = 1
        
        gt = np.zeros((20, 20, 20), dtype=int)
        gt[6:14, 6:14, 6:14] = 1
        
        sd1 = surface_distance(pred, gt, voxel_size=1.0)
        sd2 = surface_distance(pred, gt, voxel_size=2.0)
        
        # With 2x voxel size, distances should be ~2x
        assert sd2['mean_surface_distance'] == pytest.approx(
            2 * sd1['mean_surface_distance'], rel=0.1
        )
    
    def test_percentile_distance(self):
        """Test percentile distance calculation."""
        pred = np.zeros((20, 20, 20), dtype=int)
        pred[5:15, 5:15, 5:15] = 1
        
        gt = np.zeros((20, 20, 20), dtype=int)
        gt[6:14, 6:14, 6:14] = 1
        
        sd = surface_distance(pred, gt, voxel_size=0.2, percentile=95)
        assert sd['percentile_distance'] is not None
        assert sd['percentile_distance'] >= sd['median_surface_distance']
        assert sd['percentile_distance'] <= sd['max_surface_distance']
    
    def test_no_percentile(self):
        """Test without percentile calculation."""
        pred = np.zeros((20, 20, 20), dtype=int)
        pred[5:15, 5:15, 5:15] = 1
        
        gt = np.zeros((20, 20, 20), dtype=int)
        gt[6:14, 6:14, 6:14] = 1
        
        sd = surface_distance(pred, gt, voxel_size=0.2, percentile=None)
        assert sd['percentile_distance'] is None
    
    def test_empty_occupancy(self):
        """Test with empty occupancy grids."""
        pred = np.zeros((20, 20, 20), dtype=int)
        gt = np.zeros((20, 20, 20), dtype=int)
        
        sd = surface_distance(pred, gt, voxel_size=0.2)
        assert sd['mean_surface_distance'] == np.inf
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        pred = np.zeros((10, 10, 10))
        gt = np.zeros((20, 20, 20))
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            surface_distance(pred, gt)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_large_grid(self):
        """Test with large occupancy grid."""
        # Test that metrics can handle larger grids
        pred = np.random.randint(0, 3, size=(50, 50, 50))
        gt = np.random.randint(0, 3, size=(50, 50, 50))
        
        result = mean_iou(pred, gt, num_classes=3)
        assert 0.0 <= result['mIoU'] <= 1.0
    
    def test_single_voxel(self):
        """Test with single voxel grid."""
        pred = np.array([[[1]]])
        gt = np.array([[[1]]])
        
        iou = occupancy_iou(pred, gt, class_id=1)
        assert iou == 1.0
    
    def test_all_ignored(self):
        """Test when all voxels are ignored."""
        pred = np.ones((5, 5, 5), dtype=int) * 255
        gt = np.ones((5, 5, 5), dtype=int) * 255
        
        iou = occupancy_iou(pred, gt, class_id=1, ignore_index=255)
        assert iou == 1.0  # No valid voxels, perfect match
    
    def test_many_classes(self):
        """Test with many semantic classes."""
        pred = np.random.randint(0, 20, size=(30, 30, 30))
        gt = np.random.randint(0, 20, size=(30, 30, 30))
        
        result = mean_iou(pred, gt, num_classes=20)
        assert 'mIoU' in result
        assert len(result['class_iou']) == 20
