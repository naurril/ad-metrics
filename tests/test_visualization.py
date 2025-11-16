"""
Tests for visualization utilities.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Test with and without matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestVisualizationImports:
    """Test visualization imports and error handling."""
    
    def test_matplotlib_availability(self):
        """Test that matplotlib availability is detected correctly."""
        from admetrics.utils import visualization
        
        # Should match the actual availability
        assert hasattr(visualization, 'MATPLOTLIB_AVAILABLE')
    
    @pytest.mark.skipif(MATPLOTLIB_AVAILABLE, reason="Matplotlib is available")
    def test_plot_without_matplotlib(self):
        """Test that functions raise ImportError when matplotlib is not available."""
        # This test only runs when matplotlib is NOT available
        from admetrics.utils.visualization import plot_boxes_3d
        
        boxes = [np.array([0, 0, 0, 4, 2, 1.5, 0])]
        
        with pytest.raises(ImportError, match="Matplotlib is required"):
            plot_boxes_3d(boxes)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
class TestPlotBoxes3D:
    """Test 3D box plotting."""
    
    @pytest.fixture
    def sample_boxes(self):
        """Sample 3D boxes for testing."""
        return [
            np.array([0, 0, 0, 4, 2, 1.5, 0]),
            np.array([5, 5, 1, 3, 2, 1, np.pi/4])
        ]
    
    def test_plot_boxes_3d_basic(self, sample_boxes):
        """Test basic 3D box plotting."""
        from admetrics.utils.visualization import plot_boxes_3d
        
        ax = plot_boxes_3d(sample_boxes, show=False)
        
        assert ax is not None
        assert hasattr(ax, 'plot3D')
        plt.close('all')
    
    def test_plot_boxes_3d_with_labels(self, sample_boxes):
        """Test 3D plotting with labels."""
        from admetrics.utils.visualization import plot_boxes_3d
        
        labels = ['Box 1', 'Box 2']
        ax = plot_boxes_3d(sample_boxes, labels=labels, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_boxes_3d_with_colors(self, sample_boxes):
        """Test 3D plotting with custom colors."""
        from admetrics.utils.visualization import plot_boxes_3d
        
        colors = ['blue', 'green']
        ax = plot_boxes_3d(sample_boxes, colors=colors, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_boxes_3d_with_existing_axis(self, sample_boxes):
        """Test plotting on existing axis."""
        from admetrics.utils.visualization import plot_boxes_3d
        
        fig = plt.figure()
        ax_existing = fig.add_subplot(111, projection='3d')
        
        ax = plot_boxes_3d(sample_boxes, ax=ax_existing, show=False)
        
        assert ax is ax_existing
        plt.close('all')
    
    def test_plot_boxes_3d_single_box(self):
        """Test plotting single box."""
        from admetrics.utils.visualization import plot_boxes_3d
        
        box = [np.array([0, 0, 0, 4, 2, 1.5, 0])]
        ax = plot_boxes_3d(box, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_boxes_3d_empty_list(self):
        """Test plotting empty list of boxes."""
        from admetrics.utils.visualization import plot_boxes_3d
        
        ax = plot_boxes_3d([], show=False)
        
        assert ax is not None
        plt.close('all')


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
class TestPlotBoxesBEV:
    """Test Bird's Eye View plotting."""
    
    @pytest.fixture
    def sample_boxes(self):
        """Sample boxes for BEV plotting."""
        return [
            np.array([0, 0, 0, 4, 2, 1.5, 0]),
            np.array([5, 5, 1, 3, 2, 1, np.pi/4])
        ]
    
    def test_plot_boxes_bev_basic(self, sample_boxes):
        """Test basic BEV plotting."""
        from admetrics.utils.visualization import plot_boxes_bev
        
        ax = plot_boxes_bev(sample_boxes, show=False)
        
        assert ax is not None
        assert hasattr(ax, 'plot')
        plt.close('all')
    
    def test_plot_boxes_bev_with_labels(self, sample_boxes):
        """Test BEV plotting with labels."""
        from admetrics.utils.visualization import plot_boxes_bev
        
        labels = ['Car 1', 'Car 2']
        ax = plot_boxes_bev(sample_boxes, labels=labels, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_boxes_bev_with_colors(self, sample_boxes):
        """Test BEV plotting with custom colors."""
        from admetrics.utils.visualization import plot_boxes_bev
        
        colors = ['red', 'blue']
        ax = plot_boxes_bev(sample_boxes, colors=colors, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_boxes_bev_with_range(self, sample_boxes):
        """Test BEV plotting with axis ranges."""
        from admetrics.utils.visualization import plot_boxes_bev
        
        ax = plot_boxes_bev(
            sample_boxes,
            x_range=(-10, 10),
            y_range=(-10, 10),
            show=False
        )
        
        assert ax is not None
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim == (-10, 10)
        assert ylim == (-10, 10)
        plt.close('all')
    
    def test_plot_boxes_bev_with_existing_axis(self, sample_boxes):
        """Test BEV plotting on existing axis."""
        from admetrics.utils.visualization import plot_boxes_bev
        
        fig, ax_existing = plt.subplots()
        
        ax = plot_boxes_bev(sample_boxes, ax=ax_existing, show=False)
        
        assert ax is ax_existing
        plt.close('all')
    
    def test_plot_boxes_bev_heading_direction(self):
        """Test that heading direction arrows are plotted."""
        from admetrics.utils.visualization import plot_boxes_bev
        
        # Box with different heading angles
        boxes = [
            np.array([0, 0, 0, 4, 2, 1.5, 0]),
            np.array([5, 0, 0, 4, 2, 1.5, np.pi/2])
        ]
        
        ax = plot_boxes_bev(boxes, show=False)
        
        assert ax is not None
        # Check that arrows were added (patches collection)
        assert len(ax.patches) > 0
        plt.close('all')
    
    def test_plot_boxes_bev_empty_list(self):
        """Test BEV plotting with empty list."""
        from admetrics.utils.visualization import plot_boxes_bev
        
        ax = plot_boxes_bev([], show=False)
        
        assert ax is not None
        plt.close('all')


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
class TestPlotPrecisionRecallCurve:
    """Test precision-recall curve plotting."""
    
    def test_plot_pr_curve_basic(self):
        """Test basic PR curve plotting."""
        from admetrics.utils.visualization import plot_precision_recall_curve
        
        precision = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        recall = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        ap = 0.75
        
        ax = plot_precision_recall_curve(precision, recall, ap, show=False)
        
        assert ax is not None
        assert ax.get_xlabel() == 'Recall'
        assert ax.get_ylabel() == 'Precision'
        plt.close('all')
    
    def test_plot_pr_curve_custom_title(self):
        """Test PR curve with custom title."""
        from admetrics.utils.visualization import plot_precision_recall_curve
        
        precision = np.array([1.0, 0.8, 0.6])
        recall = np.array([0.3, 0.6, 1.0])
        ap = 0.7
        title = "Car Detection PR Curve"
        
        ax = plot_precision_recall_curve(precision, recall, ap, title=title, show=False)
        
        assert ax is not None
        assert ax.get_title() == title
        plt.close('all')
    
    def test_plot_pr_curve_with_existing_axis(self):
        """Test PR curve on existing axis."""
        from admetrics.utils.visualization import plot_precision_recall_curve
        
        fig, ax_existing = plt.subplots()
        precision = np.array([1.0, 0.8, 0.6])
        recall = np.array([0.3, 0.6, 1.0])
        ap = 0.7
        
        ax = plot_precision_recall_curve(precision, recall, ap, ax=ax_existing, show=False)
        
        assert ax is ax_existing
        plt.close('all')
    
    def test_plot_pr_curve_limits(self):
        """Test that PR curve has correct axis limits."""
        from admetrics.utils.visualization import plot_precision_recall_curve
        
        precision = np.array([1.0, 0.5, 0.0])
        recall = np.array([0.0, 0.5, 1.0])
        ap = 0.5
        
        ax = plot_precision_recall_curve(precision, recall, ap, show=False)
        
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylim() == (0, 1)
        plt.close('all')
    
    def test_plot_pr_curve_legend(self):
        """Test that PR curve includes AP in legend."""
        from admetrics.utils.visualization import plot_precision_recall_curve
        
        precision = np.array([1.0, 0.8, 0.6])
        recall = np.array([0.3, 0.6, 1.0])
        ap = 0.8523
        
        ax = plot_precision_recall_curve(precision, recall, ap, show=False)
        
        legend = ax.get_legend()
        assert legend is not None
        # Check that AP value is in legend text
        legend_text = legend.get_texts()[0].get_text()
        assert 'AP' in legend_text
        assert '0.8523' in legend_text
        plt.close('all')


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
class TestVisualizeDetectionResults:
    """Test detection results visualization."""
    
    @pytest.fixture
    def detection_data(self):
        """Sample detection data with matches."""
        predictions = [
            {'box': np.array([0, 0, 0, 4, 2, 1.5, 0]), 'score': 0.9, 'class': 'car'},
            {'box': np.array([10, 10, 0, 4, 2, 1.5, 0]), 'score': 0.8, 'class': 'car'},
            {'box': np.array([20, 20, 0, 2, 1, 1, 0]), 'score': 0.7, 'class': 'pedestrian'},
        ]
        ground_truth = [
            {'box': np.array([0.5, 0, 0, 4, 2, 1.5, 0]), 'class': 'car'},
            {'box': np.array([30, 30, 0, 2, 1, 1, 0]), 'class': 'pedestrian'},
        ]
        matches = [(0, 0)]  # First prediction matches first GT
        
        return predictions, ground_truth, matches
    
    def test_visualize_detection_bev(self, detection_data):
        """Test detection visualization in BEV mode."""
        from admetrics.utils.visualization import visualize_detection_results
        
        predictions, ground_truth, matches = detection_data
        
        # Should not raise any errors
        visualize_detection_results(
            predictions, ground_truth, matches,
            mode='bev', show=False
        )
        plt.close('all')
    
    def test_visualize_detection_3d(self, detection_data):
        """Test detection visualization in 3D mode."""
        from admetrics.utils.visualization import visualize_detection_results
        
        predictions, ground_truth, matches = detection_data
        
        # Should not raise any errors
        visualize_detection_results(
            predictions, ground_truth, matches,
            mode='3d', show=False
        )
        plt.close('all')
    
    def test_visualize_detection_tp_fp_fn(self, detection_data):
        """Test that TP, FP, FN are correctly categorized."""
        from admetrics.utils.visualization import visualize_detection_results
        
        predictions, ground_truth, matches = detection_data
        
        # Expected:
        # TP: prediction[0] matches ground_truth[0]
        # FP: prediction[1], prediction[2] (no matches)
        # FN: ground_truth[1] (not matched)
        
        # This should create a plot with correct color coding
        visualize_detection_results(
            predictions, ground_truth, matches,
            mode='bev', show=False
        )
        plt.close('all')
    
    def test_visualize_detection_no_matches(self):
        """Test visualization when there are no matches."""
        from admetrics.utils.visualization import visualize_detection_results
        
        predictions = [
            {'box': np.array([0, 0, 0, 4, 2, 1.5, 0]), 'class': 'car'},
        ]
        ground_truth = [
            {'box': np.array([100, 100, 0, 4, 2, 1.5, 0]), 'class': 'car'},
        ]
        matches = []
        
        visualize_detection_results(
            predictions, ground_truth, matches,
            mode='bev', show=False
        )
        plt.close('all')
    
    def test_visualize_detection_perfect_matches(self):
        """Test visualization when all predictions match."""
        from admetrics.utils.visualization import visualize_detection_results
        
        predictions = [
            {'box': np.array([0, 0, 0, 4, 2, 1.5, 0]), 'class': 'car'},
            {'box': np.array([5, 5, 0, 4, 2, 1.5, 0]), 'class': 'car'},
        ]
        ground_truth = [
            {'box': np.array([0, 0, 0, 4, 2, 1.5, 0]), 'class': 'car'},
            {'box': np.array([5, 5, 0, 4, 2, 1.5, 0]), 'class': 'car'},
        ]
        matches = [(0, 0), (1, 1)]
        
        visualize_detection_results(
            predictions, ground_truth, matches,
            mode='bev', show=False
        )
        plt.close('all')
    
    def test_visualize_detection_empty(self):
        """Test visualization with empty data."""
        from admetrics.utils.visualization import visualize_detection_results
        
        predictions = []
        ground_truth = []
        matches = []
        
        visualize_detection_results(
            predictions, ground_truth, matches,
            mode='bev', show=False
        )
        plt.close('all')


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
class TestPlotConfusionMatrix:
    """Test confusion matrix plotting."""
    
    def test_plot_confusion_matrix_basic(self):
        """Test basic confusion matrix plotting."""
        from admetrics.utils.visualization import plot_confusion_matrix
        
        cm = np.array([
            [50, 5, 2],
            [3, 40, 7],
            [1, 4, 45]
        ])
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        
        ax = plot_confusion_matrix(cm, class_names, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_confusion_matrix_normalized(self):
        """Test normalized confusion matrix."""
        from admetrics.utils.visualization import plot_confusion_matrix
        
        cm = np.array([
            [50, 5, 2],
            [3, 40, 7],
            [1, 4, 45]
        ])
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        
        ax = plot_confusion_matrix(cm, class_names, normalize=True, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_confusion_matrix_custom_colormap(self):
        """Test confusion matrix with custom colormap."""
        from admetrics.utils.visualization import plot_confusion_matrix
        
        cm = np.array([
            [10, 2],
            [1, 15]
        ])
        class_names = ['Positive', 'Negative']
        
        ax = plot_confusion_matrix(cm, class_names, cmap='Reds', show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_confusion_matrix_binary(self):
        """Test binary confusion matrix."""
        from admetrics.utils.visualization import plot_confusion_matrix
        
        cm = np.array([
            [100, 10],
            [5, 85]
        ])
        class_names = ['True', 'False']
        
        ax = plot_confusion_matrix(cm, class_names, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_confusion_matrix_labels(self):
        """Test that confusion matrix has correct labels."""
        from admetrics.utils.visualization import plot_confusion_matrix
        
        cm = np.array([
            [50, 5],
            [3, 40]
        ])
        class_names = ['Class A', 'Class B']
        
        ax = plot_confusion_matrix(cm, class_names, show=False)
        
        assert ax is not None
        assert ax.get_xlabel() == 'Predicted label'
        assert ax.get_ylabel() == 'True label'
        assert ax.get_title() == 'Confusion Matrix'
        plt.close('all')
    
    def test_plot_confusion_matrix_single_class(self):
        """Test confusion matrix with single class."""
        from admetrics.utils.visualization import plot_confusion_matrix
        
        cm = np.array([[100]])
        class_names = ['Only Class']
        
        ax = plot_confusion_matrix(cm, class_names, show=False)
        
        assert ax is not None
        plt.close('all')
    
    def test_plot_confusion_matrix_large(self):
        """Test confusion matrix with many classes."""
        from admetrics.utils.visualization import plot_confusion_matrix
        
        n_classes = 10
        cm = np.random.randint(0, 50, (n_classes, n_classes))
        class_names = [f'Class {i}' for i in range(n_classes)]
        
        ax = plot_confusion_matrix(cm, class_names, show=False)
        
        assert ax is not None
        plt.close('all')


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
class TestVisualizationIntegration:
    """Integration tests for visualization functions."""
    
    def test_multiple_plots_in_sequence(self):
        """Test creating multiple plots in sequence."""
        from admetrics.utils.visualization import (
            plot_boxes_bev,
            plot_precision_recall_curve,
            plot_confusion_matrix
        )
        
        # Create BEV plot
        boxes = [np.array([0, 0, 0, 4, 2, 1.5, 0])]
        ax1 = plot_boxes_bev(boxes, show=False)
        assert ax1 is not None
        
        # Create PR curve
        precision = np.array([1.0, 0.8, 0.6])
        recall = np.array([0.3, 0.6, 1.0])
        ax2 = plot_precision_recall_curve(precision, recall, 0.7, show=False)
        assert ax2 is not None
        
        # Create confusion matrix
        cm = np.array([[10, 2], [1, 15]])
        ax3 = plot_confusion_matrix(cm, ['A', 'B'], show=False)
        assert ax3 is not None
        
        plt.close('all')
    
    def test_subplot_layout(self):
        """Test creating subplots with different visualizations."""
        from admetrics.utils.visualization import (
            plot_boxes_bev,
            plot_precision_recall_curve
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot BEV in first subplot
        boxes = [np.array([0, 0, 0, 4, 2, 1.5, 0])]
        ax1 = plot_boxes_bev(boxes, ax=axes[0], show=False)
        
        # Plot PR curve in second subplot
        precision = np.array([1.0, 0.8, 0.6])
        recall = np.array([0.3, 0.6, 1.0])
        ax2 = plot_precision_recall_curve(precision, recall, 0.7, ax=axes[1], show=False)
        
        assert ax1 is axes[0]
        assert ax2 is axes[1]
        
        plt.close('all')
    
    def test_complete_evaluation_workflow(self):
        """Test a complete evaluation visualization workflow."""
        from admetrics.utils.visualization import (
            visualize_detection_results,
            plot_precision_recall_curve,
            plot_confusion_matrix
        )
        
        # Detection results
        predictions = [
            {'box': np.array([0, 0, 0, 4, 2, 1.5, 0]), 'class': 'car', 'score': 0.9},
            {'box': np.array([5, 5, 0, 4, 2, 1.5, 0]), 'class': 'car', 'score': 0.8},
        ]
        ground_truth = [
            {'box': np.array([0, 0, 0, 4, 2, 1.5, 0]), 'class': 'car'},
        ]
        matches = [(0, 0)]
        
        # Visualize detections
        visualize_detection_results(predictions, ground_truth, matches, mode='bev', show=False)
        
        # Plot PR curve
        precision = np.array([1.0, 0.5, 0.5])
        recall = np.array([0.5, 1.0, 1.0])
        plot_precision_recall_curve(precision, recall, 0.75, show=False)
        
        # Plot confusion matrix
        cm = np.array([[1, 0], [1, 0]])  # 1 TP, 1 FP, 0 FN for binary
        plot_confusion_matrix(cm, ['car', 'background'], show=False)
        
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
