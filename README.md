# Metrics for Autonomous Driving

A comprehensive Python library for evaluating autonomous driving systems, including 3D object detection, multi-object tracking, trajectory prediction, and ego vehicle localization.

## Features

### Supported Metrics

#### Detection Metrics
- **3D IoU (Intersection over Union)**: 3D bounding box overlap calculation
- **BEV IoU**: Bird's Eye View IoU for 2D top-down evaluation
- **Average Precision (AP)**: Standard precision-recall based metrics
- **Mean Average Precision (mAP)**: Averaged AP across multiple classes
- **NuScenes Detection Score (NDS)**: NuScenes benchmark composite metric
- **Average Orientation Similarity (AOS)**: KITTI orientation-aware metric
- **True Positive (TP) / False Positive (FP) / False Negative (FN)**: Confusion metrics
- **Precision, Recall, F1-Score**: Standard classification metrics

#### Tracking Metrics
- **MOTA (Multiple Object Tracking Accuracy)**: Overall tracking accuracy with ID switches
- **MOTP (Multiple Object Tracking Precision)**: Average localization error
- **HOTA (Higher Order Tracking Accuracy)**: Balanced detection and association metric
- **IDF1 (ID F1 Score)**: Identity-based F1 score for tracking consistency
- **ID Switches**: Number of identity switches
- **Fragmentations**: Track interruption count
- **Trajectory Classification**: Mostly tracked, partially tracked, mostly lost

#### Trajectory Prediction Metrics
- **ADE (Average Displacement Error)**: Mean trajectory accuracy
- **FDE (Final Displacement Error)**: Endpoint prediction accuracy
- **Miss Rate**: Binary success metric at threshold
- **minADE/minFDE**: Best mode for multi-modal predictions
- **Brier-FDE**: Probability-weighted displacement error
- **NLL (Negative Log-Likelihood)**: Probabilistic prediction quality
- **Collision Rate**: Safety metric for obstacle avoidance
- **Drivable Area Compliance**: Trajectory feasibility metric

#### Localization Metrics
- **ATE (Absolute Trajectory Error)**: Position accuracy for ego vehicle localization
- **RTE (Relative Trajectory Error)**: Drift analysis over distance (e.g., 100m, 200m)
- **ARE (Absolute Rotation Error)**: Heading/orientation accuracy (yaw or quaternion)
- **Lateral Error**: Cross-track error for lane keeping assessment
- **Longitudinal Error**: Along-track positioning error
- **Convergence Rate**: Initialization and recovery speed
- **Map Alignment Score**: HD map lane centerline alignment quality

### Dataset Support

- KITTI format
- NuScenes format
- Waymo Open Dataset format
- Generic format (custom annotations)

## Installation

### From Source

```bash
git clone <repository-url>
cd metrics
pip install -e .
```

### Using pip

```bash
pip install .
```

## Quick Start

```python
from admetrics import evaluate_3d_detection
from admetrics.utils import load_predictions, load_ground_truth

# Load data
predictions = load_predictions('predictions.json')
ground_truth = load_ground_truth('ground_truth.json')

# Calculate metrics
results = evaluate_3d_detection(
    predictions=predictions,
    ground_truth=ground_truth,
    metrics=['ap', 'iou_3d', 'nds'],
    iou_threshold=0.5
)

print(results)
```

## Usage Examples

### Basic 3D IoU Calculation

```python
from admetrics.detection.iou import calculate_iou_3d

# Define 3D bounding boxes [x, y, z, w, h, l, yaw]
box1 = [0, 0, 0, 4, 2, 1.5, 0]
box2 = [1, 0, 0, 4, 2, 1.5, 0]

iou = calculate_iou_3d(box1, box2)
print(f"3D IoU: {iou:.4f}")
```

### Calculate Average Precision

```python
from admetrics.detection.ap import calculate_ap

results = calculate_ap(
    predictions=pred_boxes,
    ground_truth=gt_boxes,
    iou_threshold=0.7,
    num_recall_points=40
)

print(f"AP@0.7: {results['ap']:.4f}")
```

### NuScenes Detection Score

```python
from admetrics.detection.nds import calculate_nds

nds_score = calculate_nds(
    predictions=predictions,
    ground_truth=ground_truth,
    class_names=['car', 'pedestrian', 'cyclist']
)

print(f"NDS: {nds_score:.4f}")
```

### Multi-Object Tracking

```python
from admetrics.tracking import calculate_multi_frame_mota

# predictions: Dict[frame_id -> List[detection]]
# Each detection: {'box': [...], 'track_id': int, 'class': str}
results = calculate_multi_frame_mota(predictions, ground_truth)

print(f"MOTA: {results['mota']:.4f}")
print(f"ID Switches: {results['num_switches']}")
```

### Trajectory Prediction

```python
from admetrics.prediction import calculate_ade, calculate_fde

# Predicted trajectory: (T, 2) for x,y positions
# Ground truth: (T, 2)
ade = calculate_ade(predicted_trajectory, ground_truth_trajectory)
fde = calculate_fde(predicted_trajectory, ground_truth_trajectory)

print(f"ADE: {ade:.2f}m")
print(f"FDE: {fde:.2f}m")
```

### Ego Vehicle Localization

```python
from admetrics.localization import calculate_localization_metrics

# Evaluate ego pose estimation (GPS, SLAM, sensor fusion)
# Supports 3D (x,y,z), 4D (x,y,z,yaw), or 7D (x,y,z,qw,qx,qy,qz) poses
metrics = calculate_localization_metrics(
    predicted_poses,      # (N, 4) or (N, 7)
    ground_truth_poses,   # From RTK-GPS or ground truth
    timestamps=timestamps,
    lane_width=3.5,
    align=False  # Set True for SLAM drift analysis with Umeyama alignment
)

print(f"ATE: {metrics['ate_mean']:.3f}m")
print(f"Lateral Error: {metrics['lateral_mean']:.3f}m")
print(f"Heading Error: {metrics['are_mean']:.2f}°")
```

## Documentation

For detailed documentation, see:

- [API Reference](docs/api_reference.md) - Complete API documentation
- [Detection Metrics](docs/DETECTION_METRICS.md) - Detection metric explanations
- [Dataset Formats](docs/dataset_formats.md) - Supported data formats
- [Metrics Reference](docs/METRICS_REFERENCE.md) - Comprehensive list of all 40+ metrics
- [Tracking Metrics](docs/TRACKING_METRICS.md) - Multi-object tracking metrics guide
- [Trajectory Prediction](docs/TRAJECTORY_PREDICTION.md) - Motion forecasting metrics guide
- [Examples](examples/) - Working code examples

## Project Structure

```
metrics/
├── admetrics/              # Main package
│   ├── __init__.py
│   ├── detection/         # 3D object detection metrics
│   │   ├── __init__.py
│   │   ├── iou.py         # IoU calculations (3D, BEV, GIoU)
│   │   ├── ap.py          # Average Precision metrics
│   │   ├── nds.py         # NuScenes Detection Score
│   │   ├── aos.py         # Average Orientation Similarity
│   │   ├── confusion.py   # TP/FP/FN, Precision, Recall
│   │   └── distance.py    # Distance-based metrics
│   ├── tracking/          # Multi-object tracking metrics
│   │   ├── __init__.py
│   │   └── tracking.py    # MOTA, MOTP, HOTA, IDF1
│   ├── prediction/        # Trajectory prediction metrics
│   │   ├── __init__.py
│   │   └── trajectory.py  # ADE, FDE, NLL, safety metrics
│   ├── localization/      # Ego vehicle localization metrics
│   │   ├── __init__.py
│   │   └── localization.py # ATE, RTE, ARE, map alignment
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── transforms.py  # Coordinate transformations
│       ├── matching.py    # Detection matching algorithms
│       ├── nms.py         # Non-maximum suppression
│       └── visualization.py
├── tests/                  # Test suite (111 tests, 66% coverage)
│   ├── test_iou.py
│   ├── test_ap.py
│   ├── test_nds.py
│   ├── test_confusion.py
│   ├── test_tracking.py   # Tracking metrics tests
│   ├── test_trajectory.py # Trajectory prediction tests
│   ├── test_localization.py # Localization metrics tests
│   └── test_utils.py
├── examples/               # Example scripts
│   ├── basic_usage.py
│   ├── kitti_evaluation.py
│   ├── nuscenes_evaluation.py
│   ├── tracking_evaluation.py
│   ├── trajectory_prediction.py
│   └── localization_evaluation.py
├── docs/                   # Documentation
│   ├── api_reference.md
│   ├── DETECTION_METRICS.md
│   ├── dataset_formats.md
│   ├── METRICS_REFERENCE.md    # Complete metrics reference guide
│   ├── TRACKING_METRICS.md     # Tracking metrics guide
│   ├── TRAJECTORY_PREDICTION.md # Trajectory metrics guide
│   └── LOCALIZATION_METRICS.md  # Localization metrics guide
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0

Optional:
- matplotlib >= 3.3.0 (for visualization)
- open3d >= 0.13.0 (for 3D visualization)

## Testing

Run all tests:

```bash
pytest tests/
```

Run specific test modules:

```bash
pytest tests/test_localization.py -v
pytest tests/test_tracking.py -v
pytest tests/test_trajectory.py -v
```

For coverage report:

```bash
pytest --cov=admetrics tests/
```

Current test statistics: **111 tests, 66% coverage**

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{admetrics,
  title={Metrics for Autonomous Driving: Detection, Tracking, Prediction, and Localization},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/metrics}
}
```

## References

- KITTI Dataset: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- NuScenes Dataset: [https://www.nuscenes.org/](https://www.nuscenes.org/)
- Waymo Open Dataset: [https://waymo.com/open/](https://waymo.com/open/)
- Argoverse: [https://www.argoverse.org/](https://www.argoverse.org/)

## Acknowledgments

This library implements metrics based on evaluation protocols from:
- KITTI 3D Object Detection Benchmark
- KITTI Odometry Benchmark
- NuScenes Detection and Tracking Tasks
- Waymo Open Dataset Challenges
- Argoverse Motion Forecasting
