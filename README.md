# AD-Metrics: Autonomous Driving Evaluation Metrics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-304%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](tests/)

A comprehensive Python library for evaluating autonomous driving perception and planning systems across **125+ metrics** in 9 categories: detection, tracking, trajectory prediction, localization, occupancy, planning, vector maps, simulation quality, and utilities.

## 🚀 Features

### 📊 Complete Metric Coverage

| Category | Metrics | Key Features |
|----------|---------|--------------|
| **Detection** | 25 | IoU (3D/BEV/GIoU), AP, mAP, NDS, AOS, Distance/Orientation Errors |
| **Tracking** | 14 | MOTA, MOTP, HOTA, IDF1, ID Switches, Fragmentations |
| **Trajectory Prediction** | 11 | ADE, FDE, minADE/minFDE, NLL, Brier-FDE, Safety Metrics |
| **Localization** | 8 | ATE, RTE, ARE, Lateral/Longitudinal Errors, Map Alignment |
| **Occupancy** | 10 | mIoU, Scene Completion, Chamfer Distance, Precision/Recall |
| **Planning** | 12 | L2 Distance, Collision Rate, Driving Score, Comfort Metrics |
| **Vector Maps** | 8 | Chamfer/Fréchet Distance, Topology, Lane Detection, AP |
| **Simulation Quality** | 29 | Camera/LiDAR/Radar Quality, Noise, Temporal, Sim2Real Gap |
| **Utilities** | 8+ | Matching, NMS, Transforms, Visualization |

### 🎯 Benchmark Support

- **KITTI**: 3D Detection, Odometry, AOS
- **nuScenes**: Detection (NDS), Tracking, Occupancy, Maps
- **Waymo Open Dataset**: Detection, Tracking
- **Argoverse**: Trajectory Prediction, Map Detection
- **OpenLane-V2**: Topology, Lane Graph
- **nuPlan**: Planning, Driving Score
- **CARLA**: End-to-End Planning
- **SemanticKITTI/Occ3D**: Occupancy Prediction

## 📦 Installation

### From GitHub

```bash
git clone https://github.com/naurril/ad-metrics.git
cd ad-metrics
pip install -e .
```

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0

Optional:
- matplotlib >= 3.3.0 (visualization)
- open3d >= 0.13.0 (3D visualization)

## 🚀 Quick Start

### Detection Evaluation

```python
from admetrics.detection import calculate_ap, calculate_iou_3d, calculate_nds

# 3D IoU
box1 = [0, 0, 0, 4, 2, 1.5, 0]  # [x, y, z, w, h, l, yaw]
box2 = [1, 0, 0, 4, 2, 1.5, 0]
iou = calculate_iou_3d(box1, box2)

# Average Precision
ap_result = calculate_ap(predictions, ground_truth, iou_threshold=0.7)
print(f"AP@0.7: {ap_result['ap']:.4f}")

# NuScenes Detection Score
nds = calculate_nds(predictions, ground_truth, class_names=['car', 'pedestrian'])
print(f"NDS: {nds:.4f}")
```

### Tracking Evaluation

```python
from admetrics.tracking import calculate_multi_frame_mota, calculate_hota

# CLEAR MOT metrics
results = calculate_multi_frame_mota(predictions, ground_truth)
print(f"MOTA: {results['mota']:.4f}")
print(f"ID Switches: {results['num_switches']}")

# HOTA (balanced tracking)
hota = calculate_hota(predictions, ground_truth)
print(f"HOTA: {hota:.4f}")
```

### Trajectory Prediction

```python
from admetrics.prediction import calculate_ade, calculate_multimodal_ade

# Single-modal prediction
ade = calculate_ade(predicted_traj, ground_truth_traj)
fde = calculate_fde(predicted_traj, ground_truth_traj)

# Multi-modal prediction (K modes)
result = calculate_multimodal_ade(predicted_modes, ground_truth_traj)
print(f"minADE: {result['minADE']:.2f}m")
```

### Localization Evaluation

```python
from admetrics.localization import calculate_localization_metrics

# Comprehensive ego pose evaluation
# Supports 3D (x,y,z), 4D (x,y,z,yaw), or 7D (x,y,z,qw,qx,qy,qz)
metrics = calculate_localization_metrics(
    predicted_poses,      # (N, 4) or (N, 7)
    ground_truth_poses,
    timestamps=timestamps,
    lane_width=3.5,
    align=False  # Set True for SLAM drift analysis
)

print(f"ATE: {metrics['ate_mean']:.3f}m")
print(f"Lateral Error: {metrics['lateral_mean']:.3f}m")
print(f"Heading Error: {metrics['are_mean']:.2f}°")
```

### Occupancy Prediction

```python
from admetrics.occupancy import mean_iou, scene_completion

# Voxel-based metrics
miou = mean_iou(pred_occupancy, gt_occupancy, num_classes=18)
print(f"mIoU: {miou:.4f}")

# Scene completion
sc_result = scene_completion(pred_occupancy, gt_occupancy)
print(f"SC-IoU: {sc_result['sc_iou']:.4f}")
```

### Vector Map Detection

```python
from admetrics.vectormap import chamfer_distance_polyline, topology_metrics

# Geometric accuracy
cd = chamfer_distance_polyline(pred_polylines, gt_polylines)
print(f"Chamfer Distance: {cd['chamfer_distance']:.3f}m")

# Topology evaluation
topo = topology_metrics(pred_graph, gt_graph)
print(f"Successor Accuracy: {topo['successor_accuracy']:.4f}")
```

### Planning Evaluation

```python
from admetrics.planning import driving_score, collision_rate

# nuPlan-style comprehensive evaluation
score = driving_score(
    predicted_trajectory,
    expert_trajectory,
    obstacles,
    route
)
print(f"Driving Score: {score:.2f}/100")

# Safety metrics
collision = collision_rate(predicted_trajectory, obstacles)
print(f"Collision Rate: {collision:.2%}")
```

## 📚 Documentation

- **[Quick Reference](docs/METRICS_REFERENCE.md)** - All metrics at a glance
- **[Detection Metrics](docs/DETECTION_METRICS.md)** - IoU, AP, NDS, AOS details
- **[Tracking Metrics](docs/TRACKING_METRICS.md)** - MOTA, HOTA, IDF1 guide
- **[Trajectory Prediction](docs/TRAJECTORY_PREDICTION.md)** - ADE, FDE, probabilistic metrics
- **[Localization Metrics](docs/LOCALIZATION_METRICS.md)** - Ego pose evaluation
- **[Occupancy Metrics](docs/OCCUPANCY_METRICS.md)** - Voxel-based metrics
- **[Planning Metrics](docs/END_TO_END_METRICS.md)** - Driving evaluation
- **[Vector Map Metrics](docs/VECTORMAP_METRICS.md)** - HD map detection
- **[Simulation Quality](docs/SIMULATION_QUALITY.md)** - Sensor fidelity
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Dataset Formats](docs/dataset_formats.md)** - KITTI, nuScenes, Waymo
- **[Examples](examples/)** - Working code examples

## 📁 Project Structure

```
ad-metrics/
├── admetrics/              # Main package
│   ├── detection/         # 3D detection (25 metrics, 94% coverage)
│   ├── tracking/          # MOT (14 metrics, 89% coverage)
│   ├── prediction/        # Trajectory (11 metrics, 95% coverage)
│   ├── localization/      # Ego pose (8 metrics, 91% coverage)
│   ├── occupancy/         # Voxel grids (10 metrics, 98% coverage)
│   ├── planning/          # End-to-end (12 metrics, 95% coverage)
│   ├── vectormap/         # HD maps (8 metrics, 98% coverage)
│   ├── simulation/        # Sensor quality (29 metrics, 66% coverage)
│   └── utils/             # Matching, NMS, transforms
├── tests/                  # 304 tests, 80% coverage
├── examples/               # 10 example scripts
├── docs/                   # Comprehensive documentation
└── README.md
```

## ✅ Testing

**Current Status: 304 passing tests, 2 skipped, 80% code coverage**

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=admetrics tests/
```

Run specific categories:
```bash
pytest tests/test_detection/ -v
pytest tests/test_tracking.py -v
pytest tests/test_localization.py -v
```

### Test Coverage by Module

| Module | Tests | Coverage |
|--------|-------|----------|
| `detection/iou.py` | 14 | 91% |
| `detection/ap.py` | 9 | 98% |
| `detection/nds.py` | 5 | 96% |
| `detection/aos.py` | 18 | 96% |
| `detection/distance.py` | 31 | 94% |
| `detection/confusion.py` | 5 | 90% |
| `tracking/tracking.py` | 16 | 89% |
| `prediction/trajectory.py` | 26 | 95% |
| `localization/localization.py` | 27 | 91% |
| `occupancy/occupancy.py` | 39 | 98% |
| `planning/planning.py` | 42 | 95% |
| `vectormap/vectormap.py` | 45 | 98% |
| `simulation/sensor_quality.py` | 17 | 66% |
| `utils/*` | 12 | 50% |
| **Total** | **304** | **80%** |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-metric`)
3. Add tests for new metrics
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📖 Citation

If you use this library in your research, please cite:

```bibtex
@software{admetrics2025,
  title={AD-Metrics: Comprehensive Evaluation Metrics for Autonomous Driving},
  author={Contributors},
  year={2025},
  url={https://github.com/naurril/ad-metrics}
}
```

## 🙏 Acknowledgments

This library implements metrics based on evaluation protocols from:

- **KITTI 3D Object Detection & Odometry Benchmarks**
- **nuScenes Detection, Tracking & Occupancy Tasks**
- **Waymo Open Dataset Challenges**
- **Argoverse Motion Forecasting & HD Map Detection**
- **OpenLane-V2 Topology Challenge**
- **nuPlan Planning Challenge**
- **CARLA Autonomous Driving Challenge**

## 🔗 References

- KITTI: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- nuScenes: [https://www.nuscenes.org/](https://www.nuscenes.org/)
- Waymo: [https://waymo.com/open/](https://waymo.com/open/)
- Argoverse: [https://www.argoverse.org/](https://www.argoverse.org/)
- OpenLane-V2: [https://openlanev2.github.io/](https://openlanev2.github.io/)
- nuPlan: [https://www.nuscenes.org/nuplan](https://www.nuscenes.org/nuplan)

---

**Made with ❤️ for the autonomous driving community**


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
