# AD-Metrics: Autonomous Driving Evaluation Metrics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-92%25-green.svg)](tests/)

A comprehensive Python library for evaluating autonomous driving perception and planning systems across **124 metric functions** (260+ individual metrics) in 9 categories: detection, tracking, trajectory prediction, localization, occupancy, planning, vector maps, simulation quality, and utilities.

## 🚀 Features

### 📊 Complete Metric Coverage

| Category | Functions | Metric Outputs | Key Features |
|----------|-----------|----------------|--------------|
| **Detection** | 24 | 40+ metrics | IoU (3D/BEV/GIoU), AP, mAP, NDS, AOS, Confusion, Distance Errors |
| **Tracking** | 21 | 50+ metrics | MOTA, MOTP, HOTA, IDF1, AMOTA, TID/LGD, MOTAL, CLR, OWTA |
| **Trajectory Prediction** | 10 | 10+ metrics | ADE, FDE, minADE/minFDE, NLL, Brier-FDE, Collision Rate |
| **Localization** | 8 | 25+ metrics | ATE, RTE, ARE, Lateral/Longitudinal Errors, Drift Analysis |
| **Occupancy** | 9 | 25+ metrics | mIoU, Precision/Recall, Chamfer Distance, Scene Completion, VPQ |
| **Planning** | 20 | 50+ metrics | L2 Distance, Collision, Progress, Comfort, Driving Score, Safety |
| **Vector Maps** | 12 | 20+ metrics | Chamfer/Fréchet Distance, Topology, Lane Detection, 3D Metrics |
| **Simulation Quality** | 11 | 40+ metrics | Camera/LiDAR/Radar Quality, Noise, Temporal, Sim2Real, Weather |
| **Utilities** | 9 | - | Matching (Greedy/Hungarian), NMS, Transforms, Visualization |
| **TOTAL** | **124** | **260+** | Comprehensive autonomous driving evaluation metrics |

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

## 📚 Documentation

### API Reference

Build the comprehensive API documentation locally:

```bash
cd docs
./build_docs.sh
```

Then open `docs/_build/html/index.html` in your browser.

The documentation includes:
- **Automatic API Reference**: Generated from code docstrings for all 89 metric functions (165+ metrics)
- **Conceptual Guides**: Detailed explanations of each metric category
- **Usage Examples**: Code examples for every function
- **Type Hints**: Full type annotation documentation
- **Search**: Full-text search across all documentation

See [`docs/README_DOCS.md`](docs/README_DOCS.md) for complete documentation building instructions.

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
from admetrics.tracking import (
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_amota,
    calculate_tid_lgd,
    calculate_id_f1
)

# CLEAR MOT metrics
results = calculate_multi_frame_mota(predictions, ground_truth)
print(f"MOTA: {results['mota']:.4f}")
print(f"ID Switches: {results['num_switches']}")

# HOTA (balanced tracking)
hota_results = calculate_hota(predictions, ground_truth)
print(f"HOTA: {hota_results['hota']:.4f}")

# nuScenes tracking metrics
amota_results = calculate_amota(predictions, ground_truth, recall_thresholds=[0.2, 0.4, 0.6, 0.8])
print(f"AMOTA: {amota_results['amota']:.4f}")

# Track initialization and gap metrics
tid_lgd = calculate_tid_lgd(predictions, ground_truth)
print(f"TID: {tid_lgd['tid']:.2f}%, LGD: {tid_lgd['lgd']:.2f}%")

# Identity preservation
idf1_results = calculate_id_f1(predictions, ground_truth)
print(f"IDF1: {idf1_results['idf1']:.4f}")
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
from admetrics.occupancy import calculate_mean_iou, calculate_scene_completion

# Voxel-based metrics
miou = calculate_mean_iou(pred_occupancy, gt_occupancy, num_classes=18)
print(f"mIoU: {miou:.4f}")

# Scene completion
sc_result = calculate_scene_completion(pred_occupancy, gt_occupancy)
print(f"SC-IoU: {sc_result['sc_iou']:.4f}")
```

### Vector Map Detection

```python
from admetrics.vectormap import calculate_chamfer_distance_polyline, calculate_topology_metrics

# Geometric accuracy
cd = calculate_chamfer_distance_polyline(pred_polylines, gt_polylines)
print(f"Chamfer Distance: {cd['calculate_chamfer_distance']:.3f}m")

# Topology evaluation
topo = calculate_topology_metrics(pred_graph, gt_graph)
print(f"Successor Accuracy: {topo['successor_accuracy']:.4f}")
```

### Planning Evaluation

```python
from admetrics.planning import calculate_driving_score, calculate_collision_rate

# nuPlan-style comprehensive evaluation
score = calculate_driving_score(
    predicted_trajectory,
    expert_trajectory,
    obstacles,
    route
)
print(f"Driving Score: {score:.2f}/100")

# Safety metrics
collision = calculate_collision_rate(predicted_trajectory, obstacles)
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
- **[API Interface Reference](docs/API_INTERFACE_REFERENCE.md)** - Complete API documentation (124 functions)
- **[API Reference](docs/api_reference.md)** - Auto-generated API docs
- **[Dataset Formats](docs/dataset_formats.md)** - KITTI, nuScenes, Waymo
- **[Examples](examples/)** - Working code examples

## 📁 Project Structure

```
ad-metrics/
├── admetrics/              # Main package
│   ├── detection/         # 3D detection (24 functions, 94% coverage)
│   ├── tracking/          # MOT (21 functions, 92% coverage)
│   ├── prediction/        # Trajectory (10 functions, 95% coverage)
│   ├── localization/      # Ego pose (8 functions, 91% coverage)
│   ├── occupancy/         # Voxel grids (9 functions, 98% coverage)
│   ├── planning/          # End-to-end (20 functions, 95% coverage)
│   ├── vectormap/         # HD maps (12 functions, 98% coverage)
│   ├── simulation/        # Sensor quality (11 functions, 91% coverage)
│   └── utils/             # Matching, NMS, transforms (9 functions)
├── tests/                  # 45 tracking tests, 92% coverage
├── examples/               # 10 example scripts
├── docs/                   # Comprehensive documentation
└── README.md
```

## ✅ Testing

**Current Status: 45 tracking tests passing, 92% code coverage on tracking module, 100% metric function coverage**

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

| Module | Tests | Functions | Coverage |
|--------|-------|-----------|----------|
| `detection/iou.py` | 14 | 4 | 91% |
| `detection/ap.py` | 9 | 4 | 98% |
| `detection/nds.py` | 5 | 3 | 96% |
| `detection/aos.py` | 18 | 4 | 96% |
| `detection/distance.py` | 31 | 6 | 94% |
| `detection/confusion.py` | 8 | 4 | 96% |
| `tracking/tracking.py` | 45 | 21 | 92% |
| `prediction/trajectory.py` | 26 | 10 | 95% |
| `localization/localization.py` | 27 | 8 | 91% |
| `occupancy/occupancy.py` | 39 | 6 | 98% |
| `planning/planning.py` | 42 | 11 | 95% |
| `vectormap/vectormap.py` | 45 | 8 | 98% |
| `simulation/sensor_quality.py` | 20 | 7 | 91% |
| `utils/visualization.py` | 35 | - | 91% |
| `utils/matching.py` | 6 | - | 74% |
| `utils/nms.py` | 8 | - | 88% |
| `utils/transforms.py` | 11 | - | 95% |
| **Total** | **45+** | **124** | **92%** |

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📖 Citation

If you use this library in your research, please cite:

```bibtex
@software{admetrics2025,
  title={AD-Metrics: Comprehensive Evaluation Metrics for Autonomous Driving},
  author={},
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

