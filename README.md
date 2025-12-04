# AD-Metrics: Autonomous Driving Evaluation Metrics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-92%25-green.svg)](tests/)

A comprehensive Python library for evaluating autonomous driving perception and planning systems across **124 metric functions** (260+ individual metrics) in 9 categories: detection, tracking, trajectory prediction, localization, occupancy, planning, vector maps, simulation quality, and utilities.

## Features

### Complete Metric Coverage

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

### Benchmark Support

- **KITTI**: 3D Detection, Odometry, AOS
- **nuScenes**: Detection (NDS), Tracking, Occupancy, Maps
- **Waymo Open Dataset**: Detection, Tracking
- **Argoverse**: Trajectory Prediction, Map Detection
- **OpenLane-V2**: Topology, Lane Graph
- **nuPlan**: Planning, Driving Score
- **CARLA**: End-to-End Planning
- **SemanticKITTI/Occ3D**: Occupancy Prediction

## Installation

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
- pytest

Optional:
- matplotlib >= 3.3.0 (visualization)
- open3d >= 0.13.0 (3D visualization)


## Quick Start

- [Quick Start](./docs/quick_start.md)

## Documentation


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



## Testing

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

## Acknowledgments

This library implements metrics based on evaluation protocols from:

- **KITTI 3D Object Detection & Odometry Benchmarks**
- **nuScenes Detection, Tracking & Occupancy Tasks**
- **Waymo Open Dataset Challenges**
- **Argoverse Motion Forecasting & HD Map Detection**
- **OpenLane-V2 Topology Challenge**
- **nuPlan Planning Challenge**
- **CARLA Autonomous Driving Challenge**

## References

- KITTI: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- nuScenes: [https://www.nuscenes.org/](https://www.nuscenes.org/)
- Waymo: [https://waymo.com/open/](https://waymo.com/open/)
- Argoverse: [https://www.argoverse.org/](https://www.argoverse.org/)
- OpenLane-V2: [https://openlanev2.github.io/](https://openlanev2.github.io/)
- nuPlan: [https://www.nuscenes.org/nuplan](https://www.nuscenes.org/nuplan)

---

