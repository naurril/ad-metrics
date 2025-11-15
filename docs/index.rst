AD-Metrics Documentation
========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   api/index
   metrics_guide
   examples

Introduction
------------

AD-Metrics is a comprehensive Python library for evaluating autonomous driving systems across 125+ metrics in 9 categories.

Features
~~~~~~~~

* **Detection**: 25 metrics including IoU, AP, NDS, AOS
* **Tracking**: 14 metrics including MOTA, HOTA, IDF1
* **Trajectory Prediction**: 11 metrics including ADE, FDE, NLL
* **Localization**: 8 metrics including ATE, RTE, ARE
* **Occupancy**: 10 metrics including mIoU, Scene Completion
* **Planning**: 12 metrics including Driving Score, Collision Rate
* **Vector Maps**: 8 metrics including Chamfer Distance, Topology
* **Simulation Quality**: 29 metrics for sensor fidelity
* **Utilities**: Matching, NMS, transforms

Supported Benchmarks
~~~~~~~~~~~~~~~~~~~~

* KITTI (Detection, Odometry, AOS)
* nuScenes (Detection, Tracking, Occupancy, Maps)
* Waymo Open Dataset
* Argoverse (Prediction, Maps)
* OpenLane-V2 (Topology)
* nuPlan (Planning)
* CARLA (End-to-End)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
