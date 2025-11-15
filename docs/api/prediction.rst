Trajectory Prediction Metrics
=============================

.. automodule:: admetrics.prediction
   :members:
   :undoc-members:
   :show-inheritance:

Displacement Metrics
--------------------

.. autofunction:: admetrics.prediction.trajectory.calculate_ade
.. autofunction:: admetrics.prediction.trajectory.calculate_fde
.. autofunction:: admetrics.prediction.trajectory.calculate_miss_rate

Multi-Modal Metrics
-------------------

.. autofunction:: admetrics.prediction.trajectory.calculate_multimodal_ade
.. autofunction:: admetrics.prediction.trajectory.calculate_multimodal_fde

Probabilistic Metrics
---------------------

.. autofunction:: admetrics.prediction.trajectory.calculate_brier_fde
.. autofunction:: admetrics.prediction.trajectory.calculate_nll

Safety Metrics
--------------

.. autofunction:: admetrics.prediction.trajectory.calculate_collision_rate
.. autofunction:: admetrics.prediction.trajectory.calculate_drivable_area_compliance
