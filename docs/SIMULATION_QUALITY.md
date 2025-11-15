# Simulation Quality Metrics

This document describes metrics for evaluating the quality and realism of simulated sensor data in autonomous vehicle simulators (CARLA, LGSVL, AirSim, Waymo Sim, nuPlan, etc.).

## Table of Contents

1. [Overview](#overview)
2. [Camera Image Quality](#camera-image-quality)
3. [LiDAR Point Cloud Quality](#lidar-point-cloud-quality)
4. [Radar Quality](#radar-quality)
5. [Sensor Noise Characteristics](#sensor-noise-characteristics)
6. [Multimodal Sensor Alignment](#multimodal-sensor-alignment)
7. [Temporal Consistency](#temporal-consistency)
8. [Sim-to-Real Gap](#sim-to-real-gap)
9. [Best Practices](#best-practices)
10. [References](#references)
11. [See Also](#see-also)

## Overview

Simulation quality is critical for successful sim-to-real transfer in autonomous driving. Poor simulation fidelity leads to:
- **Performance degradation** when deploying to real vehicles
- **False confidence** in perception/planning algorithms
- **Wasted development time** on unrealistic scenarios

### Key Evaluation Dimensions

| Dimension | What It Measures | Why It Matters |
|-----------|------------------|----------------|
| **Visual Realism** | Camera image fidelity | Affects vision-based perception |
| **Geometric Accuracy** | LiDAR/radar spatial quality | Critical for localization, mapping |
| **Noise Characteristics** | Sensor noise patterns | Impacts robustness, filtering |
| **Calibration** | Multi-sensor alignment | Essential for sensor fusion |
| **Temporal Coherence** | Frame-to-frame consistency | Affects tracking, prediction |
| **Performance Gap** | Sim vs real detection accuracy | Quantifies domain shift |

## Camera Image Quality

### Purpose
Evaluate visual realism of simulated camera images compared to real-world data.

### Metrics

#### PSNR (Peak Signal-to-Noise Ratio)
**Formula:**
```
PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
```

**Interpretation:**
- **> 40 dB**: Excellent quality (near-identical)
- **30-40 dB**: Good quality
- **20-30 dB**: Acceptable
- **< 20 dB**: Poor quality

#### Color Distribution (KL Divergence)
**Measures:** Similarity of RGB histograms between sim and real

**Interpretation:**
- **< 0.1**: Very similar color distribution
- **0.1-0.5**: Moderate similarity
- **> 0.5**: Significant color shift (lighting/material differences)

#### Brightness/Contrast Ratios
**Measures:** Ratio of mean brightness and standard deviation

**Interpretation:**
- **~1.0**: Well-matched
- **> 1.2**: Sim is brighter/higher contrast
- **< 0.8**: Sim is darker/lower contrast

### Usage

```python
from admetrics.simulation import camera_image_quality

quality = camera_image_quality(
    sim_images,      # (N, H, W, 3)
    real_images,     # (N, H, W, 3)
    metrics=['psnr', 'ssim', 'color_distribution', 'brightness', 'contrast']
)

print(f"PSNR: {quality['psnr']:.2f} dB")
print(f"Color KL: {quality['color_kl_divergence']:.4f}")
```

### Common Issues
- **Too bright/clean**: Simulations often lack weather effects, dirt, glare
- **Perfect colors**: Real cameras have color response curves
- **No motion blur**: Cameras have exposure time
- **Incorrect dynamic range**: HDR vs LDR differences

## LiDAR Point Cloud Quality

### Purpose
Evaluate geometric fidelity of simulated LiDAR scans.

### Metrics

#### Chamfer Distance
**Measures:** Average nearest-neighbor distance between point clouds (bidirectional)

**Formula:**
```
Chamfer = (1/|Sim|) Σ min_dist(p_sim, Real) + (1/|Real|) Σ min_dist(p_real, Sim)
```

**Interpretation:**
- **< 0.2m**: Excellent match
- **0.2-0.5m**: Good match
- **> 0.5m**: Significant geometric differences

#### Point Density Ratio
**Measures:** Ratio of point counts

**Interpretation:**
- **~1.0**: Matching density
- **> 1.5**: Sim has too many points (unrealistic)
- **< 0.7**: Sim is too sparse

#### Range Distribution
**Measures:** KL divergence of distance-from-sensor distributions

**Why:** Simulators might not model occlusion, max range correctly

#### Vertical Angle Distribution
**Measures:** Distribution of elevation angles

**Why:** Checks beam pattern matching (16/32/64/128 channel LiDARs)

### Usage

```python
from admetrics.simulation import lidar_point_cloud_quality

quality = lidar_point_cloud_quality(
    sim_points,    # (N, 3) or (N, 4) with intensity
    real_points,   # (M, 3) or (M, 4)
    max_range=100.0
)

print(f"Chamfer Distance: {quality['chamfer_distance']:.3f}m")
print(f"Point Density Ratio: {quality['point_density_ratio']:.3f}")
```

### Common Issues
- **Perfect reflections**: Real LiDAR has material-dependent returns
- **No dropouts**: Glass, water absorb/scatter beams
- **Ideal beam patterns**: Real LiDARs have beam divergence
- **Missing multi-path**: Reflections, scattering

## Radar Quality

### Purpose
Evaluate radar simulation fidelity (less common but important for robust AV).

### Metrics

#### Detection Density
**Measures:** Ratio of detection counts

#### Spatial Accuracy (Chamfer)
**Measures:** Position accuracy of radar detections

#### Velocity Distribution
**Measures:** Radial velocity histogram matching

**Why:** Radar measures Doppler shift - critical for velocity estimation

#### RCS Distribution (Radar Cross-Section)
**Measures:** Signal strength distribution

**Why:** RCS varies by material, angle - affects detection range

### Usage

```python
from admetrics.simulation import radar_quality

quality = radar_quality(
    sim_detections,  # (N, 5) [x, y, z, velocity, rcs]
    real_detections  # (M, 5)
)

print(f"Detection Density: {quality['detection_density_ratio']:.3f}")
print(f"RCS Distribution KL: {quality['rcs_distribution_kl']:.4f}")
```

### Common Issues
- **Too many detections**: Radar has clutter, false alarms
- **Unrealistic RCS**: Material properties affect returns
- **Missing multi-path**: Ground reflections common
- **Ideal velocity**: Real radar has velocity resolution limits

## Sensor Noise Characteristics

### Purpose
Validate that simulated sensor noise matches real-world noise patterns.

### Metrics

#### Noise Std Ratio
**Measures:** Ratio of noise standard deviations

**Critical:** Most simulators underestimate noise

#### KS Test (Kolmogorov-Smirnov)
**Measures:** Distribution similarity test

**Interpretation:**
- **p-value > 0.05**: Distributions match (cannot reject null hypothesis)
- **p-value < 0.05**: Distributions differ significantly

#### Signal-to-Noise Ratio (SNR)
**Measures:** SNR in dB

**Formula:**
```
SNR = 10 * log10(signal_power / noise_power)
```

### Usage

```python
from admetrics.simulation import sensor_noise_characteristics

# Repeated measurements of same target
quality = sensor_noise_characteristics(
    sim_measurements,   # (N, D)
    real_measurements,  # (M, D)
    ground_truth=gt     # (D,)
)

print(f"Noise Std Ratio: {quality['noise_std_ratio']:.3f}")
print(f"KS p-value: {quality['noise_distribution_ks_pvalue']:.4f}")
```

### Common Issues
- **Too little noise**: Simulations are often optimistic
- **Wrong distribution**: Real sensors have non-Gaussian tails
- **No systematic errors**: Bias, drift, temperature effects
- **Missing correlations**: Spatial/temporal noise correlations

## Multimodal Sensor Alignment

### Purpose
Evaluate calibration quality between different sensor modalities.

### Metrics

#### Detection Agreement Rate
**Measures:** Fraction of detections visible in both sensors

**Why:** Mis-calibration leads to disagreement

#### Spatial Alignment Error
**Measures:** Mean position difference for matched detections

**Interpretation:**
- **< 0.2m**: Excellent calibration
- **0.2-0.5m**: Acceptable
- **> 0.5m**: Poor calibration (fusion will fail)

#### Size/Orientation Consistency
**Measures:** Bounding box parameter consistency

### Usage

```python
from admetrics.simulation import multimodal_sensor_alignment

quality = multimodal_sensor_alignment(
    camera_detections,  # (N, 7) [x, y, z, l, w, h, yaw]
    lidar_detections    # (M, 7)
)

print(f"Agreement Rate: {quality['detection_agreement_rate']:.2%}")
print(f"Spatial Alignment: {quality['spatial_alignment_error']:.3f}m")
```

### Common Issues
- **Perfect alignment**: Real systems have calibration errors
- **No temporal offset**: Camera/LiDAR not synchronized
- **Missing FOV differences**: Different sensor coverage

## Temporal Consistency

### Purpose
Evaluate frame-to-frame coherence of sensor data.

### Metrics

#### Detection Count Variance
**Measures:** Stability of detection counts over time

**Why:** Flickering detections indicate poor tracking

#### Frame-to-Frame Consistency
**Measures:** Fraction of detections that persist between frames

**Interpretation:**
- **> 80%**: Good temporal coherence
- **50-80%**: Moderate flickering
- **< 50%**: Severe instability

#### Flicker Rate
**Measures:** Rate of appearance/disappearance events

### Usage

```python
from admetrics.simulation import temporal_consistency

quality = temporal_consistency(
    detections_sequence,  # List of (N_t, D) arrays
    fps=10.0
)

print(f"Frame Consistency: {quality['frame_to_frame_consistency']:.2%}")
print(f"Flicker Rate: {quality['flicker_rate']:.2%}")
```

### Common Issues
- **Unstable detections**: Real trackers smooth detections
- **No prediction**: Real systems use Kalman filters
- **Missing occlusion handling**: Objects appear/disappear realistically

## Sim-to-Real Gap

### Purpose
Quantify performance degradation from simulation to real-world deployment.

### Metrics

#### Precision/Recall Gap
**Measures:** Difference in detection metrics

**Formula:**
```
Gap = Performance_sim - Performance_real
```

#### Performance Drop
**Measures:** Percentage degradation

**Interpretation:**
- **< 10%**: Excellent sim quality
- **10-20%**: Acceptable gap
- **20-30%**: Significant domain shift
- **> 30%**: Poor simulation fidelity

### Usage

```python
from admetrics.simulation import perception_sim2real_gap

gap = perception_sim2real_gap(
    sim_results,   # List of dicts with predictions/ground_truth
    real_results   # Same format
)

print(f"Performance Drop: {gap['performance_drop_pct']:.1f}%")
print(f"Precision Gap: {gap['precision_gap']:+.2%}")
```

### Mitigation Strategies

**Domain Randomization:**
- Randomize lighting, textures, weather
- Vary sensor parameters (noise, blur, distortion)
- Add artifacts (lens flare, motion blur, compression)

**Domain Adaptation:**
- CycleGAN for image translation
- Feature alignment techniques
- Self-supervised learning on real data

**Hybrid Approaches:**
- Train on sim, fine-tune on real
- Mix sim and real data
- Use sim for rare scenarios, real for common

## Best Practices

### 1. Establish Baselines

**Collect real-world data:**
```python
# Record sensor data in various conditions
real_camera_data = collect_real_camera_images(scenarios)
real_lidar_data = collect_real_lidar_scans(scenarios)
```

**Evaluate simulation:**
```python
# Compare on same scenarios
quality = camera_image_quality(sim_data, real_data)
assert quality['psnr'] > 25, "Image quality too poor"
```

### 2. Scenario Coverage

**Test diverse conditions:**
- Time of day (dawn, day, dusk, night)
- Weather (clear, rain, fog, snow)
- Traffic density (sparse, moderate, dense)
- Road types (highway, urban, rural)

### 3. Sensor Configuration

**Match real hardware:**
```python
# Configure simulator to match real sensors
lidar_config = {
    'channels': 64,
    'range': 100.0,
    'rotation_frequency': 10,  # Hz
    'points_per_second': 1.3e6,
    'noise_std': 0.02  # meters
}
```

### 4. Iterative Refinement

**Identify and fix issues:**
1. Run quality metrics
2. Identify largest gaps
3. Improve simulator (materials, lighting, physics)
4. Re-evaluate
5. Repeat until gaps acceptable

### 5. Validation Pipeline

```python
def validate_simulation(sim_data, real_data):
    """Complete validation pipeline."""
    
    results = {}
    
    # Camera quality
    cam_quality = camera_image_quality(sim_data['camera'], real_data['camera'])
    results['camera_psnr'] = cam_quality['psnr']
    
    # LiDAR quality
    lidar_quality = lidar_point_cloud_quality(sim_data['lidar'], real_data['lidar'])
    results['lidar_chamfer'] = lidar_quality['chamfer_distance']
    
    # Noise characteristics
    noise_quality = sensor_noise_characteristics(
        sim_data['noise_samples'],
        real_data['noise_samples']
    )
    results['noise_std_ratio'] = noise_quality['noise_std_ratio']
    
    # Sim2Real gap
    gap = perception_sim2real_gap(sim_data['detections'], real_data['detections'])
    results['performance_drop'] = gap['performance_drop_pct']
    
    # Summary
    passed = (
        results['camera_psnr'] > 25 and
        results['lidar_chamfer'] < 0.5 and
        abs(results['noise_std_ratio'] - 1.0) < 0.3 and
        results['performance_drop'] < 20
    )
    
    return results, passed
```

### 6. Document Limitations

**Be transparent about sim quality:**
- Known artifacts (e.g., "no rain simulation")
- Performance gaps (e.g., "15% drop in night scenarios")
- Mitigation strategies (e.g., "domain randomization applied")

## Simulator Comparisons

### Popular AV Simulators

| Simulator | Strengths | Weaknesses | Quality Focus |
|-----------|-----------|------------|---------------|
| **CARLA** | Open-source, good graphics | Limited physics | Visual realism |
| **LGSVL** | Multi-sensor, ROS integration | Discontinued | Sensor accuracy |
| **AirSim** | Photorealistic, UE4 engine | Resource intensive | Camera quality |
| **Waymo Sim** | Real-world scenarios | Closed-source | Behavior realism |
| **nuPlan** | Real sensor logs + replay | Not full sim | Data-driven |
| **MetaDrive** | Lightweight, procedural | Simple graphics | Scalability |

### Quality Assessment by Simulator

**Example results (illustrative):**

```
Simulator: CARLA
  Camera PSNR: 28 dB (Good)
  LiDAR Chamfer: 0.35m (Good)
  Performance Drop: 18% (Acceptable)

Simulator: AirSim  
  Camera PSNR: 32 dB (Excellent)
  LiDAR Chamfer: 0.45m (Acceptable)
  Performance Drop: 22% (Acceptable)

Simulator: MetaDrive
  Camera PSNR: 22 dB (Poor)
  LiDAR Chamfer: 0.25m (Good)
  Performance Drop: 28% (Significant)
```

## References

1. **CARLA: An Open Urban Driving Simulator**
   - Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017)
   - Conference on Robot Learning (CoRL) 2017
   - https://arxiv.org/abs/1711.03938
   - https://carla.org
   - https://github.com/carla-simulator/carla
   - Open-source simulator with flexible sensor configuration and evaluation metrics

2. **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**
   - Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017)
   - IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2017
   - https://arxiv.org/abs/1703.06907
   - Techniques for improving sim-to-real transfer through randomization

3. **Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey**
   - Zhao, W., Queralta, J. P., & Westerlund, T. (2020)
   - IEEE Conference on Automation Science and Engineering (CASE) 2020
   - https://arxiv.org/abs/2009.13303
   - Comprehensive survey of simulation-to-reality transfer methods

4. **Simulating LiDAR Point Cloud for Autonomous Driving using Real-world Scenes and Traffic Flows**
   - Fang, J., Zhou, D., Yan, F., Zhao, T., Zhang, F., Ma, Y., Wang, L., & Yang, R. (2018)
   - https://arxiv.org/abs/1811.07112
   - Realistic LiDAR simulation for autonomous driving validation

5. **Real-to-Sim: Predicting Residual Errors of Robotic Systems with Sparse Data using a Learning-based Unscented Kalman Filter**
   - Kurup, A., & Borrelli, F. (2021)
   - IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2021
   - https://arxiv.org/abs/2103.10403
   - Learning-based approach for modeling sensor noise and errors

## See Also

- [END_TO_END_METRICS.md](END_TO_END_METRICS.md) - Planning and driving evaluation
- [DETECTION_METRICS.md](DETECTION_METRICS.md) - Object detection metrics
- [LOCALIZATION_METRICS.md](LOCALIZATION_METRICS.md) - Ego pose metrics
- [METRICS_REFERENCE.md](METRICS_REFERENCE.md) - Quick reference for all metrics
