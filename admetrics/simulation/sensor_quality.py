"""
Sensor Quality Metrics for Simulation Validation.

Metrics to evaluate the realism and fidelity of simulated sensor data compared to
real-world sensor data. Critical for sim-to-real transfer in autonomous driving.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist


def camera_image_quality(
    sim_images: np.ndarray,
    real_images: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate camera image simulation quality.
    
    Compares simulated camera images to real-world images to assess visual realism.
    
    Args:
        sim_images: Simulated images, shape (N, H, W, C) where C is channels (RGB)
        real_images: Real-world images, same shape as sim_images
        metrics: List of metrics to compute. Options:
            - 'psnr': Peak Signal-to-Noise Ratio
            - 'ssim': Structural Similarity Index
            - 'lpips': Learned Perceptual Image Patch Similarity (requires model)
            - 'fid': Fréchet Inception Distance (distribution-level)
            - 'color_distribution': Color histogram KL divergence
            - 'brightness': Mean brightness difference
            - 'contrast': Contrast difference
            
    Returns:
        Dictionary with requested metrics
        
    Example:
        >>> sim_imgs = np.random.rand(10, 224, 224, 3) * 255
        >>> real_imgs = sim_imgs + np.random.randn(*sim_imgs.shape) * 10
        >>> quality = camera_image_quality(sim_imgs, real_imgs, ['psnr', 'color_distribution'])
    """
    if sim_images.shape != real_images.shape:
        raise ValueError(f"Image shape mismatch: {sim_images.shape} vs {real_images.shape}")
    
    if metrics is None:
        metrics = ['psnr', 'color_distribution', 'brightness', 'contrast']
    
    results = {}
    
    # Peak Signal-to-Noise Ratio
    if 'psnr' in metrics:
        mse = np.mean((sim_images - real_images) ** 2)
        if mse == 0:
            results['psnr'] = float('inf')
        else:
            max_pixel = 255.0
            results['psnr'] = float(20 * np.log10(max_pixel / np.sqrt(mse)))
    
    # Structural Similarity Index (simplified version)
    if 'ssim' in metrics:
        # Simplified SSIM calculation (full implementation requires scipy.ndimage)
        mu_sim = np.mean(sim_images, axis=(1, 2, 3))
        mu_real = np.mean(real_images, axis=(1, 2, 3))
        var_sim = np.var(sim_images, axis=(1, 2, 3))
        var_real = np.var(real_images, axis=(1, 2, 3))
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        ssim_values = []
        for i in range(len(sim_images)):
            luminance = (2 * mu_sim[i] * mu_real[i] + C1) / (mu_sim[i]**2 + mu_real[i]**2 + C1)
            contrast = (2 * np.sqrt(var_sim[i]) * np.sqrt(var_real[i]) + C2) / (var_sim[i] + var_real[i] + C2)
            ssim_values.append(luminance * contrast)
        
        results['ssim'] = float(np.mean(ssim_values))
    
    # Color distribution similarity (KL divergence of RGB histograms)
    if 'color_distribution' in metrics:
        kl_divs = []
        for channel in range(sim_images.shape[-1]):
            sim_hist, _ = np.histogram(sim_images[..., channel].flatten(), bins=256, range=(0, 256), density=True)
            real_hist, _ = np.histogram(real_images[..., channel].flatten(), bins=256, range=(0, 256), density=True)
            
            # Add small epsilon to avoid log(0)
            sim_hist = sim_hist + 1e-10
            real_hist = real_hist + 1e-10
            
            # Normalize
            sim_hist = sim_hist / np.sum(sim_hist)
            real_hist = real_hist / np.sum(real_hist)
            
            kl_div = stats.entropy(sim_hist, real_hist)
            kl_divs.append(kl_div)
        
        results['color_kl_divergence'] = float(np.mean(kl_divs))
    
    # Brightness comparison
    if 'brightness' in metrics:
        sim_brightness = np.mean(sim_images)
        real_brightness = np.mean(real_images)
        results['brightness_diff'] = float(abs(sim_brightness - real_brightness))
        results['brightness_ratio'] = float(sim_brightness / (real_brightness + 1e-6))
    
    # Contrast comparison
    if 'contrast' in metrics:
        sim_contrast = np.std(sim_images)
        real_contrast = np.std(real_images)
        results['contrast_diff'] = float(abs(sim_contrast - real_contrast))
        results['contrast_ratio'] = float(sim_contrast / (real_contrast + 1e-6))
    
    return results


def lidar_point_cloud_quality(
    sim_points: np.ndarray,
    real_points: np.ndarray,
    max_range: float = 100.0
) -> Dict[str, float]:
    """
    Evaluate LiDAR point cloud simulation quality.
    
    Compares simulated LiDAR scans to real-world scans to assess geometric fidelity.
    
    Args:
        sim_points: Simulated point cloud, shape (N, 3) or (N, 4) for [x, y, z] or [x, y, z, intensity]
        real_points: Real-world point cloud, shape (M, 3) or (M, 4)
        max_range: Maximum sensor range in meters
        
    Returns:
        Dictionary with quality metrics:
            - chamfer_distance: Average nearest-neighbor distance (bidirectional)
            - point_density: Ratio of simulated to real point counts
            - range_distribution: KL divergence of range histograms
            - intensity_correlation: Correlation of intensity values (if available)
            - vertical_distribution: Distribution of vertical angles
            
    Example:
        >>> sim_pc = np.random.randn(1000, 3) * 10
        >>> real_pc = sim_pc + np.random.randn(*sim_pc.shape) * 0.5
        >>> quality = lidar_point_cloud_quality(sim_pc, real_pc)
    """
    results = {}
    
    # Extract XYZ coordinates
    sim_xyz = sim_points[:, :3]
    real_xyz = real_points[:, :3]
    
    # Chamfer Distance (symmetric)
    # Subsample for efficiency if too many points
    if len(sim_xyz) > 10000:
        sim_sample = sim_xyz[np.random.choice(len(sim_xyz), 10000, replace=False)]
    else:
        sim_sample = sim_xyz
        
    if len(real_xyz) > 10000:
        real_sample = real_xyz[np.random.choice(len(real_xyz), 10000, replace=False)]
    else:
        real_sample = real_xyz
    
    # Sim to real distances
    dist_sim_to_real = cdist(sim_sample, real_sample)
    min_dist_sim = np.min(dist_sim_to_real, axis=1)
    chamfer_sim_to_real = np.mean(min_dist_sim)
    
    # Real to sim distances
    dist_real_to_sim = cdist(real_sample, sim_sample)
    min_dist_real = np.min(dist_real_to_sim, axis=1)
    chamfer_real_to_sim = np.mean(min_dist_real)
    
    results['chamfer_distance'] = float((chamfer_sim_to_real + chamfer_real_to_sim) / 2)
    
    # Point density comparison
    results['point_density_ratio'] = float(len(sim_xyz) / (len(real_xyz) + 1e-6))
    results['point_count_sim'] = len(sim_xyz)
    results['point_count_real'] = len(real_xyz)
    
    # Range distribution (distance from origin)
    sim_ranges = np.linalg.norm(sim_xyz, axis=1)
    real_ranges = np.linalg.norm(real_xyz, axis=1)
    
    bins = np.linspace(0, max_range, 50)
    sim_range_hist, _ = np.histogram(sim_ranges, bins=bins, density=True)
    real_range_hist, _ = np.histogram(real_ranges, bins=bins, density=True)
    
    # Add epsilon and normalize
    sim_range_hist = sim_range_hist + 1e-10
    real_range_hist = real_range_hist + 1e-10
    sim_range_hist = sim_range_hist / np.sum(sim_range_hist)
    real_range_hist = real_range_hist / np.sum(real_range_hist)
    
    results['range_distribution_kl'] = float(stats.entropy(sim_range_hist, real_range_hist))
    
    # Intensity correlation (if available)
    if sim_points.shape[1] >= 4 and real_points.shape[1] >= 4:
        # Match points for intensity comparison
        if len(sim_sample) <= len(real_sample):
            dist_matrix = cdist(sim_sample, real_sample)
            matched_real_idx = np.argmin(dist_matrix, axis=1)
            
            sim_intensities = sim_points[np.random.choice(len(sim_xyz), len(sim_sample), replace=False), 3]
            real_intensities = real_points[matched_real_idx, 3]
            
            if len(sim_intensities) > 0 and len(real_intensities) > 0:
                correlation = np.corrcoef(sim_intensities, real_intensities)[0, 1]
                results['intensity_correlation'] = float(correlation)
    
    # Vertical angle distribution
    sim_vertical_angles = np.arctan2(sim_xyz[:, 2], np.linalg.norm(sim_xyz[:, :2], axis=1))
    real_vertical_angles = np.arctan2(real_xyz[:, 2], np.linalg.norm(real_xyz[:, :2], axis=1))
    
    angle_bins = np.linspace(-np.pi/2, np.pi/2, 30)
    sim_angle_hist, _ = np.histogram(sim_vertical_angles, bins=angle_bins, density=True)
    real_angle_hist, _ = np.histogram(real_vertical_angles, bins=angle_bins, density=True)
    
    sim_angle_hist = sim_angle_hist + 1e-10
    real_angle_hist = real_angle_hist + 1e-10
    sim_angle_hist = sim_angle_hist / np.sum(sim_angle_hist)
    real_angle_hist = real_angle_hist / np.sum(real_angle_hist)
    
    results['vertical_angle_kl'] = float(stats.entropy(sim_angle_hist, real_angle_hist))
    
    return results


def radar_quality(
    sim_detections: np.ndarray,
    real_detections: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate radar simulation quality.
    
    Compares simulated radar detections to real radar data.
    
    Args:
        sim_detections: Simulated radar detections, shape (N, D) where D includes:
            [x, y, z, velocity, rcs] - position, radial velocity, radar cross-section
        real_detections: Real radar detections, same format
        
    Returns:
        Dictionary with quality metrics:
            - detection_density: Ratio of detection counts
            - velocity_distribution: KL divergence of velocity distributions
            - rcs_distribution: Radar cross-section distribution similarity
            - spatial_accuracy: Chamfer distance of detection positions
            
    Example:
        >>> sim_radar = np.random.randn(50, 5)
        >>> real_radar = np.random.randn(60, 5)
        >>> quality = radar_quality(sim_radar, real_radar)
    """
    results = {}
    
    # Detection density
    results['detection_density_ratio'] = float(len(sim_detections) / (len(real_detections) + 1e-6))
    results['detection_count_sim'] = len(sim_detections)
    results['detection_count_real'] = len(real_detections)
    
    if len(sim_detections) == 0 or len(real_detections) == 0:
        return results
    
    # Spatial accuracy (Chamfer distance on positions)
    sim_positions = sim_detections[:, :3]
    real_positions = real_detections[:, :3]
    
    dist_matrix = cdist(sim_positions, real_positions)
    chamfer_sim = np.mean(np.min(dist_matrix, axis=1))
    chamfer_real = np.mean(np.min(dist_matrix, axis=0))
    results['spatial_chamfer'] = float((chamfer_sim + chamfer_real) / 2)
    
    # Velocity distribution (if available)
    if sim_detections.shape[1] >= 4 and real_detections.shape[1] >= 4:
        sim_velocities = sim_detections[:, 3]
        real_velocities = real_detections[:, 3]
        
        bins = np.linspace(-30, 30, 40)  # m/s range
        sim_vel_hist, _ = np.histogram(sim_velocities, bins=bins, density=True)
        real_vel_hist, _ = np.histogram(real_velocities, bins=bins, density=True)
        
        sim_vel_hist = sim_vel_hist + 1e-10
        real_vel_hist = real_vel_hist + 1e-10
        sim_vel_hist = sim_vel_hist / np.sum(sim_vel_hist)
        real_vel_hist = real_vel_hist / np.sum(real_vel_hist)
        
        results['velocity_distribution_kl'] = float(stats.entropy(sim_vel_hist, real_vel_hist))
    
    # RCS distribution (if available)
    if sim_detections.shape[1] >= 5 and real_detections.shape[1] >= 5:
        sim_rcs = sim_detections[:, 4]
        real_rcs = real_detections[:, 4]
        
        # RCS in dBsm, typically ranges from -10 to 40
        bins = np.linspace(-10, 40, 30)
        sim_rcs_hist, _ = np.histogram(sim_rcs, bins=bins, density=True)
        real_rcs_hist, _ = np.histogram(real_rcs, bins=bins, density=True)
        
        sim_rcs_hist = sim_rcs_hist + 1e-10
        real_rcs_hist = real_rcs_hist + 1e-10
        sim_rcs_hist = sim_rcs_hist / np.sum(sim_rcs_hist)
        real_rcs_hist = real_rcs_hist / np.sum(real_rcs_hist)
        
        results['rcs_distribution_kl'] = float(stats.entropy(sim_rcs_hist, real_rcs_hist))
    
    return results


def sensor_noise_characteristics(
    sim_measurements: np.ndarray,
    real_measurements: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compare noise characteristics between simulated and real sensors.
    
    Evaluates whether simulated sensor noise matches real-world noise patterns.
    
    Args:
        sim_measurements: Repeated measurements from simulation, shape (N, D)
        real_measurements: Repeated measurements from real sensor, shape (M, D)
        ground_truth: Optional ground truth values, shape (D,)
        
    Returns:
        Dictionary with noise statistics:
            - noise_std_ratio: Ratio of standard deviations
            - noise_distribution_ks: Kolmogorov-Smirnov test statistic
            - bias_sim: Mean error from ground truth (if provided)
            - bias_real: Mean error from ground truth (if provided)
            - snr_ratio: Signal-to-noise ratio comparison
            
    Example:
        >>> # Repeated measurements of same target
        >>> sim_meas = np.random.randn(100, 3) * 0.1 + np.array([10, 0, 0])
        >>> real_meas = np.random.randn(100, 3) * 0.15 + np.array([10, 0, 0])
        >>> gt = np.array([10, 0, 0])
        >>> quality = sensor_noise_characteristics(sim_meas, real_meas, gt)
    """
    results = {}
    
    # Standard deviation comparison
    sim_std = np.std(sim_measurements, axis=0)
    real_std = np.std(real_measurements, axis=0)
    
    results['noise_std_sim'] = float(np.mean(sim_std))
    results['noise_std_real'] = float(np.mean(real_std))
    results['noise_std_ratio'] = float(np.mean(sim_std) / (np.mean(real_std) + 1e-6))
    
    # Distribution comparison (Kolmogorov-Smirnov test)
    # Flatten and compare overall distributions
    sim_flat = sim_measurements.flatten()
    real_flat = real_measurements.flatten()
    
    ks_statistic, ks_pvalue = stats.ks_2samp(sim_flat, real_flat)
    results['noise_distribution_ks_statistic'] = float(ks_statistic)
    results['noise_distribution_ks_pvalue'] = float(ks_pvalue)
    
    # Bias analysis (if ground truth available)
    if ground_truth is not None:
        sim_errors = sim_measurements - ground_truth
        real_errors = real_measurements - ground_truth
        
        results['bias_sim'] = float(np.mean(np.linalg.norm(sim_errors, axis=1)))
        results['bias_real'] = float(np.mean(np.linalg.norm(real_errors, axis=1)))
        results['bias_ratio'] = float(results['bias_sim'] / (results['bias_real'] + 1e-6))
        
        # Signal-to-noise ratio
        signal_power_sim = np.mean(ground_truth ** 2)
        noise_power_sim = np.mean(sim_errors ** 2)
        snr_sim = 10 * np.log10(signal_power_sim / (noise_power_sim + 1e-10))
        
        noise_power_real = np.mean(real_errors ** 2)
        snr_real = 10 * np.log10(signal_power_sim / (noise_power_real + 1e-10))
        
        results['snr_sim_db'] = float(snr_sim)
        results['snr_real_db'] = float(snr_real)
        results['snr_ratio'] = float(snr_sim / (snr_real + 1e-6))
    
    return results


def multimodal_sensor_alignment(
    camera_detections: np.ndarray,
    lidar_detections: np.ndarray,
    camera_to_lidar_transform: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate alignment quality between different sensor modalities.
    
    Critical for sensor fusion validation in simulation.
    
    Args:
        camera_detections: 3D bounding boxes from camera, shape (N, 7) [x, y, z, l, w, h, yaw]
        lidar_detections: 3D bounding boxes from LiDAR, shape (M, 7)
        camera_to_lidar_transform: Optional 4x4 transformation matrix
        
    Returns:
        Dictionary with alignment metrics:
            - spatial_alignment_error: Mean position error for matched detections
            - detection_agreement_rate: Fraction of detections that match
            - size_consistency: Mean size difference for matched detections
            - temporal_sync_error: Timestamp synchronization error (if available)
            
    Example:
        >>> cam_det = np.random.randn(10, 7)
        >>> lidar_det = cam_det + np.random.randn(*cam_det.shape) * 0.2
        >>> quality = multimodal_sensor_alignment(cam_det, lidar_det)
    """
    results = {}
    
    if len(camera_detections) == 0 or len(lidar_detections) == 0:
        results['detection_agreement_rate'] = 0.0
        return results
    
    # Apply transformation if provided
    if camera_to_lidar_transform is not None:
        # Transform camera detections to LiDAR frame
        cam_positions = camera_detections[:, :3]
        ones = np.ones((len(cam_positions), 1))
        cam_positions_h = np.hstack([cam_positions, ones])
        transformed_positions = (camera_to_lidar_transform @ cam_positions_h.T).T[:, :3]
        camera_detections = camera_detections.copy()
        camera_detections[:, :3] = transformed_positions
    
    # Match detections using Hungarian algorithm (simplified: greedy matching)
    cam_centers = camera_detections[:, :3]
    lidar_centers = lidar_detections[:, :3]
    
    dist_matrix = cdist(cam_centers, lidar_centers)
    
    # Greedy matching with threshold
    matches = []
    match_threshold = 2.0  # meters
    
    used_lidar = set()
    for cam_idx in range(len(camera_detections)):
        distances = dist_matrix[cam_idx]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < match_threshold and min_idx not in used_lidar:
            matches.append((cam_idx, min_idx, distances[min_idx]))
            used_lidar.add(min_idx)
    
    # Detection agreement rate
    total_detections = max(len(camera_detections), len(lidar_detections))
    results['detection_agreement_rate'] = float(len(matches) / total_detections)
    results['matched_detections'] = len(matches)
    results['camera_only_detections'] = len(camera_detections) - len(matches)
    results['lidar_only_detections'] = len(lidar_detections) - len(matches)
    
    if len(matches) == 0:
        return results
    
    # Spatial alignment error
    spatial_errors = [dist for _, _, dist in matches]
    results['spatial_alignment_error'] = float(np.mean(spatial_errors))
    results['spatial_alignment_std'] = float(np.std(spatial_errors))
    
    # Size consistency (L, W, H)
    cam_sizes = camera_detections[[m[0] for m in matches], 3:6]
    lidar_sizes = lidar_detections[[m[1] for m in matches], 3:6]
    
    size_errors = np.abs(cam_sizes - lidar_sizes)
    results['size_consistency_error'] = float(np.mean(size_errors))
    results['length_error'] = float(np.mean(size_errors[:, 0]))
    results['width_error'] = float(np.mean(size_errors[:, 1]))
    results['height_error'] = float(np.mean(size_errors[:, 2]))
    
    # Orientation consistency
    cam_yaws = camera_detections[[m[0] for m in matches], 6]
    lidar_yaws = lidar_detections[[m[1] for m in matches], 6]
    
    yaw_errors = np.abs(np.arctan2(np.sin(cam_yaws - lidar_yaws), np.cos(cam_yaws - lidar_yaws)))
    results['orientation_error_rad'] = float(np.mean(yaw_errors))
    results['orientation_error_deg'] = float(np.degrees(np.mean(yaw_errors)))
    
    return results


def temporal_consistency(
    detections_sequence: List[np.ndarray],
    fps: float = 10.0
) -> Dict[str, float]:
    """
    Evaluate temporal consistency of sensor detections.
    
    Simulated sensors should maintain realistic temporal coherence.
    
    Args:
        detections_sequence: List of detection arrays over time, each shape (N_t, D)
        fps: Frames per second of sensor
        
    Returns:
        Dictionary with temporal metrics:
            - detection_count_variance: Variance in number of detections
            - motion_smoothness: Smoothness of tracked object motion
            - appearance_disappearance_rate: Rate of flickering detections
            - frame_to_frame_consistency: Average detection overlap between frames
            
    Example:
        >>> sequence = [np.random.randn(10, 7) for _ in range(20)]
        >>> quality = temporal_consistency(sequence)
    """
    results = {}
    
    if len(detections_sequence) < 2:
        return results
    
    # Detection count variance
    detection_counts = [len(dets) for dets in detections_sequence]
    results['detection_count_mean'] = float(np.mean(detection_counts))
    results['detection_count_std'] = float(np.std(detection_counts))
    results['detection_count_variance'] = float(np.var(detection_counts))
    
    # Frame-to-frame consistency (IoU-based matching rate)
    frame_consistencies = []
    for i in range(len(detections_sequence) - 1):
        curr_dets = detections_sequence[i]
        next_dets = detections_sequence[i + 1]
        
        if len(curr_dets) == 0 or len(next_dets) == 0:
            continue
        
        # Simple distance-based matching
        curr_centers = curr_dets[:, :3] if curr_dets.shape[1] >= 3 else curr_dets
        next_centers = next_dets[:, :3] if next_dets.shape[1] >= 3 else next_dets
        
        dist_matrix = cdist(curr_centers, next_centers)
        
        # Count matches within threshold (accounting for motion at fps)
        max_motion = 20.0 / fps  # Assume max 20 m/s motion
        matches = np.sum(np.min(dist_matrix, axis=1) < max_motion)
        
        consistency = matches / max(len(curr_dets), len(next_dets))
        frame_consistencies.append(consistency)
    
    if len(frame_consistencies) > 0:
        results['frame_to_frame_consistency'] = float(np.mean(frame_consistencies))
        results['consistency_std'] = float(np.std(frame_consistencies))
    
    # Appearance/disappearance rate (flickering)
    # Count new and lost detections per frame
    new_detections = []
    lost_detections = []
    
    for i in range(len(detections_sequence) - 1):
        curr_count = len(detections_sequence[i])
        next_count = len(detections_sequence[i + 1])
        
        # Simplified: just count changes
        if next_count > curr_count:
            new_detections.append(next_count - curr_count)
        elif curr_count > next_count:
            lost_detections.append(curr_count - next_count)
    
    if len(new_detections) > 0:
        results['avg_new_detections_per_frame'] = float(np.mean(new_detections))
    if len(lost_detections) > 0:
        results['avg_lost_detections_per_frame'] = float(np.mean(lost_detections))
    
    flicker_rate = (len(new_detections) + len(lost_detections)) / (len(detections_sequence) - 1)
    results['flicker_rate'] = float(flicker_rate)
    
    return results


def perception_sim2real_gap(
    sim_detections: List[Dict],
    real_detections: List[Dict],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Measure the sim-to-real gap for perception performance.
    
    Compares object detection/tracking performance between simulation and real world.
    
    Args:
        sim_detections: Detection results from simulation, list of dicts with:
            - 'predictions': predicted boxes
            - 'ground_truth': ground truth boxes
            - 'scores': confidence scores
        real_detections: Detection results from real-world data, same format
        metrics: List of metrics to compute ['ap', 'recall', 'precision', 'latency']
        
    Returns:
        Dictionary with sim2real gap metrics:
            - ap_gap: Difference in Average Precision
            - recall_gap: Difference in recall
            - precision_gap: Difference in precision
            - performance_drop: Overall performance degradation
            
    Example:
        >>> sim_results = [{'predictions': np.random.randn(5, 7), 'ground_truth': np.random.randn(5, 7)}]
        >>> real_results = [{'predictions': np.random.randn(4, 7), 'ground_truth': np.random.randn(5, 7)}]
        >>> gap = perception_sim2real_gap(sim_results, real_results)
    """
    results = {}
    
    if metrics is None:
        metrics = ['recall', 'precision']
    
    # Calculate metrics for sim and real
    sim_tp = sim_fp = sim_fn = 0
    real_tp = real_fp = real_fn = 0
    
    iou_threshold = 0.5
    
    for det_dict in sim_detections:
        preds = det_dict.get('predictions', np.array([]))
        gts = det_dict.get('ground_truth', np.array([]))
        
        if len(preds) > 0 and len(gts) > 0:
            # Simple matching based on distance
            pred_centers = preds[:, :3] if preds.shape[1] >= 3 else preds
            gt_centers = gts[:, :3] if gts.shape[1] >= 3 else gts
            
            dist_matrix = cdist(pred_centers, gt_centers)
            
            matched_gts = set()
            for pred_idx in range(len(preds)):
                min_dist = np.min(dist_matrix[pred_idx])
                min_idx = np.argmin(dist_matrix[pred_idx])
                
                if min_dist < 2.0 and min_idx not in matched_gts:  # 2m threshold
                    sim_tp += 1
                    matched_gts.add(min_idx)
                else:
                    sim_fp += 1
            
            sim_fn += len(gts) - len(matched_gts)
        elif len(preds) > 0:
            sim_fp += len(preds)
        elif len(gts) > 0:
            sim_fn += len(gts)
    
    for det_dict in real_detections:
        preds = det_dict.get('predictions', np.array([]))
        gts = det_dict.get('ground_truth', np.array([]))
        
        if len(preds) > 0 and len(gts) > 0:
            pred_centers = preds[:, :3] if preds.shape[1] >= 3 else preds
            gt_centers = gts[:, :3] if gts.shape[1] >= 3 else gts
            
            dist_matrix = cdist(pred_centers, gt_centers)
            
            matched_gts = set()
            for pred_idx in range(len(preds)):
                min_dist = np.min(dist_matrix[pred_idx])
                min_idx = np.argmin(dist_matrix[pred_idx])
                
                if min_dist < 2.0 and min_idx not in matched_gts:
                    real_tp += 1
                    matched_gts.add(min_idx)
                else:
                    real_fp += 1
            
            real_fn += len(gts) - len(matched_gts)
        elif len(preds) > 0:
            real_fp += len(preds)
        elif len(gts) > 0:
            real_fn += len(gts)
    
    # Calculate precision and recall
    if 'precision' in metrics:
        sim_precision = sim_tp / (sim_tp + sim_fp + 1e-6)
        real_precision = real_tp / (real_tp + real_fp + 1e-6)
        
        results['precision_sim'] = float(sim_precision)
        results['precision_real'] = float(real_precision)
        results['precision_gap'] = float(sim_precision - real_precision)
    
    if 'recall' in metrics:
        sim_recall = sim_tp / (sim_tp + sim_fn + 1e-6)
        real_recall = real_tp / (real_tp + real_fn + 1e-6)
        
        results['recall_sim'] = float(sim_recall)
        results['recall_real'] = float(real_recall)
        results['recall_gap'] = float(sim_recall - real_recall)
    
    # Overall performance drop (F1-based)
    if 'precision' in metrics and 'recall' in metrics:
        sim_f1 = 2 * (sim_precision * sim_recall) / (sim_precision + sim_recall + 1e-6)
        real_f1 = 2 * (real_precision * real_recall) / (real_precision + real_recall + 1e-6)
        
        results['f1_sim'] = float(sim_f1)
        results['f1_real'] = float(real_f1)
        results['f1_gap'] = float(sim_f1 - real_f1)
        results['performance_drop_pct'] = float((sim_f1 - real_f1) / (sim_f1 + 1e-6) * 100)
    
    return results
