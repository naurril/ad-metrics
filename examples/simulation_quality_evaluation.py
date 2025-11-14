"""
Example usage of simulation quality metrics for autonomous driving simulators.

This script demonstrates how to evaluate the fidelity and realism of simulated
sensor data (camera, LiDAR, radar) compared to real-world data.
"""

import numpy as np
from admetrics.simulation import (
    camera_image_quality,
    lidar_point_cloud_quality,
    radar_quality,
    sensor_noise_characteristics,
    multimodal_sensor_alignment,
    temporal_consistency,
    perception_sim2real_gap,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def example_1_camera_quality():
    """Example 1: Camera Image Quality Assessment."""
    print_section("Example 1: Camera Image Quality")
    
    # Generate synthetic camera images (simulated vs real)
    np.random.seed(42)
    
    # Real-world images (with realistic noise and artifacts)
    real_images = np.random.rand(5, 224, 224, 3) * 255
    
    # Simulated images (similar but with different characteristics)
    # Simulation might have perfect lighting, less noise, different color distribution
    sim_images = real_images.copy()
    sim_images = sim_images * 1.1  # Slightly brighter
    sim_images = np.clip(sim_images + np.random.randn(*sim_images.shape) * 5, 0, 255)
    
    quality = camera_image_quality(
        sim_images, 
        real_images,
        metrics=['psnr', 'color_distribution', 'brightness', 'contrast']
    )
    
    print("Camera Quality Metrics:")
    print(f"  PSNR: {quality['psnr']:.2f} dB")
    print(f"  Color KL Divergence: {quality['color_kl_divergence']:.4f}")
    print(f"  Brightness Ratio: {quality['brightness_ratio']:.3f}")
    print(f"  Contrast Ratio: {quality['contrast_ratio']:.3f}")
    print("\nInterpretation:")
    print("  - PSNR > 30 dB: Good quality")
    print("  - KL divergence < 0.1: Similar color distribution")
    print("  - Ratios close to 1.0: Well-matched brightness/contrast")


def example_2_lidar_quality():
    """Example 2: LiDAR Point Cloud Quality."""
    print_section("Example 2: LiDAR Point Cloud Quality")
    
    np.random.seed(42)
    
    # Real LiDAR scan (1000 points with intensity)
    real_angles = np.random.uniform(0, 2*np.pi, 1000)
    real_ranges = np.random.uniform(5, 80, 1000)
    real_z = np.random.uniform(-2, 3, 1000)
    
    real_points = np.column_stack([
        real_ranges * np.cos(real_angles),
        real_ranges * np.sin(real_angles),
        real_z,
        np.random.uniform(0, 255, 1000)  # intensity
    ])
    
    # Simulated LiDAR (different point density, slightly different distribution)
    sim_angles = np.random.uniform(0, 2*np.pi, 1200)  # More points in sim
    sim_ranges = np.random.uniform(5, 80, 1200)
    sim_z = np.random.uniform(-2, 3, 1200)
    
    sim_points = np.column_stack([
        sim_ranges * np.cos(sim_angles),
        sim_ranges * np.sin(sim_angles),
        sim_z,
        np.random.uniform(50, 200, 1200)  # Different intensity distribution
    ])
    
    quality = lidar_point_cloud_quality(sim_points, real_points, max_range=100.0)
    
    print("LiDAR Quality Metrics:")
    print(f"  Chamfer Distance: {quality['chamfer_distance']:.3f} m")
    print(f"  Point Density Ratio: {quality['point_density_ratio']:.3f}")
    print(f"    Sim points: {quality['point_count_sim']}")
    print(f"    Real points: {quality['point_count_real']}")
    print(f"  Range Distribution KL: {quality['range_distribution_kl']:.4f}")
    print(f"  Vertical Angle KL: {quality['vertical_angle_kl']:.4f}")
    if 'intensity_correlation' in quality:
        print(f"  Intensity Correlation: {quality['intensity_correlation']:.3f}")
    
    print("\nInterpretation:")
    print("  - Chamfer < 0.5m: Good geometric match")
    print("  - Density ratio ~1.0: Matching point cloud density")
    print("  - Low KL divergence: Similar spatial distributions")


def example_3_radar_quality():
    """Example 3: Radar Detection Quality."""
    print_section("Example 3: Radar Detection Quality")
    
    np.random.seed(42)
    
    # Real radar detections [x, y, z, radial_velocity, rcs]
    real_radar = np.column_stack([
        np.random.uniform(-50, 50, 30),  # x
        np.random.uniform(-30, 30, 30),  # y
        np.random.uniform(-2, 2, 30),     # z
        np.random.uniform(-20, 20, 30),   # velocity
        np.random.uniform(-5, 30, 30)     # RCS in dBsm
    ])
    
    # Simulated radar (different detection characteristics)
    sim_radar = np.column_stack([
        np.random.uniform(-50, 50, 35),   # More detections
        np.random.uniform(-30, 30, 35),
        np.random.uniform(-2, 2, 35),
        np.random.uniform(-20, 20, 35),
        np.random.uniform(0, 35, 35)      # Different RCS distribution
    ])
    
    quality = radar_quality(sim_radar, real_radar)
    
    print("Radar Quality Metrics:")
    print(f"  Detection Density Ratio: {quality['detection_density_ratio']:.3f}")
    print(f"    Sim detections: {quality['detection_count_sim']}")
    print(f"    Real detections: {quality['detection_count_real']}")
    print(f"  Spatial Chamfer: {quality['spatial_chamfer']:.3f} m")
    if 'velocity_distribution_kl' in quality:
        print(f"  Velocity Distribution KL: {quality['velocity_distribution_kl']:.4f}")
    if 'rcs_distribution_kl' in quality:
        print(f"  RCS Distribution KL: {quality['rcs_distribution_kl']:.4f}")
    
    print("\nInterpretation:")
    print("  - Density ratio ~1.0: Similar detection rates")
    print("  - Low spatial error: Accurate positioning")
    print("  - Velocity/RCS KL: Realistic detection characteristics")


def example_4_sensor_noise():
    """Example 4: Sensor Noise Characteristics."""
    print_section("Example 4: Sensor Noise Characteristics")
    
    np.random.seed(42)
    
    # Ground truth position
    ground_truth = np.array([10.0, 5.0, 0.0])
    
    # Real sensor measurements (100 repeated measurements with realistic noise)
    real_noise_std = 0.15  # meters
    real_measurements = ground_truth + np.random.randn(100, 3) * real_noise_std
    
    # Simulated sensor measurements (different noise characteristics)
    sim_noise_std = 0.10  # Simulation often has less noise
    sim_measurements = ground_truth + np.random.randn(100, 3) * sim_noise_std
    
    quality = sensor_noise_characteristics(
        sim_measurements,
        real_measurements,
        ground_truth=ground_truth
    )
    
    print("Noise Characteristics:")
    print(f"  Noise Std (Sim): {quality['noise_std_sim']:.4f} m")
    print(f"  Noise Std (Real): {quality['noise_std_real']:.4f} m")
    print(f"  Std Ratio: {quality['noise_std_ratio']:.3f}")
    print(f"\n  Bias (Sim): {quality['bias_sim']:.4f} m")
    print(f"  Bias (Real): {quality['bias_real']:.4f} m")
    print(f"\n  SNR (Sim): {quality['snr_sim_db']:.2f} dB")
    print(f"  SNR (Real): {quality['snr_real_db']:.2f} dB")
    print(f"\n  KS Test Statistic: {quality['noise_distribution_ks_statistic']:.4f}")
    print(f"  KS Test p-value: {quality['noise_distribution_ks_pvalue']:.4f}")
    
    print("\nInterpretation:")
    print("  - Std ratio ~1.0: Matching noise levels")
    print("  - KS p-value > 0.05: Similar noise distributions")
    print("  - Low bias: Accurate measurements")
    print(f"  - Warning: Sim has {(1 - quality['noise_std_ratio']) * 100:.1f}% less noise!")


def example_5_multimodal_alignment():
    """Example 5: Multimodal Sensor Alignment."""
    print_section("Example 5: Multimodal Sensor Alignment")
    
    np.random.seed(42)
    
    # Generate ground truth detections
    gt_detections = np.array([
        [10, 5, 0, 4.5, 2.0, 1.8, 0.1],
        [20, -3, 0, 4.5, 2.0, 1.8, -0.2],
        [15, 10, 0, 5.0, 2.2, 2.0, 0.5],
    ])
    
    # Camera detections (slight position errors, size errors)
    camera_detections = gt_detections.copy()
    camera_detections[:, :3] += np.random.randn(3, 3) * 0.3  # Position noise
    camera_detections[:, 3:6] += np.random.randn(3, 3) * 0.2  # Size errors
    
    # LiDAR detections (different errors, might miss some detections)
    lidar_detections = gt_detections.copy()
    lidar_detections[:, :3] += np.random.randn(3, 3) * 0.2
    lidar_detections[:, 3:6] += np.random.randn(3, 3) * 0.15
    
    quality = multimodal_sensor_alignment(camera_detections, lidar_detections)
    
    print("Multimodal Alignment Metrics:")
    print(f"  Detection Agreement Rate: {quality['detection_agreement_rate']:.2%}")
    print(f"  Matched Detections: {quality['matched_detections']}")
    print(f"  Camera-only Detections: {quality['camera_only_detections']}")
    print(f"  LiDAR-only Detections: {quality['lidar_only_detections']}")
    print(f"\n  Spatial Alignment Error: {quality['spatial_alignment_error']:.3f} m")
    print(f"  Size Consistency Error: {quality['size_consistency_error']:.3f} m")
    print(f"    Length: {quality['length_error']:.3f} m")
    print(f"    Width: {quality['width_error']:.3f} m")
    print(f"    Height: {quality['height_error']:.3f} m")
    print(f"  Orientation Error: {quality['orientation_error_deg']:.2f}°")
    
    print("\nInterpretation:")
    print("  - Agreement > 90%: Good sensor calibration")
    print("  - Spatial error < 0.3m: Well-aligned sensors")
    print("  - Low size/orientation errors: Consistent detections")


def example_6_temporal_consistency():
    """Example 6: Temporal Consistency Evaluation."""
    print_section("Example 6: Temporal Consistency")
    
    np.random.seed(42)
    
    # Generate detection sequence (20 frames at 10 Hz)
    fps = 10.0
    num_frames = 20
    
    # Simulate objects moving smoothly
    detections_sequence = []
    for t in range(num_frames):
        # Base objects
        num_objects = 8 + np.random.randint(-2, 3)  # Slight variation
        
        detections = []
        for obj_id in range(num_objects):
            # Object moves smoothly
            x = obj_id * 5 + t * 0.5 + np.random.randn() * 0.1
            y = np.random.randn() * 2
            z = 0
            
            detections.append([x, y, z, 4.5, 2.0, 1.8, 0.0])
        
        detections_sequence.append(np.array(detections))
    
    quality = temporal_consistency(detections_sequence, fps=fps)
    
    print("Temporal Consistency Metrics:")
    print(f"  Detection Count (Mean ± Std): {quality['detection_count_mean']:.1f} ± {quality['detection_count_std']:.1f}")
    print(f"  Detection Count Variance: {quality['detection_count_variance']:.2f}")
    print(f"\n  Frame-to-Frame Consistency: {quality['frame_to_frame_consistency']:.2%}")
    print(f"  Consistency Std: {quality['consistency_std']:.3f}")
    print(f"\n  Flicker Rate: {quality['flicker_rate']:.2%}")
    if 'avg_new_detections_per_frame' in quality:
        print(f"  Avg New Detections/Frame: {quality['avg_new_detections_per_frame']:.2f}")
    if 'avg_lost_detections_per_frame' in quality:
        print(f"  Avg Lost Detections/Frame: {quality['avg_lost_detections_per_frame']:.2f}")
    
    print("\nInterpretation:")
    print("  - High frame consistency: Stable detections")
    print("  - Low flicker rate: No spurious appearances/disappearances")
    print("  - Low variance: Consistent scene representation")


def example_7_sim2real_gap():
    """Example 7: Perception Sim-to-Real Gap."""
    print_section("Example 7: Perception Sim-to-Real Performance Gap")
    
    np.random.seed(42)
    
    # Simulate detection results on 10 scenes
    sim_results = []
    real_results = []
    
    for scene_id in range(10):
        # Ground truth objects
        num_gt = 5 + np.random.randint(-2, 3)
        ground_truth = np.random.randn(num_gt, 7) * 10
        ground_truth[:, 2] = 0  # z = 0
        
        # Simulation: Higher recall, more false positives
        num_sim_det = num_gt + np.random.randint(0, 3)
        sim_predictions = ground_truth[:num_sim_det].copy()
        sim_predictions += np.random.randn(*sim_predictions.shape) * 0.3
        
        # Real world: Lower recall, fewer false positives, more noise
        num_real_det = max(1, num_gt - np.random.randint(0, 2))
        real_predictions = ground_truth[:num_real_det].copy()
        real_predictions += np.random.randn(*real_predictions.shape) * 0.8
        
        sim_results.append({
            'predictions': sim_predictions,
            'ground_truth': ground_truth
        })
        
        real_results.append({
            'predictions': real_predictions,
            'ground_truth': ground_truth
        })
    
    gap = perception_sim2real_gap(sim_results, real_results)
    
    print("Sim-to-Real Performance Gap:")
    print(f"\nSimulation Performance:")
    print(f"  Precision: {gap['precision_sim']:.2%}")
    print(f"  Recall: {gap['recall_sim']:.2%}")
    print(f"  F1-Score: {gap['f1_sim']:.2%}")
    
    print(f"\nReal-World Performance:")
    print(f"  Precision: {gap['precision_real']:.2%}")
    print(f"  Recall: {gap['recall_real']:.2%}")
    print(f"  F1-Score: {gap['f1_real']:.2%}")
    
    print(f"\nPerformance Gaps:")
    print(f"  Precision Gap: {gap['precision_gap']:+.2%}")
    print(f"  Recall Gap: {gap['recall_gap']:+.2%}")
    print(f"  F1 Gap: {gap['f1_gap']:+.2%}")
    print(f"  Performance Drop: {gap['performance_drop_pct']:.1f}%")
    
    print("\nInterpretation:")
    print("  - Positive gap: Sim performs better (common)")
    print("  - Performance drop: Expected degradation in real world")
    print("  - Large gaps (>20%): Significant sim-to-real domain shift")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Simulation Quality Metrics - Usage Examples")
    print("=" * 80)
    
    example_1_camera_quality()
    example_2_lidar_quality()
    example_3_radar_quality()
    example_4_sensor_noise()
    example_5_multimodal_alignment()
    example_6_temporal_consistency()
    example_7_sim2real_gap()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("\n")
    print("Key Takeaways for Simulation Quality:")
    print("  1. Camera: PSNR, color distribution, brightness/contrast matching")
    print("  2. LiDAR: Chamfer distance, point density, spatial distributions")
    print("  3. Radar: Detection density, velocity/RCS characteristics")
    print("  4. Noise: Std ratio, KS test, SNR comparison")
    print("  5. Calibration: Multimodal alignment, spatial consistency")
    print("  6. Temporal: Frame-to-frame consistency, flicker rate")
    print("  7. Sim2Real: Precision/recall gaps, performance degradation")
    print("=" * 80 + "\n")
