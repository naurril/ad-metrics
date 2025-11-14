"""
Trajectory Prediction Metrics - Usage Examples

Demonstrates how to evaluate trajectory prediction models using various metrics:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- Miss Rate
- Multi-modal metrics
- Probabilistic metrics (Brier-FDE, NLL)
- Safety metrics (collision rate, drivable area compliance)
"""

import numpy as np
from admetrics.trajectory import (
    calculate_ade,
    calculate_fde,
    calculate_miss_rate,
    calculate_multimodal_ade,
    calculate_multimodal_fde,
    calculate_brier_fde,
    calculate_nll,
    calculate_trajectory_metrics,
    calculate_collision_rate,
    calculate_drivable_area_compliance,
)


def create_sample_trajectories():
    """
    Create sample predicted and ground truth trajectories.
    
    Scenario: Predicting a vehicle's path over the next 3 seconds (30 timesteps).
    """
    # Ground truth: smooth arc
    t = np.linspace(0, 3, 30)
    gt_x = t * 5  # 5 m/s forward
    gt_y = 0.5 * t**2  # Slight left turn
    ground_truth = np.stack([gt_x, gt_y], axis=1)
    
    # Single-modal prediction: slightly off
    pred_x = t * 5 + 0.2  # 0.2m offset
    pred_y = 0.5 * t**2 + 0.1 * t
    prediction = np.stack([pred_x, pred_y], axis=1)
    
    # Multi-modal predictions: 3 possible futures
    multimodal_preds = np.array([
        # Mode 0: Continues straight
        np.stack([t * 5, t * 0.2], axis=1),
        # Mode 1: Turns left (close to GT)
        np.stack([t * 5, 0.5 * t**2], axis=1),
        # Mode 2: Turns right
        np.stack([t * 5, -0.3 * t**2], axis=1),
    ])
    
    # Mode probabilities
    mode_probs = np.array([0.3, 0.5, 0.2])  # Most confident in left turn
    
    return prediction, ground_truth, multimodal_preds, mode_probs


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def main():
    """Run trajectory prediction metrics examples."""
    
    print("="*80)
    print("Trajectory Prediction Metrics - Usage Examples")
    print("="*80)
    
    # Create sample data
    prediction, ground_truth, multimodal_preds, mode_probs = create_sample_trajectories()
    
    print_section("Example 1: Basic Displacement Metrics (ADE & FDE)")
    
    # Calculate ADE and FDE
    ade = calculate_ade(prediction, ground_truth)
    fde = calculate_fde(prediction, ground_truth)
    
    print("Single-Modal Prediction Metrics:")
    print(f"  ADE (Average Displacement Error):  {ade:.4f} meters")
    print(f"  FDE (Final Displacement Error):    {fde:.4f} meters")
    print()
    print("Interpretation:")
    print(f"  - On average, predictions are {ade:.2f}m away from ground truth")
    print(f"  - Final predicted position is {fde:.2f}m from actual final position")
    
    print_section("Example 2: Miss Rate")
    
    # Calculate miss rate at different thresholds
    thresholds = [1.0, 2.0, 4.0]
    
    print("Miss Rate at Different Thresholds:")
    for threshold in thresholds:
        result = calculate_miss_rate(prediction, ground_truth, threshold=threshold)
        status = "MISS" if result['is_miss'] else "HIT"
        print(f"  Threshold {threshold:.1f}m: {status} (FDE: {result['fde']:.4f}m)")
    print()
    print("Note: Miss rate of 1.0 means FDE exceeds threshold (prediction 'missed')")
    print("      Miss rate of 0.0 means FDE is within threshold (prediction 'hit')")
    
    print_section("Example 3: Multi-Modal Predictions")
    
    # Calculate multi-modal metrics
    mm_ade = calculate_multimodal_ade(multimodal_preds, ground_truth)
    mm_fde = calculate_multimodal_fde(multimodal_preds, ground_truth)
    
    print("Multi-Modal Prediction Metrics (3 modes):")
    print(f"  minADE (Best mode):     {mm_ade['min_ade']:.4f} meters")
    print(f"  meanADE (Average):      {mm_ade['mean_ade']:.4f} meters")
    print(f"  Best mode index:        {mm_ade['best_mode']}")
    print()
    print(f"  minFDE (Best mode):     {mm_fde['min_fde']:.4f} meters")
    print(f"  meanFDE (Average):      {mm_fde['mean_fde']:.4f} meters")
    print()
    print("Mode Descriptions:")
    print("  Mode 0: Continues straight")
    print("  Mode 1: Turns left (closest to ground truth)")
    print("  Mode 2: Turns right")
    
    print_section("Example 4: Probabilistic Metrics - Brier-FDE")
    
    # Calculate Brier-FDE with mode probabilities
    brier = calculate_brier_fde(multimodal_preds, ground_truth, mode_probs)
    
    print("Brier-FDE (Probability-Weighted FDE):")
    print(f"  Brier-FDE:              {brier['brier_fde']:.4f} meters")
    print(f"  minFDE (Best mode):     {brier['min_fde']:.4f} meters")
    print()
    print("Mode Probabilities:")
    for i, prob in enumerate(mode_probs):
        print(f"  Mode {i}: {prob:.2f}")
    print()
    print("Interpretation:")
    print("  - Brier-FDE weights each mode's error by its predicted probability")
    print("  - Lower Brier-FDE indicates well-calibrated confidence estimates")
    print(f"  - Here: {brier['brier_fde']:.2f}m vs best mode {brier['min_fde']:.2f}m")
    
    print_section("Example 5: Negative Log-Likelihood (NLL)")
    
    # Create covariance matrices (uncertainty estimates)
    num_modes, num_timesteps = multimodal_preds.shape[0], multimodal_preds.shape[1]
    
    # Mode 1 is most confident (small covariance), others less confident
    covs = np.zeros((num_modes, num_timesteps, 2, 2))
    for k in range(num_modes):
        for t in range(num_timesteps):
            if k == 1:  # Confident mode
                covs[k, t] = np.eye(2) * 0.1
            else:  # Less confident modes
                covs[k, t] = np.eye(2) * 0.5
    
    nll_result = calculate_nll(multimodal_preds, ground_truth, covs, mode_probs)
    
    print("Negative Log-Likelihood (NLL):")
    print(f"  NLL:                    {nll_result['nll']:.4f}")
    print(f"  Log-Likelihood:         {nll_result['log_likelihood']:.4f}")
    print(f"  Best mode:              {nll_result['best_mode']}")
    print()
    print("Interpretation:")
    print("  - NLL measures how well predictions explain the observed trajectory")
    print("  - Accounts for both prediction accuracy AND uncertainty estimates")
    print("  - Lower NLL is better (higher likelihood)")
    
    print_section("Example 6: Comprehensive Metrics")
    
    # Get all metrics at once for single-modal
    metrics_single = calculate_trajectory_metrics(
        prediction, ground_truth, 
        miss_threshold=2.0, 
        multimodal=False
    )
    
    print("Single-Modal Comprehensive Metrics:")
    for key, value in metrics_single.items():
        print(f"  {key:<15} {value}")
    
    # Get all metrics for multi-modal
    metrics_multi = calculate_trajectory_metrics(
        multimodal_preds, ground_truth,
        miss_threshold=2.0,
        multimodal=True
    )
    
    print()
    print("Multi-Modal Comprehensive Metrics:")
    for key, value in metrics_multi.items():
        print(f"  {key:<15} {value}")
    
    print_section("Example 7: Safety Metrics - Collision Detection")
    
    # Define static obstacles (e.g., parked cars, pedestrians)
    obstacles = [
        {'center': [8, 2], 'radius': 1.0},   # Parked car at (8, 2)
        {'center': [12, -1], 'radius': 0.5}, # Pedestrian at (12, -1)
    ]
    
    collision_result = calculate_collision_rate(
        prediction, obstacles, safety_margin=0.5
    )
    
    print("Collision Detection:")
    print(f"  Collision Rate:         {collision_result['collision_rate']:.2%}")
    print(f"  Number of Collisions:   {collision_result['num_collisions']}")
    print(f"  Total Timesteps:        {collision_result['total_timesteps']}")
    if collision_result['collision_timesteps']:
        print(f"  Collision Timesteps:    {collision_result['collision_timesteps']}")
    else:
        print(f"  Collision Timesteps:    None (Safe trajectory!)")
    
    print_section("Example 8: Drivable Area Compliance")
    
    # Define drivable area (e.g., road boundaries)
    drivable_area_rect = {
        'type': 'rectangle',
        'x_min': 0, 'x_max': 20,
        'y_min': -2, 'y_max': 10
    }
    
    compliance_result = calculate_drivable_area_compliance(
        prediction, drivable_area_rect
    )
    
    print("Drivable Area Compliance (Rectangle):")
    print(f"  Compliance Rate:        {compliance_result['compliance_rate']:.2%}")
    print(f"  Number of Violations:   {compliance_result['num_violations']}")
    print(f"  Total Timesteps:        {compliance_result['total_timesteps']}")
    if compliance_result['violation_timesteps']:
        print(f"  Violation Timesteps:    {compliance_result['violation_timesteps']}")
    else:
        print(f"  Violation Timesteps:    None (Fully compliant!)")
    
    # Polygon-based drivable area (e.g., lane boundaries)
    drivable_area_polygon = {
        'type': 'polygon',
        'vertices': [[0, -1], [20, -1], [20, 8], [15, 10], [0, 10]]
    }
    
    compliance_poly = calculate_drivable_area_compliance(
        prediction, drivable_area_polygon
    )
    
    print()
    print("Drivable Area Compliance (Polygon/Lane):")
    print(f"  Compliance Rate:        {compliance_poly['compliance_rate']:.2%}")
    print(f"  Number of Violations:   {compliance_poly['num_violations']}")
    
    print_section("Example 9: Metrics Comparison Table")
    
    print("Metrics Summary:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Value':<15} {'Unit':<25}")
    print("-" * 70)
    print(f"{'ADE':<30} {ade:>6.4f}        {'meters':<25}")
    print(f"{'FDE':<30} {fde:>6.4f}        {'meters':<25}")
    print(f"{'Miss Rate (@2m)':<30} {metrics_single['miss_rate']:>6.4f}        {'0=hit, 1=miss':<25}")
    print(f"{'minADE (multi-modal)':<30} {mm_ade['min_ade']:>6.4f}        {'meters':<25}")
    print(f"{'minFDE (multi-modal)':<30} {mm_fde['min_fde']:>6.4f}        {'meters':<25}")
    print(f"{'Brier-FDE':<30} {brier['brier_fde']:>6.4f}        {'meters (weighted)':<25}")
    print(f"{'NLL':<30} {nll_result['nll']:>6.4f}        {'nats':<25}")
    print(f"{'Collision Rate':<30} {collision_result['collision_rate']:>6.2%}        {'percentage':<25}")
    print(f"{'Drivable Area Compliance':<30} {compliance_result['compliance_rate']:>6.2%}        {'percentage':<25}")
    print("-" * 70)
    
    print_section("Key Insights")
    
    print("When to Use Each Metric:")
    print()
    print("ADE (Average Displacement Error):")
    print("  - Overall trajectory accuracy across all timesteps")
    print("  - Good for comparing general prediction quality")
    print("  - Sensitive to errors throughout the entire trajectory")
    print()
    print("FDE (Final Displacement Error):")
    print("  - End-point accuracy (where will vehicle be?)")
    print("  - Critical for planning and decision making")
    print("  - Often used with longer prediction horizons")
    print()
    print("Miss Rate:")
    print("  - Binary success/failure at a threshold")
    print("  - Useful for safety-critical applications")
    print("  - Common thresholds: 2m for vehicles, 0.5m for pedestrians")
    print()
    print("minADE/minFDE (Multi-modal):")
    print("  - Best-case performance across all predicted modes")
    print("  - Tests if model can predict correct future scenario")
    print("  - Standard for multi-modal benchmarks (Argoverse, nuScenes)")
    print()
    print("Brier-FDE:")
    print("  - Probability-weighted error")
    print("  - Evaluates confidence calibration")
    print("  - Penalizes over-confident wrong predictions")
    print()
    print("NLL (Negative Log-Likelihood):")
    print("  - Proper scoring rule for probabilistic predictions")
    print("  - Evaluates both accuracy and uncertainty quantification")
    print("  - Requires covariance/uncertainty estimates")
    print()
    print("Collision Rate:")
    print("  - Safety metric for obstacle avoidance")
    print("  - Critical for deployment readiness")
    print("  - Should be 0% for production systems")
    print()
    print("Drivable Area Compliance:")
    print("  - Legal/physical constraint satisfaction")
    print("  - Ensures predictions respect road geometry")
    print("  - Important for realistic trajectory generation")
    
    print("\n" + "="*80)
    print("All trajectory prediction examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
