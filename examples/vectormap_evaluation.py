"""
Vector Map Detection Evaluation Examples.

Demonstrates evaluation of HD map vector extraction from sensor data.
Use cases:
- Lane line detection (centerlines, boundaries)
- Road boundary detection
- Crosswalk detection
- Topology estimation (lane connectivity)

Benchmarks: nuScenes Map, Argoverse 2, OpenLane-V2
"""

import numpy as np
from admetrics.vectormap import (
    chamfer_distance_polyline,
    frechet_distance,
    polyline_iou,
    lane_detection_metrics,
    topology_metrics,
    endpoint_error,
    direction_accuracy,
    vectormap_ap
)


def example_1_chamfer_distance():
    """Example 1: Evaluate polyline geometry with Chamfer distance."""
    print("=" * 70)
    print("Example 1: Chamfer Distance for Polyline Matching")
    print("=" * 70)
    
    # Predicted lane centerline
    pred_lane = np.array([
        [0, 0],
        [10, 0.2],
        [20, 0.1],
        [30, 0.3]
    ])
    
    # Ground truth lane centerline
    gt_lane = np.array([
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0]
    ])
    
    # Calculate Chamfer distance
    result = chamfer_distance_polyline(pred_lane, gt_lane, max_distance=0.5)
    
    print(f"\nChamfer Distance: {result['chamfer_distance']:.3f}m")
    print(f"  Pred -> GT: {result['chamfer_pred_to_gt']:.3f}m")
    print(f"  GT -> Pred: {result['chamfer_gt_to_pred']:.3f}m")
    print(f"  Precision (within 0.5m): {result['precision']:.1%}")
    print(f"  Recall (within 0.5m): {result['recall']:.1%}")
    print("\nInterpretation: Low Chamfer distance indicates accurate geometry")
    print()


def example_2_frechet_distance():
    """Example 2: Evaluate curve similarity with Fréchet distance."""
    print("=" * 70)
    print("Example 2: Fréchet Distance for Curve Similarity")
    print("=" * 70)
    
    # Predicted curved lane (quarter circle)
    theta_pred = np.linspace(0, np.pi/2, 15)
    pred_curve = np.column_stack([
        20 * np.cos(theta_pred),
        20 * np.sin(theta_pred)
    ])
    
    # Ground truth (slightly different radius)
    theta_gt = np.linspace(0, np.pi/2, 15)
    gt_curve = np.column_stack([
        21 * np.cos(theta_gt),
        21 * np.sin(theta_gt)
    ])
    
    # Calculate Fréchet distance
    dist = frechet_distance(pred_curve, gt_curve)
    
    print(f"\nFréchet Distance: {dist:.3f}m")
    print("\nInterpretation:")
    print(f"  - Measures curve similarity considering point ordering")
    print(f"  - {dist:.2f}m indicates {'good' if dist < 1.5 else 'poor'} curve matching")
    print(f"  - Better than Chamfer for curved lanes")
    print()


def example_3_polyline_iou():
    """Example 3: Lane overlap evaluation with Polyline IoU."""
    print("=" * 70)
    print("Example 3: Polyline IoU for Lane Overlap")
    print("=" * 70)
    
    # Predicted lane boundary
    pred_boundary = np.array([
        [0, 0],
        [50, 0.5],
        [100, 0.3]
    ])
    
    # Ground truth lane boundary
    gt_boundary = np.array([
        [0, 0.2],
        [50, 0.3],
        [100, 0.5]
    ])
    
    # Calculate IoU with lane width tolerance
    iou_narrow = polyline_iou(pred_boundary, gt_boundary, width=0.5, num_samples=50)
    iou_wide = polyline_iou(pred_boundary, gt_boundary, width=1.5, num_samples=50)
    
    print(f"\nPolyline IoU:")
    print(f"  Width=0.5m: {iou_narrow:.1%}")
    print(f"  Width=1.5m: {iou_wide:.1%}")
    print("\nInterpretation:")
    print(f"  - Width represents lane tolerance")
    print(f"  - Higher width → more lenient matching")
    print(f"  - Use for evaluating lane boundary accuracy")
    print()


def example_4_lane_detection():
    """Example 4: Complete lane detection evaluation."""
    print("=" * 70)
    print("Example 4: Lane Detection Metrics (Precision, Recall, F1)")
    print("=" * 70)
    
    # Predicted lanes (3 lanes detected)
    pred_lanes = [
        np.array([[0, 0], [100, 0]]),      # Lane 1 - correct
        np.array([[0, 3.5], [100, 3.6]]),  # Lane 2 - correct
        np.array([[0, 7], [100, 7.2]]),    # Lane 3 - correct
        np.array([[0, 10.5], [100, 11]])   # Lane 4 - false positive
    ]
    
    # Ground truth lanes (4 lanes)
    gt_lanes = [
        np.array([[0, 0.1], [100, 0.1]]),    # Lane 1
        np.array([[0, 3.6], [100, 3.5]]),    # Lane 2
        np.array([[0, 7.1], [100, 7.0]]),    # Lane 3
        np.array([[0, 14], [100, 14.1]])     # Lane 4 - missed detection
    ]
    
    # Evaluate detection
    metrics = lane_detection_metrics(pred_lanes, gt_lanes, distance_threshold=1.0)
    
    print(f"\nLane Detection Performance:")
    print(f"  Precision: {metrics['precision']:.1%} ({metrics['tp']}/{metrics['tp'] + metrics['fp']} correct)")
    print(f"  Recall: {metrics['recall']:.1%} ({metrics['tp']}/{metrics['tp'] + metrics['fn']} detected)")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"\nDetection Summary:")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"  Mean Chamfer Distance (matched): {metrics['chamfer_distance']:.3f}m")
    print("\nInterpretation:")
    print(f"  - {metrics['tp']} lanes correctly detected")
    print(f"  - {metrics['fp']} spurious detection(s)")
    print(f"  - {metrics['fn']} lane(s) missed")
    print()


def example_5_topology():
    """Example 5: Evaluate lane topology and connectivity."""
    print("=" * 70)
    print("Example 5: Lane Topology Evaluation")
    print("=" * 70)
    
    # Predicted topology
    pred_topology = {
        'successors': [1, 2],      # Lane 0 connects to lanes 1, 2
        'predecessors': [],
        'left_neighbor': [3],
        'right_neighbor': []
    }
    
    # Ground truth topology
    gt_topology = {
        'successors': [1, 2],
        'predecessors': [],
        'left_neighbor': [3],
        'right_neighbor': [4]      # Missed connection
    }
    
    # Lane matching (pred_id -> gt_id)
    lane_matches = {0: 0, 1: 1, 2: 2, 3: 3}
    
    # Evaluate topology
    metrics = topology_metrics(pred_topology, gt_topology, lane_matches)
    
    print(f"\nTopology Performance:")
    print(f"  Precision: {metrics['topology_precision']:.1%}")
    print(f"  Recall: {metrics['topology_recall']:.1%}")
    print(f"  F1 Score: {metrics['topology_f1']:.3f}")
    print(f"\nConnection Summary:")
    print(f"  Correct: {metrics['correct_connections']}")
    print(f"  Predicted: {metrics['pred_connections']}")
    print(f"  Ground Truth: {metrics['gt_connections']}")
    print("\nInterpretation:")
    print(f"  - Topology captures lane-to-lane relationships")
    print(f"  - Critical for path planning and navigation")
    print(f"  - {metrics['topology_recall']:.0%} of connections detected")
    print()


def example_6_endpoint_and_direction():
    """Example 6: Evaluate endpoint and direction accuracy."""
    print("=" * 70)
    print("Example 6: Endpoint and Direction Accuracy")
    print("=" * 70)
    
    # Predicted lane merging scenario
    pred_merge = np.array([
        [0, 0],
        [25, 0.5],
        [50, 1.2],
        [75, 1.8],
        [100, 2.1]
    ])
    
    # Ground truth
    gt_merge = np.array([
        [0, 0.2],
        [25, 0.6],
        [50, 1.0],
        [75, 1.6],
        [100, 2.0]
    ])
    
    # Endpoint accuracy
    ep_errors = endpoint_error(pred_merge, gt_merge)
    
    print(f"\nEndpoint Errors:")
    print(f"  Start Point Error: {ep_errors['start_error']:.3f}m")
    print(f"  End Point Error: {ep_errors['end_error']:.3f}m")
    print(f"  Mean Endpoint Error: {ep_errors['mean_endpoint_error']:.3f}m")
    
    # Direction accuracy
    dir_acc = direction_accuracy(pred_merge, gt_merge, num_samples=10)
    
    print(f"\nDirection Accuracy:")
    print(f"  Mean Direction Error: {dir_acc['mean_direction_error_deg']:.2f}°")
    print(f"  Direction Accuracy (< 15°): {dir_acc['direction_accuracy']:.1%}")
    
    print("\nInterpretation:")
    print(f"  - Endpoint error measures merge point accuracy")
    print(f"  - Direction error measures heading alignment")
    print(f"  - Both critical for lane-level localization")
    print()


def example_7_average_precision():
    """Example 7: Calculate Average Precision for vector map detection."""
    print("=" * 70)
    print("Example 7: Average Precision (AP) for Vector Map Detection")
    print("=" * 70)
    
    # Predicted lanes with confidence scores (sorted by confidence)
    pred_lanes = [
        {'polyline': np.array([[0, 0], [100, 0]]), 'score': 0.95},      # TP
        {'polyline': np.array([[0, 3.5], [100, 3.6]]), 'score': 0.92},  # TP
        {'polyline': np.array([[0, 7], [100, 7.1]]), 'score': 0.88},    # TP
        {'polyline': np.array([[0, 10.5], [100, 11]]), 'score': 0.75},  # FP (no match)
        {'polyline': np.array([[0, 14], [100, 14.2]]), 'score': 0.65},  # TP
        {'polyline': np.array([[0, 20], [100, 20.5]]), 'score': 0.55},  # FP
    ]
    
    # Ground truth lanes (4 lanes)
    gt_lanes = [
        {'polyline': np.array([[0, 0.1], [100, 0.1]])},
        {'polyline': np.array([[0, 3.6], [100, 3.5]])},
        {'polyline': np.array([[0, 7.1], [100, 7.0]])},
        {'polyline': np.array([[0, 14.1], [100, 14.0]])},
    ]
    
    # Calculate AP at multiple thresholds
    ap_results = vectormap_ap(
        pred_lanes,
        gt_lanes,
        distance_thresholds=[0.5, 1.0, 1.5]
    )
    
    print(f"\nAverage Precision Results:")
    print(f"  AP @ 0.5m: {ap_results['ap_0.5']:.1%}")
    print(f"  AP @ 1.0m: {ap_results['ap_1.0']:.1%}")
    print(f"  AP @ 1.5m: {ap_results['ap_1.5']:.1%}")
    print(f"  mAP: {ap_results['map']:.1%}")
    
    print("\nInterpretation:")
    print(f"  - AP measures detection quality across confidence levels")
    print(f"  - Higher threshold = stricter geometric accuracy")
    print(f"  - mAP = {ap_results['map']:.1%} summarizes overall performance")
    print(f"  - Similar to object detection AP but uses Chamfer distance")
    print()


def example_8_complete_evaluation():
    """Example 8: Complete vector map evaluation pipeline."""
    print("=" * 70)
    print("Example 8: Complete Vector Map Evaluation Pipeline")
    print("=" * 70)
    
    # Simulate a complex intersection scenario
    # 6 lanes: 3 straight, 2 merging, 1 splitting
    
    pred_lanes = [
        np.array([[0, 0], [100, 0.2]]),         # Lane 1
        np.array([[0, 3.5], [100, 3.6]]),       # Lane 2
        np.array([[0, 7], [50, 7], [100, 10]]), # Lane 3 (merging)
        np.array([[0, 10.5], [100, 10.7]]),     # Lane 4
    ]
    
    gt_lanes = [
        np.array([[0, 0], [100, 0]]),
        np.array([[0, 3.5], [100, 3.5]]),
        np.array([[0, 7], [50, 7.2], [100, 10]]),
        np.array([[0, 10.5], [100, 10.5]]),
        np.array([[0, 14], [100, 14]]),  # Missed lane
    ]
    
    print("\n1. DETECTION QUALITY")
    print("-" * 70)
    det_metrics = lane_detection_metrics(pred_lanes, gt_lanes, distance_threshold=1.0)
    print(f"   Precision: {det_metrics['precision']:.1%}")
    print(f"   Recall: {det_metrics['recall']:.1%}")
    print(f"   F1 Score: {det_metrics['f1_score']:.3f}")
    
    print("\n2. GEOMETRIC ACCURACY")
    print("-" * 70)
    chamfer_distances = []
    for i, (pred, gt) in enumerate(zip(pred_lanes[:4], gt_lanes[:4])):
        result = chamfer_distance_polyline(pred, gt)
        chamfer_distances.append(result['chamfer_distance'])
        print(f"   Lane {i+1} Chamfer: {result['chamfer_distance']:.3f}m")
    print(f"   Mean Chamfer: {np.mean(chamfer_distances):.3f}m")
    
    print("\n3. ENDPOINT ACCURACY")
    print("-" * 70)
    for i, (pred, gt) in enumerate(zip(pred_lanes[:4], gt_lanes[:4])):
        ep_error = endpoint_error(pred, gt)
        print(f"   Lane {i+1}: Start={ep_error['start_error']:.2f}m, End={ep_error['end_error']:.2f}m")
    
    print("\n4. DIRECTION ACCURACY")
    print("-" * 70)
    direction_accs = []
    for i, (pred, gt) in enumerate(zip(pred_lanes[:4], gt_lanes[:4])):
        dir_acc = direction_accuracy(pred, gt)
        direction_accs.append(dir_acc['direction_accuracy'])
        print(f"   Lane {i+1}: {dir_acc['mean_direction_error_deg']:.1f}° error, "
              f"{dir_acc['direction_accuracy']:.0%} accurate")
    
    print("\n5. OVERALL ASSESSMENT")
    print("-" * 70)
    # Create predictions with scores for AP
    pred_with_scores = [{'polyline': p, 'score': 0.9 - i*0.1} for i, p in enumerate(pred_lanes)]
    gt_with_format = [{'polyline': g} for g in gt_lanes]
    ap_results = vectormap_ap(pred_with_scores, gt_with_format, distance_thresholds=[1.0])
    
    print(f"   Detection F1: {det_metrics['f1_score']:.3f}")
    print(f"   Mean Geometric Error: {np.mean(chamfer_distances):.3f}m")
    print(f"   Mean Direction Accuracy: {np.mean(direction_accs):.1%}")
    print(f"   AP @ 1.0m: {ap_results['ap_1.0']:.1%}")
    
    # Overall score (weighted combination)
    overall_score = (
        0.3 * det_metrics['f1_score'] +
        0.3 * (1.0 - min(np.mean(chamfer_distances) / 2.0, 1.0)) +
        0.2 * np.mean(direction_accs) +
        0.2 * ap_results['ap_1.0']
    )
    print(f"\n   Overall Vector Map Score: {overall_score:.1%}")
    print()


def main():
    """Run all examples."""
    example_1_chamfer_distance()
    example_2_frechet_distance()
    example_3_polyline_iou()
    example_4_lane_detection()
    example_5_topology()
    example_6_endpoint_and_direction()
    example_7_average_precision()
    example_8_complete_evaluation()
    
    print("=" * 70)
    print("All vector map evaluation examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Chamfer distance: Primary metric for polyline geometry")
    print("  2. Fréchet distance: Better for curved lanes (considers ordering)")
    print("  3. Lane detection metrics: Precision, recall, F1 for detection quality")
    print("  4. Topology metrics: Evaluate lane connectivity and relationships")
    print("  5. Endpoint/direction: Critical for lane merging and localization")
    print("  6. Average Precision: Overall detection quality with confidence")
    print("\nBenchmarks:")
    print("  - nuScenes Map Expansion Challenge")
    print("  - Argoverse 2 HD Map Challenge")
    print("  - OpenLane-V2 (topology-aware)")
    print("  - MapTR, VectorMapNet, HDMapNet")
    print()


if __name__ == '__main__':
    main()
