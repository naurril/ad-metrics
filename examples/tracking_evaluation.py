"""
Multi-Object Tracking Metrics - Usage Examples

Demonstrates how to evaluate 3D multi-object tracking using various metrics:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision)
- HOTA (Higher Order Tracking Accuracy)
- IDF1 (ID F1 Score)
"""

import numpy as np
from admetrics.tracking import (
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_id_f1
)


def create_sample_tracking_sequence():
    """
    Create a sample tracking sequence with 3 objects tracked across 5 frames.
    
    Returns:
        predictions: Dict[frame_id -> List[detection]]
        ground_truth: Dict[frame_id -> List[detection]]
    """
    # Create predictions (with some errors and ID switches)
    predictions = {
        0: [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'},
            {'box': [20, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 3, 'class': 'pedestrian'},
        ],
        1: [
            {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
            {'box': [11, 0, 0, 4, 2, 1.5, 0], 'track_id': 2, 'class': 'car'},
            {'box': [21, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 3, 'class': 'pedestrian'},
            {'box': [30, 0, 0, 4, 2, 1.5, 0], 'track_id': 4, 'class': 'car'},  # False positive
        ],
        2: [
            {'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
            {'box': [12, 0, 0, 4, 2, 1.5, 0], 'track_id': 5, 'class': 'car'},  # ID switch!
            {'box': [22, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 3, 'class': 'pedestrian'},
        ],
        3: [
            {'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
            {'box': [13, 0, 0, 4, 2, 1.5, 0], 'track_id': 5, 'class': 'car'},
            # Pedestrian lost (fragmentation starts)
        ],
        4: [
            {'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 1, 'class': 'car'},
            {'box': [14, 0, 0, 4, 2, 1.5, 0], 'track_id': 5, 'class': 'car'},
            {'box': [24, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 3, 'class': 'pedestrian'},  # Recovered
        ],
    }
    
    # Ground truth (perfect trajectories)
    ground_truth = {
        0: [
            {'box': [0, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
            {'box': [10, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'},
            {'box': [20, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 102, 'class': 'pedestrian'},
        ],
        1: [
            {'box': [1, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
            {'box': [11, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'},
            {'box': [21, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 102, 'class': 'pedestrian'},
        ],
        2: [
            {'box': [2, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
            {'box': [12, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'},
            {'box': [22, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 102, 'class': 'pedestrian'},
        ],
        3: [
            {'box': [3, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
            {'box': [13, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'},
            {'box': [23, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 102, 'class': 'pedestrian'},
        ],
        4: [
            {'box': [4, 0, 0, 4, 2, 1.5, 0], 'track_id': 100, 'class': 'car'},
            {'box': [14, 0, 0, 4, 2, 1.5, 0], 'track_id': 101, 'class': 'car'},
            {'box': [24, 0, 0, 2, 1.5, 1.8, 0], 'track_id': 102, 'class': 'pedestrian'},
        ],
    }
    
    return predictions, ground_truth


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def main():
    """Run tracking metrics examples."""
    
    print("="*80)
    print("Multi-Object Tracking Metrics - Usage Examples")
    print("="*80)
    
    # Create sample data
    predictions, ground_truth = create_sample_tracking_sequence()
    
    print_section("Example 1: CLEAR MOT Metrics (MOTA & MOTP)")
    
    # Calculate MOTA and related metrics
    mota_results = calculate_multi_frame_mota(
        predictions,
        ground_truth,
        iou_threshold=0.5
    )
    
    print("CLEAR MOT Metrics:")
    print(f"  MOTA (Multiple Object Tracking Accuracy): {mota_results['mota']:.4f}")
    print(f"  MOTP (Multiple Object Tracking Precision): {mota_results['motp']:.4f}")
    print()
    print("Detection Counts:")
    print(f"  True Positives (Matches):     {mota_results['num_matches']}")
    print(f"  False Positives:              {mota_results['num_false_positives']}")
    print(f"  Misses (False Negatives):     {mota_results['num_misses']}")
    print(f"  ID Switches:                  {mota_results['num_switches']}")
    print(f"  Fragmentations:               {mota_results['num_fragmentations']}")
    print(f"  Total Ground Truth Objects:   {mota_results['total_gt']}")
    print()
    print("Quality Metrics:")
    print(f"  Precision: {mota_results['precision']:.4f}")
    print(f"  Recall:    {mota_results['recall']:.4f}")
    print()
    print("Trajectory-Level Metrics:")
    print(f"  Mostly Tracked (≥80%):   {mota_results['mostly_tracked']} trajectories")
    print(f"  Partially Tracked (20-80%): {mota_results['partially_tracked']} trajectories")
    print(f"  Mostly Lost (<20%):      {mota_results['mostly_lost']} trajectories")
    print(f"  Total GT Trajectories:   {mota_results['num_gt_trajectories']}")
    
    print_section("Example 2: HOTA (Higher Order Tracking Accuracy)")
    
    # Calculate HOTA
    hota_results = calculate_hota(
        predictions,
        ground_truth,
        iou_threshold=0.5
    )
    
    print("HOTA Metrics:")
    print(f"  HOTA (Higher Order Tracking Accuracy): {hota_results['hota']:.4f}")
    print(f"  DetA (Detection Accuracy):             {hota_results['det_a']:.4f}")
    print(f"  AssA (Association Accuracy):           {hota_results['ass_a']:.4f}")
    print()
    print("Note: HOTA = sqrt(DetA × AssA)")
    print(f"      Verification: sqrt({hota_results['det_a']:.4f} × {hota_results['ass_a']:.4f}) = {np.sqrt(hota_results['det_a'] * hota_results['ass_a']):.4f}")
    print()
    print("Detection Breakdown:")
    print(f"  True Positives:  {hota_results['tp']}")
    print(f"  False Positives: {hota_results['fp']}")
    print(f"  False Negatives: {hota_results['fn']}")
    
    print_section("Example 3: IDF1 (ID F1 Score)")
    
    # Calculate ID-based metrics
    id_results = calculate_id_f1(
        predictions,
        ground_truth,
        iou_threshold=0.5
    )
    
    print("ID-Based Metrics:")
    print(f"  IDF1 (ID F1 Score):     {id_results['idf1']:.4f}")
    print(f"  IDP (ID Precision):     {id_results['idp']:.4f}")
    print(f"  IDR (ID Recall):        {id_results['idr']:.4f}")
    print()
    print("ID Assignment Counts:")
    print(f"  Correct ID Assignments (IDTP): {id_results['idtp']}")
    print(f"  Incorrect IDs (IDFP):          {id_results['idfp']}")
    print(f"  Missed IDs (IDFN):             {id_results['idfn']}")
    print()
    print("Note: IDF1 measures how well track identities are preserved.")
    print("      Low IDF1 indicates frequent ID switches.")
    
    print_section("Example 4: Metrics Comparison")
    
    print("Summary Table:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Value':<15} {'Interpretation':<25}")
    print("-" * 70)
    print(f"{'MOTA':<30} {mota_results['mota']:>6.4f}        {'Overall tracking quality':<25}")
    print(f"{'MOTP':<30} {mota_results['motp']:>6.4f}        {'Localization accuracy':<25}")
    print(f"{'HOTA':<30} {hota_results['hota']:>6.4f}        {'Balanced det + assoc':<25}")
    print(f"{'IDF1':<30} {id_results['idf1']:>6.4f}        {'ID consistency':<25}")
    print(f"{'Precision':<30} {mota_results['precision']:>6.4f}        {'Detection precision':<25}")
    print(f"{'Recall':<30} {mota_results['recall']:>6.4f}        {'Detection recall':<25}")
    print("-" * 70)
    
    print_section("Example 5: Error Analysis")
    
    print("Error Breakdown:")
    print(f"  False Positives:   {mota_results['num_false_positives']} detections")
    print(f"  Missed Detections: {mota_results['num_misses']} objects")
    print(f"  ID Switches:       {mota_results['num_switches']} switches")
    print(f"  Fragmentations:    {mota_results['num_fragmentations']} fragments")
    print()
    
    # Calculate error rates
    total_errors = (mota_results['num_false_positives'] + 
                   mota_results['num_misses'] + 
                   mota_results['num_switches'])
    
    if total_errors > 0:
        fp_rate = mota_results['num_false_positives'] / total_errors
        miss_rate = mota_results['num_misses'] / total_errors
        idsw_rate = mota_results['num_switches'] / total_errors
        
        print("Error Distribution:")
        print(f"  False Positives:   {fp_rate*100:>5.1f}%")
        print(f"  Missed Detections: {miss_rate*100:>5.1f}%")
        print(f"  ID Switches:       {idsw_rate*100:>5.1f}%")
    
    print_section("Key Insights")
    
    print("Metric Interpretations:")
    print()
    print("MOTA (Multiple Object Tracking Accuracy):")
    print("  - Ranges from -∞ to 1.0 (can be negative with many errors)")
    print("  - Penalizes false positives, missed detections, and ID switches")
    print("  - Higher is better")
    print()
    print("MOTP (Multiple Object Tracking Precision):")
    print("  - Average localization error for matched detections")
    print("  - Measured in same units as coordinates (meters)")
    print("  - Lower is better")
    print()
    print("HOTA (Higher Order Tracking Accuracy):")
    print("  - Balanced metric combining detection and association")
    print("  - Ranges from 0 to 1")
    print("  - More intuitive than MOTA, especially for comparing trackers")
    print()
    print("IDF1 (ID F1 Score):")
    print("  - F1 score for identity preservation")
    print("  - Ranges from 0 to 1")
    print("  - Directly measures ID switch performance")
    
    print("\n" + "="*80)
    print("All tracking examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
