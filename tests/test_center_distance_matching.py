import pytest
from admetrics.detection.ap import calculate_map_multi_frame


def test_center_distance_all_match():
    # Two frames, one GT per frame and one prediction exactly near the GT center
    predictions = {
        0: [{'box': [0.0, 0.0, 0, 0, 0, 0, 0], 'score': 0.9, 'class': 'car'}],
        1: [{'box': [10.0, 0.0, 0, 0, 0, 0, 0], 'score': 0.8, 'class': 'car'}]
    }

    ground_truth = {
        0: [{'box': [0.0, 0.0, 0, 0, 0, 0, 0], 'class': 'car'}],
        1: [{'box': [10.0, 0.0, 0, 0, 0, 0, 0], 'class': 'car'}]
    }

    thresholds = [0.5, 1.0, 2.0, 4.0]
    res = calculate_map_multi_frame(
        predictions_per_frame=predictions,
        ground_truth_per_frame=ground_truth,
        class_names=['car'],
        iou_thresholds=thresholds,
        matching='center_distance'
    )

    # All matches should be perfect across all thresholds
    assert pytest.approx(res['mAP'], rel=1e-6) == 1.0
    for thr in thresholds:
        assert pytest.approx(res['AP_per_threshold'][float(thr)], rel=1e-6) == 1.0
    assert pytest.approx(res['AP_per_class']['car'], rel=1e-6) == 1.0


def test_center_distance_mixed_thresholds():
    # Single frame, prediction slightly outside 0.5m but inside 1.0m threshold
    predictions = {
        0: [{'box': [0.6, 0.0, 0, 0, 0, 0, 0], 'score': 0.9, 'class': 'car'}]
    }

    ground_truth = {
        0: [{'box': [0.0, 0.0, 0, 0, 0, 0, 0], 'class': 'car'}]
    }

    thresholds = [0.5, 1.0]
    res = calculate_map_multi_frame(
        predictions_per_frame=predictions,
        ground_truth_per_frame=ground_truth,
        class_names=['car'],
        iou_thresholds=thresholds,
        matching='center_distance'
    )

    # For threshold 0.5 -> miss (AP=0), for threshold 1.0 -> hit (AP=1), so mean = 0.5
    assert pytest.approx(res['AP_per_threshold'][0.5], rel=1e-6) == 0.0
    assert pytest.approx(res['AP_per_threshold'][1.0], rel=1e-6) == 1.0
    assert pytest.approx(res['mAP'], rel=1e-6) == 0.5
