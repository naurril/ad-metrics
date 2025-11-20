"""
Tests for multi-frame mAP calculation.
"""
import numpy as np
from admetrics.detection.ap import calculate_map_multi_frame


def _make_box(x):
    # simplified 3d box [x,y,z,w,h,l,yaw]
    return [x, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def test_calculate_map_multi_frame():
    # two frames, each with one GT and one correct prediction
    preds = {
        0: [{'box': _make_box(0), 'score': 0.9, 'class': 'car'}],
        1: [{'box': _make_box(10), 'score': 0.8, 'class': 'car'}],
    }
    gts = {
        0: [{'box': _make_box(0), 'class': 'car'}],
        1: [{'box': _make_box(10), 'class': 'car'}],
    }

    res = calculate_map_multi_frame(preds, gts, class_names=['car'])
    assert 'mAP' in res
    # Allow small numerical tolerance from interpolation
    assert abs(res['mAP'] - 1.0) < 1e-6

    # construct a slightly more complex scenario
    preds = {
        0: [
            {'box': _make_box(0), 'score': 0.6, 'class': 'car'},
        ],
        1: [
            {'box': _make_box(10), 'score': 0.9, 'class': 'car'},
            {'box': _make_box(11), 'score': 0.95, 'class': 'car'},  # should be FP or extra
        ]
    }
    gts = {
        0: [{'box': _make_box(0), 'class': 'car'}],
        1: [{'box': _make_box(10), 'class': 'car'}],
    }

    res = calculate_map_multi_frame(preds, gts, class_names=['car'])
    assert 'num_gt' in res and res['num_gt'] == 2
    assert 'num_preds' in res and res['num_preds'] >= 2
    assert 0.0 <= res['mAP'] <= 1.0