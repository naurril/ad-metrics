"""
Example: Multi-frame mAP calculation using `calculate_map_multi_frame`.
Demonstrates default mode `framewise_then_global` and alternative modes.
"""
import numpy as np
from admetrics.detection.ap import calculate_map_multi_frame


def _make_box(x):
    return [x, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def create_sample_sequence():
    # Predictions and GT across 4 frames
    preds = {
        0: [{'box': _make_box(0), 'score': 0.9, 'class': 'car'}],
        1: [{'box': _make_box(1), 'score': 0.8, 'class': 'car'}],
        2: [{'box': _make_box(2), 'score': 0.4, 'class': 'car'}],
        3: [{'box': _make_box(30), 'score': 0.95, 'class': 'car'}],  # false positive in frame 3
    }

    gts = {
        0: [{'box': _make_box(0), 'class': 'car'}],
        1: [{'box': _make_box(1), 'class': 'car'}],
        2: [{'box': _make_box(2), 'class': 'car'}],
        3: [{'box': _make_box(3), 'class': 'car'}],
    }

    return preds, gts


def main():
    preds, gts = create_sample_sequence()
    classes = ['car']

    print('Running multi-frame mAP calculation')
    res = calculate_map_multi_frame(preds, gts, class_names=classes)
    print('Result:', res)


if __name__ == '__main__':
    main()
