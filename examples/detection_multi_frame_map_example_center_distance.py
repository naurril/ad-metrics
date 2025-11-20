"""
Example: Multi-frame mAP using nuScenes-style center-distance matching.

This demonstrates using `matching='center_distance'` and multiple
distance thresholds (e.g. [0.5, 1.0, 2.0, 4.0] meters) which are treated
as matching radii instead of IoU thresholds.
"""
from admetrics.detection.ap import calculate_map_multi_frame


def _make_box(x, y=0.0):
    # box format: [x, y, z, w, h, l, yaw]
    return [x, y, 0.0, 1.0, 1.0, 1.0, 0.0]


def create_sequence():
    # Predictions span 4 frames; one false positive far away
    preds = {
        0: [{'box': _make_box(0.0), 'score': 0.9, 'class': 'car'}],
        1: [{'box': _make_box(1.0), 'score': 0.8, 'class': 'car'}],
        2: [{'box': _make_box(2.0), 'score': 0.6, 'class': 'car'}],
        3: [{'box': _make_box(30.0), 'score': 0.95, 'class': 'car'}],
    }

    gts = {
        0: [{'box': _make_box(0.0), 'class': 'car'}],
        1: [{'box': _make_box(1.0), 'class': 'car'}],
        2: [{'box': _make_box(2.0), 'class': 'car'}],
        3: [{'box': _make_box(3.0), 'class': 'car'}],
    }
    return preds, gts


def main():
    preds, gts = create_sequence()
    classes = ['car']

    thresholds = [0.5, 1.0, 2.0, 4.0]

    print('Center-distance multi-frame evaluation (thresholds in meters):', thresholds)
    res = calculate_map_multi_frame(
        preds,
        gts,
        class_names=classes,
        mode='framewise_then_global',
        iou_thresholds=thresholds,
        matching='center_distance',
    )

    print('\nAP per threshold:')
    for t, ap in zip(thresholds, res.get('AP_per_threshold', [])):
        print(f'  {t} m: {ap:.4f}')

    print('\nAP per class:')
    for cls, ap in res.get('AP_per_class', {}).items():
        print(f'  {cls}: {ap:.4f}')

    print('\nMean AP (averaged across thresholds):', res['mAP'])


if __name__ == '__main__':
    main()
