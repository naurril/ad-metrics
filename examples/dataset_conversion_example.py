"""
Example: Convert KITTI-like lines and nuScenes-like results into per-frame
detections using `admetrics.detection.dataset_utils`, then compute multi-frame
mAP using `calculate_map_multi_frame`.
"""
from admetrics.detection import dataset_utils as du
from admetrics.detection.ap import calculate_map_multi_frame


def main():
    # Simple KITTI-style lines with frame prefix
    k_lines = [
        "0 Car 0.0 0 0.0 0 0 10 10 1.5 1.8 4.0 0 0 0 0.0 0.9",
        "1 Car 0.0 0 0.0 0 0 10 10 1.5 1.8 4.0 1 0 0 0.0 0.85",
    ]

    k_per_frame = du.parse_kitti_lines_file(k_lines)

    # nuScenes-like results keyed by sample token
    nus_results = {
        's1': [
            {
                'translation': [0.0, 0.0, 0.0],
                'size': [1.8, 4.0, 1.5],
                'rotation': [1.0, 0.0, 0.0, 0.0],
                'detection_name': 'car',
                'detection_score': 0.95,
            }
        ],
        's2': [
            {
                'translation': [1.0, 0.0, 0.0],
                'size': [1.8, 4.0, 1.5],
                'rotation': [1.0, 0.0, 0.0, 0.0],
                'detection_name': 'car',
                'detection_score': 0.7,
            }
        ],
    }

    nus_per_frame = du.nuscenes_results_to_per_frame(nus_results)

    # For demonstration, convert tokens to integer-like frame ids for the
    # KITTI detections and use both sets separately.
    print('KITTI-converted per-frame detections:', k_per_frame)
    print('\nnuScenes-converted per-frame detections:', nus_per_frame)

    # Compute mAP on the KITTI-converted set (single class 'car')
    classes = ['car']
    # map expects integer frame ids for KITTI; nuscenes example uses string tokens
    res_k = calculate_map_multi_frame(k_per_frame, {}, class_names=classes)
    print('\nKITTI-converted mAP:', res_k['mAP'])

    # Compute center-distance mAP for nuscenes-like results
    res_n = calculate_map_multi_frame(
        nus_per_frame,
        nus_per_frame,
        class_names=classes,
        matching='center_distance',
        iou_thresholds=[0.5, 1.0, 2.0, 4.0],
    )
    print('\nnuScenes-converted mAP (center-distance):', res_n['mAP'])


if __name__ == '__main__':
    main()
