from admetrics.detection import dataset_utils as du


def test_parse_kitti_lines_file():
    lines = [
        # frame_id, type, truncated, occluded, alpha, x1, y1, x2, y2,
        # h, w, l, x, y, z, ry, score
        "0 Car 0.0 0 0.0 0 0 10 10 1.5 1.8 4.0 0 0 0 0.0 0.9",
        "1 Pedestrian 0.0 0 0.0 5 5 15 30 1.7 0.6 0.8 1 0 0 0 1.57 0.75",
    ]

    per_frame = du.parse_kitti_lines_file(lines)
    assert 0 in per_frame and 1 in per_frame
    car = per_frame[0][0]
    assert car['class'].lower().startswith('car')
    assert 'bbox_2d' in car and 'box' in car
    assert abs(car.get('score', 0) - 0.9) < 1e-6


def test_nuscenes_results_to_per_frame():
    results = {
        'sampletoken1': [
            {
                'translation': [1.0, 2.0, 0.0],
                'size': [1.6, 4.0, 1.5],
                'rotation': [0.7071, 0.0, 0.7071, 0.0],
                'detection_name': 'car',
                'detection_score': 0.88
            }
        ],
        'sampletoken2': [
            {
                'translation': [0.0, -1.0, 0.0],
                'size': [0.5, 0.5, 1.7],
                'rotation': [1.0, 0.0, 0.0, 0.0],
                'detection_name': 'pedestrian',
                'detection_score': 0.66
            }
        ]
    }

    out = du.nuscenes_results_to_per_frame(results)
    assert 'sampletoken1' in out and 'sampletoken2' in out
    d0 = out['sampletoken1'][0]
    assert d0['class'] == 'car'
    assert abs(d0['box'][0] - 1.0) < 1e-6
    assert abs(d0['score'] - 0.88) < 1e-6
