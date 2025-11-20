"""
Dataset conversion utilities for detection evaluation.

Provides helpers to convert common dataset record formats (KITTI, nuScenes)
or generic record lists into the per-frame dicts expected by
`calculate_map_multi_frame`.

These are lightweight converters - adapt them to your dataset's exact schema
as needed.
"""
from typing import List, Dict, Any


def group_by_frame(records: List[Dict[str, Any]], frame_key: str = 'frame_id') -> Dict[int, List[Dict]]:
    """
    Group a flat list of detection/annotation records by frame id.

    Args:
        records: list of dicts where each dict contains a frame identifier key.
        frame_key: key name that contains the frame id (int-like).

    Returns:
        Dict mapping frame_id (int) -> list of records for that frame.
    """
    out: Dict[int, List[Dict]] = {}
    for r in records:
        if frame_key not in r:
            raise KeyError(f"Record missing frame key '{frame_key}': {r}")
        fid = int(r[frame_key])
        out.setdefault(fid, []).append(r)
    return out


def parse_kitti_line(line: str) -> Dict:
    """
    Parse a single KITTI detection/label line into a canonical detection dict.

    KITTI format (per line):
    type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry [score]

    Returns:
        dict with keys: 'class', 'truncated', 'occluded', 'alpha', 'bbox_2d',
        'box' (x,y,z,w,h,l,ry), and 'score' if present.
    """
    parts = line.strip().split()
    if len(parts) < 15:
        raise ValueError('KITTI line must have at least 15 fields')

    obj_type = parts[0]
    truncated = float(parts[1])
    occluded = int(float(parts[2]))
    alpha = float(parts[3])
    x1, y1, x2, y2 = map(float, parts[4:8])
    h, w, l = map(float, parts[8:11])
    x, y, z = map(float, parts[11:14])
    ry = float(parts[14])

    score = None
    if len(parts) >= 16:
        try:
            score = float(parts[15])
        except Exception:
            score = None

    det = {
        'class': obj_type,
        'truncated': truncated,
        'occluded': occluded,
        'alpha': alpha,
        'bbox_2d': [x1, y1, x2, y2],
        'box': [x, y, z, w, h, l, ry]
    }
    if score is not None:
        det['score'] = score
    return det


def kitti_lines_to_per_frame(lines_per_frame: Dict[int, List[str]]) -> Dict[int, List[Dict]]:
    """
    Convert a mapping of frame_id -> list of KITTI-format lines to the per-frame
    detection dict format expected by the evaluation tools.
    """
    out: Dict[int, List[Dict]] = {}
    for fid, lines in lines_per_frame.items():
        out[fid] = [parse_kitti_line(l) for l in lines]
    return out


def nuscenes_ann_to_detection(ann: Dict[str, Any]) -> Dict:
    """
    Convert a nuScenes annotation/detection dict to canonical detection dict.

    Expected keys (common subset):
      - 'translation': [x,y,z]
      - 'size': [w,l,h] or [width,length,height]
      - 'rotation': quaternion or yaw (if yaw provided)
      - 'detection_score' or 'score'
      - 'category_name' or 'label'

    This is a best-effort conversion; adapt to your dataset schema.
    """
    det: Dict = {}
    trans = ann.get('translation') or ann.get('translation_xy') or [0.0, 0.0, 0.0]
    size = ann.get('size') or ann.get('box_size') or ann.get('dimensions') or [1.0, 1.0, 1.0]
    # Normalize size order to (w, h, l) used in this repo: [w,h,l]
    # If nuScenes provides [w,l,h] or [length,width,height], user should adapt accordingly.
    # We'll attempt to place values as [w, h, l] by common conventions.
    if len(size) >= 3:
        w, l, h = float(size[0]), float(size[1]), float(size[2])
    else:
        w, l, h = 1.0, 1.0, 1.0

    # rotation: prefer yaw if present, else 0.0
    yaw = ann.get('yaw', 0.0)
    if yaw == 0.0 and 'rotation' in ann:
        # rotation may be quaternion; leave as-is (user must preprocess)
        yaw = ann.get('rotation')

    det['class'] = ann.get('category_name') or ann.get('label') or 'unknown'
    det['box'] = [float(trans[0]), float(trans[1]), float(trans[2]), float(w), float(h), float(l), float(yaw)]
    score = ann.get('detection_score') or ann.get('score')
    if score is not None:
        det['score'] = float(score)
    return det


def parse_kitti_lines_file(lines: List[str]) -> Dict[int, List[Dict]]:
    """
    Parse a flat list of KITTI label/prediction lines where each line is prefixed
    with a frame id (int) followed by the KITTI fields. Example:

        "0 Car 0.00 0 0.00 x1 y1 x2 y2 h w l x y z ry score"

    Returns a mapping frame_id -> list of detection dicts.
    """
    out: Dict[int, List[Dict]] = {}
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) == 0:
            continue
        # first token is frame id
        try:
            fid = int(parts[0])
            rest = ' '.join(parts[1:])
        except ValueError:
            # if no frame id present, skip
            continue
        out.setdefault(fid, []).append(parse_kitti_line(rest))
    return out


def quaternion_to_yaw(q: List[float]) -> float:
    """Convert quaternion [w,x,y,z] to yaw (rotation about z).

    Returns yaw in radians.
    """
    if q is None or len(q) != 4:
        return 0.0
    w, x, y, z = map(float, q)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def nuscenes_results_to_per_frame(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict]]:
    """
    Convert nuScenes-style submission `results` mapping (sample_token -> list of
    sample_result dicts) into per-frame detection dicts usable by the evaluation
    helpers in this repo.

    Each `sample_result` is expected to contain:
      - 'translation': [x,y,z]
      - 'size': [w,l,h]
      - 'rotation': [w,x,y,z] (quaternion) or 'yaw'
      - 'detection_name': class name
      - 'detection_score': score

    Returns a dict mapping sample_token -> list of detection dicts with keys
    'class', 'box' ([x,y,z,w,h,l,yaw]), and 'score'.
    """
    out: Dict[str, List[Dict]] = {}
    for token, ann_list in results.items():
        dets: List[Dict] = []
        for ann in ann_list:
            trans = ann.get('translation') or ann.get('translation_xyz') or [0.0, 0.0, 0.0]
            size = ann.get('size') or ann.get('box_size') or ann.get('dimensions') or [1.0, 1.0, 1.0]
            # nuScenes size ordering is [w,l,h]
            if len(size) >= 3:
                w = float(size[0]); l = float(size[1]); h = float(size[2])
            else:
                w, l, h = 1.0, 1.0, 1.0

            # rotation
            rot = ann.get('rotation')
            yaw = ann.get('yaw', None)
            if yaw is None and rot is not None:
                try:
                    yaw = quaternion_to_yaw(rot)
                except Exception:
                    yaw = 0.0

            det = {
                'class': ann.get('detection_name') or ann.get('detection_name', ann.get('category_name', 'unknown')),
                'box': [float(trans[0]), float(trans[1]), float(trans[2]), float(w), float(h), float(l), float(yaw if yaw is not None else 0.0)],
            }
            sc = ann.get('detection_score') or ann.get('score')
            if sc is not None:
                det['score'] = float(sc)
            dets.append(det)
        out[token] = dets
    return out
