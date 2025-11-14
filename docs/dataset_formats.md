# Dataset Formats for Autonomous Driving

Comprehensive guide to common dataset formats used in autonomous driving benchmarks for detection, tracking, prediction, and localization.

## Table of Contents

1. [KITTI Dataset](#kitti-dataset)
2. [nuScenes Dataset](#nuscenes-dataset)
3. [Waymo Open Dataset](#waymo-open-dataset)
4. [Coordinate Systems](#coordinate-systems)
5. [Format Conversion](#format-conversion)

---

## KITTI Dataset

### Overview

KITTI is one of the most popular benchmarks for 3D object detection in autonomous driving.

**Official Site**: http://www.cvlibs.net/datasets/kitti/

### Directory Structure

```
kitti/
├── training/
│   ├── image_2/           # Left color camera images
│   ├── image_3/           # Right color camera images
│   ├── calib/             # Calibration files
│   ├── label_2/           # 2D/3D annotations
│   └── velodyne/          # Point cloud data (.bin)
└── testing/
    ├── image_2/
    ├── calib/
    └── velodyne/
```

### Label Format

Each line in `label_2/*.txt` represents one object:

```
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom height width length x y z rotation_y score
```

**Fields**:

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| `type` | str | Object class | Car, Pedestrian, Cyclist, etc. |
| `truncated` | float | Truncation level | [0, 1] (0=fully visible) |
| `occluded` | int | Occlusion state | 0=visible, 1=partly, 2=largely, 3=unknown |
| `alpha` | float | Observation angle | [-π, π] |
| `bbox_left` | float | 2D bbox left (pixels) | - |
| `bbox_top` | float | 2D bbox top (pixels) | - |
| `bbox_right` | float | 2D bbox right (pixels) | - |
| `bbox_bottom` | float | 2D bbox bottom (pixels) | - |
| `height` | float | 3D box height (m) | - |
| `width` | float | 3D box width (m) | - |
| `length` | float | 3D box length (m) | - |
| `x` | float | 3D center x (camera coords, m) | - |
| `y` | float | 3D center y (camera coords, m) | - |
| `z` | float | 3D center z (camera coords, m) | - |
| `rotation_y` | float | Rotation around Y-axis | [-π, π] |
| `score` | float | Confidence (predictions only) | [0, 1] |

**Example**:
```
Car 0.00 0 -1.58 599.41 156.40 629.75 189.25 1.48 1.60 3.69 2.84 1.47 8.41 -1.56
```

### Coordinate System

**Camera Coordinate System** (right-handed):
- **X**: Right
- **Y**: Down
- **Z**: Forward (into the scene)
- **Origin**: Camera center

**3D Box Representation**:
- `(x, y, z)`: Center of the **bottom face** of the 3D box
- `rotation_y`: Rotation around Y-axis (gravity direction)
  - 0 = aligned with camera Z-axis
  - Positive = counterclockwise when viewed from above

### Difficulty Levels

KITTI defines three difficulty levels:

| Difficulty | Min Height (px) | Max Occlusion | Max Truncation |
|------------|----------------|---------------|----------------|
| Easy       | 40             | Fully visible | 15%            |
| Moderate   | 25             | Partly (≤50%) | 30%            |
| Hard       | 25             | Difficult (>50%) | 50%         |

### Calibration Files

Each `calib/*.txt` file contains camera projection matrices:

```
P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 ...
P1: ...
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 ...
P3: ...
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 ...
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 ...
Tr_imu_to_velo: ...
```

**Key Matrices**:
- `P2`: 3x4 projection matrix for left color camera
- `R0_rect`: 3x3 rectification matrix
- `Tr_velo_to_cam`: 3x4 transformation from velodyne to camera

### Object Classes

**Main Classes** (for detection):
- `Car`
- `Pedestrian`
- `Cyclist`

**Other Classes**:
- `Van`, `Truck`, `Tram`
- `Person_sitting`
- `Misc`, `DontCare`

### IoU Thresholds

KITTI uses class-specific IoU thresholds:

| Class | 3D IoU | BEV IoU |
|-------|--------|---------|
| Car   | 0.7    | 0.7     |
| Pedestrian | 0.5 | 0.5  |
| Cyclist | 0.5   | 0.5     |

---

## nuScenes Dataset

### Overview

nuScenes is a large-scale autonomous driving dataset with full 360° coverage.

**Official Site**: https://www.nuscenes.org/

### Directory Structure

```
nuscenes/
├── v1.0-trainval/
│   ├── samples/          # Sensor data (keyframes)
│   │   ├── CAM_FRONT/
│   │   ├── CAM_FRONT_LEFT/
│   │   ├── CAM_FRONT_RIGHT/
│   │   ├── CAM_BACK/
│   │   ├── CAM_BACK_LEFT/
│   │   ├── CAM_BACK_RIGHT/
│   │   ├── LIDAR_TOP/
│   │   ├── RADAR_FRONT/
│   │   └── ...
│   ├── sweeps/           # Intermediate lidar sweeps
│   ├── maps/             # Map data
│   └── v1.0-trainval/    # JSON metadata
└── v1.0-test/
```

### Annotation Format

nuScenes uses JSON files for annotations. Key tables:

**sample_annotation** (main 3D boxes):
```json
{
  "token": "unique_id",
  "sample_token": "parent_sample_id",
  "instance_token": "tracked_instance_id",
  "attribute_tokens": ["attribute_id1", "attribute_id2"],
  "visibility_token": "visibility_level",
  "translation": [x, y, z],
  "size": [width, length, height],
  "rotation": [w, x, y, z],
  "num_lidar_pts": 123,
  "num_radar_pts": 45,
  "next": "next_frame_annotation_token",
  "prev": "prev_frame_annotation_token"
}
```

### Coordinate System

**Global Coordinate System** (right-handed):
- **X**: Right (when facing forward)
- **Y**: Forward
- **Z**: Up
- **Origin**: Arbitrary global reference

**3D Box Representation**:
- `translation`: [x, y, z] - center of the box
- `size`: [width, length, height]
  - width: dimension along X-axis
  - length: dimension along Y-axis  
  - height: dimension along Z-axis
- `rotation`: Quaternion [w, x, y, z]
- `velocity`: [vx, vy] in m/s (if available)

### Object Classes

nuScenes has 10 detection classes:

1. `car`
2. `truck`
3. `bus`
4. `trailer`
5. `construction_vehicle`
6. `pedestrian`
7. `motorcycle`
8. `bicycle`
9. `traffic_cone`
10. `barrier`

### Attributes

Objects have attributes describing state:

**Vehicle Attributes**:
- `vehicle.moving`
- `vehicle.stopped`
- `vehicle.parked`

**Pedestrian Attributes**:
- `pedestrian.standing`
- `pedestrian.sitting_lying_down`
- `pedestrian.moving`

**Cycle Attributes**:
- `cycle.with_rider`
- `cycle.without_rider`

### Visibility Levels

- `v0-40`: 0-40% visible
- `v40-60`: 40-60% visible
- `v60-80`: 60-80% visible
- `v80-100`: 80-100% visible

### NuScenes Detection Score (NDS)

nuScenes uses NDS as the primary metric, combining:
- **mAP**: Detection accuracy
- **ATE**: Translation error (< 1m)
- **ASE**: Scale error (< 1 - IoU)
- **AOE**: Orientation error (< 0.8 rad ≈ 46°)
- **AVE**: Velocity error (< 0.5 m/s)
- **AAE**: Attribute error (< 0.2)

### Distance Ranges

Evaluation at different ranges:
- **TP Threshold**: Match within 2m center distance
- **Detection Range**: Objects within 50m from ego vehicle

---

## Waymo Open Dataset

### Overview

Waymo Open Dataset is a large-scale dataset with high-resolution sensors.

**Official Site**: https://waymo.com/open/

### Data Format

Waymo uses **TFRecord** format with Protocol Buffers.

**Main Fields**:
```protobuf
message Frame {
  int64 timestamp_micros = 1;
  string context_name = 2;
  repeated CameraImage images = 3;
  repeated LaserScan laser_scans = 4;
  repeated Label laser_labels = 5;
  CameraLabels camera_labels = 6;
}

message Label {
  Box box = 1;
  Metadata metadata = 2;
  Type type = 3;
  string id = 4;
  float detection_difficulty_level = 5;
  int32 num_lidar_points_in_box = 6;
}

message Box {
  double center_x = 1;
  double center_y = 2;
  double center_z = 3;
  double length = 4;
  double width = 5;
  double height = 6;
  double heading = 7;
}
```

### Coordinate System

**Vehicle Coordinate System** (right-handed):
- **X**: Forward
- **Y**: Left
- **Z**: Up
- **Origin**: Vehicle center

**3D Box Representation**:
- `(center_x, center_y, center_z)`: Box center
- `(length, width, height)`: Box dimensions
  - length: along X-axis (forward/backward)
  - width: along Y-axis (left/right)
  - height: along Z-axis (up/down)
- `heading`: Rotation around Z-axis (yaw)
  - 0 = aligned with X-axis
  - Positive = counterclockwise from above

### Object Classes

1. `TYPE_VEHICLE`
2. `TYPE_PEDESTRIAN`
3. `TYPE_SIGN`
4. `TYPE_CYCLIST`

### Difficulty Levels

**Level 1 (Easy)**:
- At least 5 lidar points
- Not occluded

**Level 2 (Hard)**:
- At least 1 lidar point
- Any occlusion level

### Evaluation Metrics

Waymo uses:
- **mAP**: At IoU thresholds [0.5, 0.7]
- **mAPH**: Mean Average Precision weighted by Heading accuracy

---

## Coordinate Systems

### Summary of Differences

| Dataset | X-axis | Y-axis | Z-axis | Rotation | Origin |
|---------|--------|--------|--------|----------|--------|
| KITTI   | Right  | Down   | Forward| Around Y | Camera |
| nuScenes| Right  | Forward| Up     | Quaternion| Global |
| Waymo   | Forward| Left   | Up     | Around Z | Vehicle|

### Common Conventions

**Camera Coordinates** (KITTI):
- Natural for camera-based detection
- Y-down matches image coordinates

**Global Coordinates** (nuScenes):
- Enables multi-sensor fusion
- Consistent across all sensors

**Vehicle Coordinates** (Waymo):
- Intuitive for planning
- X-forward aligns with motion

---

## Format Conversion

### KITTI to Standard Format

```python
def kitti_to_standard(kitti_box, calib):
    """
    Convert KITTI format to standard [x, y, z, w, h, l, yaw].
    
    KITTI: Camera coords, Y-down, rotation_y
    Standard: LiDAR coords, Z-up
    """
    # Extract KITTI fields
    x_cam, y_cam, z_cam = kitti_box['x'], kitti_box['y'], kitti_box['z']
    h, w, l = kitti_box['height'], kitti_box['width'], kitti_box['length']
    ry = kitti_box['rotation_y']
    
    # Transform camera -> velodyne using calibration
    # Apply Tr_velo_to_cam inverse and R0_rect inverse
    pts_cam = np.array([[x_cam], [y_cam], [z_cam], [1.0]])
    pts_velo = np.linalg.inv(calib['Tr_velo_to_cam']) @ pts_cam
    
    x, y, z = pts_velo[0, 0], pts_velo[1, 0], pts_velo[2, 0]
    
    # Adjust rotation (camera Y-axis to LiDAR Z-axis)
    yaw = -ry - np.pi / 2
    
    return [x, y, z, w, l, h, yaw]
```

### nuScenes to Standard Format

```python
from pyquaternion import Quaternion

def nuscenes_to_standard(nuscenes_box):
    """
    Convert nuScenes format to standard [x, y, z, w, h, l, yaw].
    
    nuScenes: Global coords, quaternion rotation
    Standard: [x, y, z, w, h, l, yaw]
    """
    x, y, z = nuscenes_box['translation']
    w, l, h = nuscenes_box['size']  # Note: nuScenes order is [w, l, h]
    
    # Convert quaternion to yaw
    q = Quaternion(nuscenes_box['rotation'])
    yaw = q.yaw_pitch_roll[0]
    
    return [x, y, z, w, h, l, yaw]
```

### Waymo to Standard Format

```python
def waymo_to_standard(waymo_box):
    """
    Convert Waymo format to standard [x, y, z, w, h, l, yaw].
    
    Waymo: Vehicle coords, heading around Z
    Standard: [x, y, z, w, h, l, yaw]
    """
    x = waymo_box.center_x
    y = waymo_box.center_y
    z = waymo_box.center_z
    l = waymo_box.length  # Along X (forward)
    w = waymo_box.width   # Along Y (left)
    h = waymo_box.height  # Along Z (up)
    yaw = waymo_box.heading
    
    return [x, y, z, w, h, l, yaw]
```

### Standard Format Definition

Our library uses a **unified standard format**:

```python
box = [x, y, z, w, h, l, yaw]
```

Where:
- `[x, y, z]`: Box center in 3D space
- `w`: Width (dimension along local X-axis)
- `h`: Height (dimension along local Z-axis, vertical)
- `l`: Length (dimension along local Y-axis)
- `yaw`: Rotation around Z-axis (up), in radians

**Coordinate System**:
- Right-handed with Z-up
- Yaw = 0 means box is aligned with global X-axis
- Positive yaw = counterclockwise rotation when viewed from above

---

## Best Practices

### When Loading Data

1. **Verify Coordinate Systems**: Always check dataset documentation
2. **Apply Calibrations**: Use provided transformation matrices
3. **Handle Edge Cases**: Missing fields, invalid rotations
4. **Validate Ranges**: Ensure values are within expected bounds

### When Evaluating

1. **Use Dataset-Specific IoU Thresholds**: Different classes, different thresholds
2. **Filter by Difficulty**: Separate easy/moderate/hard cases
3. **Check Distance Ranges**: Some benchmarks limit evaluation range
4. **Match Box Formats**: Ensure predictions use same format as GT

### Common Pitfalls

❌ **Mixing Coordinate Systems**: Always transform to common system  
❌ **Wrong Rotation Convention**: Check rotation axis and direction  
❌ **Dimension Order**: [w, h, l] vs [l, w, h] varies by dataset  
❌ **Center vs Corner**: Some formats use bottom-center, others use geometric center  

---

## References

- **KITTI**: Geiger et al., "Are we ready for Autonomous Driving?" (CVPR 2012)
- **nuScenes**: Caesar et al., "nuScenes: A multimodal dataset for autonomous driving" (CVPR 2020)
- **Waymo**: Sun et al., "Scalability in Perception for Autonomous Driving" (CVPR 2020)
