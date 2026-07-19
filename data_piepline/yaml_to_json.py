import yaml
import os
import bisect
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import json
from tqdm import tqdm
import gc
from parameters import *


class NpEncoder(json.JSONEncoder):
    """Minimal encoder: only handles numpy types not caught by the C encoder.

    np.float64 subclasses float so the C encoder handles it natively.
    We avoid overriding iterencode() — that would force the slow pure-Python
    encoder path. Instead we use nf() at the point of assignment to convert
    numpy scalars to Python float/None before they reach the encoder.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if not np.isfinite(obj) else float(obj)
        return super().default(obj)


def nf(v):
    """Convert a numpy float scalar to Python float, or None for NaN/Inf."""
    if isinstance(v, np.floating):
        return None if not np.isfinite(v) else float(v)
    return v


def load_yaml_data(file_path):
    """Load YAML data with error handling. Binary mode + CSafeLoader is ~3x faster."""
    try:
        with open(file_path, 'rb') as file:
            return yaml.load(file, Loader=yaml.CSafeLoader)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def get_velocity_from_twist(twist):
    """Extract linear velocity vector from twist data"""
    try:
        vx = twist['linear']['x']
        vy = twist['linear']['y']
        vz = twist['linear']['z']
        return np.array([vx, vy, vz])
    except (KeyError, TypeError):
        return np.array([np.nan, np.nan, np.nan])


def calculate_velocity_vector(pos_current, pos_previous, time_current, time_previous):
    """Calculate velocity from position changes"""
    if time_previous == time_current:
        return np.array([np.nan, np.nan, np.nan])

    dt = time_current - time_previous
    dx = pos_current['x'] - pos_previous['x']
    dy = pos_current['y'] - pos_previous['y']
    dz = pos_current['z'] - pos_previous['z']

    return np.array([dx/dt, dy/dt, dz/dt])


def create_2d_oriented_box(position, rotation, extents):
    """Create 2D bounding box corners from vehicle pose and half-extents.

    New YAML format uses Z-up (ROS/Autoware convention):
      - Ground plane is XY  (X=forward, Y=left)
      - Yaw = rotation['z'] (rotation around Z-up axis)
      - extents['x'] = half forward dimension
      - extents['y'] = half lateral dimension
    """
    half_length = extents['x']
    half_width = extents['y']

    # Negate to convert from the simulator's clockwise convention to math CCW
    yaw_rad = np.radians(-rotation['z'])
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    corners_local = np.array([
        [ half_length,  half_width],
        [ half_length, -half_width],
        [-half_length, -half_width],
        [-half_length,  half_width]
    ])

    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw]
    ])

    corners_rotated = np.dot(corners_local, rotation_matrix.T)
    # Ground plane in new format is XY
    corners_world = corners_rotated + np.array([position['x'], position['y']])

    return corners_world


def find_nearest_timestamp(ts, sorted_timestamps):
    """Return the closest timestamp from a sorted list using binary search."""
    idx = bisect.bisect_left(sorted_timestamps, ts)
    if idx == 0:
        return sorted_timestamps[0]
    if idx >= len(sorted_timestamps):
        return sorted_timestamps[-1]
    before = sorted_timestamps[idx - 1]
    after = sorted_timestamps[idx]
    return before if (ts - before) <= (after - ts) else after


def extract_data_from_yaml_optimized(yaml_data, output_path):
    """Extract bounding box and velocity data from new-format YAML."""
    try:
        gt_frames = yaml_data['groundtruth_kinematic']
        total_frames = len(gt_frames)
        frames_needed = min(MAX_FRAMES, total_frames)
        step = max(1, total_frames // frames_needed) if frames_needed < total_frames else 1
        frame_indices = list(range(0, total_frames, step))

        # Build timestamp → data lookup tables for alignment
        # For ctrl_cmds, duplicate timestamps exist (keep last value)
        perc_by_ts = {
            entry['timestamp']: entry.get('objects') or []
            for entry in yaml_data.get('perception_objects', [])
        }
        perc_timestamps = sorted(perc_by_ts.keys())

        ego_est_by_ts = {
            entry['timestamp']: entry
            for entry in yaml_data.get('ego_estimated_kinematic', [])
        }
        ego_est_timestamps = sorted(ego_est_by_ts.keys())

        ctrl_by_ts = {
            entry['timestamp']: entry
            for entry in yaml_data.get('control_cmds', [])
        }
        ctrl_timestamps = sorted(ctrl_by_ts.keys())

        # Build half-extents lookup by vehicle name.
        # New format stores full dimensions under 'size'; halve them for the box function.
        # Ground-plane half-dims: x (forward) and y (lateral).
        vehicle_extents = {}
        for vs in yaml_data.get('groundtruth_size', {}).get('vehicle_sizes', []):
            s = vs['size']
            vehicle_extents[vs['name']] = {
                'x': s['x'] / 2,
                'y': s['y'] / 2,
                'z': s['z'] / 2
            }
        ego_extent = vehicle_extents.get('ego')
        npc_extent = vehicle_extents.get('npc1')

        previous_state = None
        previous_timestamp = 0

        bbox_data = {
            "metadata": {
                "total_frames": total_frames,
                "processed_frames": len(frame_indices),
                "source_file": os.path.basename(output_path).replace('_data.json', '.yaml')
            },
            "frames": []
        }

        for frame_idx in frame_indices:
            state = gt_frames[frame_idx]
            current_timestamp = state["timestamp"]

            frame_data = {
                "timestamp": current_timestamp,
                "frame_index": frame_idx,
                "ego": {"box": None, "velocity": None, "position": None},
                "ego_estimated": None,
                "npc1": {"box": None, "velocity": None, "position": None},
                "perception_objects": [],
                "control_cmds": None
            }

            # ===== EGO VEHICLE (GROUNDTRUTH) =====
            ego_data = state['groundtruth_ego']
            ego_pos = ego_data['pose']['position']
            ego_rot = ego_data['pose']['rotation']

            frame_data["ego"]["position"] = {
                "x": float(ego_pos['x']),
                "y": float(ego_pos['y']),
                "z": float(ego_pos['z'])
            }

            ego_velocity = np.array([np.nan, np.nan, np.nan])
            velocity_source = "none"

            if 'twist' in ego_data:
                ego_velocity = get_velocity_from_twist(ego_data['twist'])
                velocity_source = "twist"
            elif previous_state is not None:
                prev_ego_pos = previous_state['groundtruth_ego']['pose']['position']
                ego_velocity = calculate_velocity_vector(
                    ego_pos, prev_ego_pos, current_timestamp, previous_timestamp)
                velocity_source = "calculated"

            magnitude = np.linalg.norm(ego_velocity) if not np.isnan(ego_velocity).any() else np.nan

            frame_data["ego"]["velocity"] = {
                "x": nf(ego_velocity[0]),
                "y": nf(ego_velocity[1]),
                "z": nf(ego_velocity[2]),
                "magnitude": nf(magnitude),
                "source": velocity_source
            }

            if ego_extent is not None:
                ego_box_2d = create_2d_oriented_box(ego_pos, ego_rot, ego_extent)
                frame_data["ego"]["box"] = ego_box_2d.tolist()

            # ===== EGO ESTIMATED KINEMATIC =====
            if ego_est_timestamps:
                nearest_ts = find_nearest_timestamp(current_timestamp, ego_est_timestamps)
                est = ego_est_by_ts[nearest_ts]
                est_vel = get_velocity_from_twist(est['twist'])
                est_mag = np.linalg.norm(est_vel) if not np.isnan(est_vel).any() else np.nan
                frame_data["ego_estimated"] = {
                    "position": {
                        "x": float(est['pose']['position']['x']),
                        "y": float(est['pose']['position']['y']),
                        "z": float(est['pose']['position']['z'])
                    },
                    "velocity": {
                        "x": nf(est_vel[0]),
                        "y": nf(est_vel[1]),
                        "z": nf(est_vel[2]),
                        "magnitude": nf(est_mag)
                    }
                }

            # ===== NPC VEHICLE (GROUNDTRUTH) =====
            if state.get('groundtruth_vehicles'):
                npc = state['groundtruth_vehicles'][0]
                npc_pos = npc['pose']['position']
                npc_rot = npc['pose']['rotation']

                frame_data["npc1"]["position"] = {
                    "x": float(npc_pos['x']),
                    "y": float(npc_pos['y']),
                    "z": float(npc_pos['z'])
                }

                npc_velocity = np.array([np.nan, np.nan, np.nan])
                velocity_source = "none"

                if 'twist' in npc:
                    npc_velocity = get_velocity_from_twist(npc['twist'])
                    velocity_source = "twist"
                elif previous_state is not None and previous_state.get('groundtruth_vehicles'):
                    prev_npc_pos = previous_state['groundtruth_vehicles'][0]['pose']['position']
                    npc_velocity = calculate_velocity_vector(
                        npc_pos, prev_npc_pos, current_timestamp, previous_timestamp)
                    velocity_source = "calculated"

                magnitude = np.linalg.norm(npc_velocity) if not np.isnan(npc_velocity).any() else np.nan

                frame_data["npc1"]["velocity"] = {
                    "x": nf(npc_velocity[0]),
                    "y": nf(npc_velocity[1]),
                    "z": nf(npc_velocity[2]),
                    "magnitude": nf(magnitude),
                    "source": velocity_source
                }

                if npc_extent is not None:
                    npc_box_2d = create_2d_oriented_box(npc_pos, npc_rot, npc_extent)
                    frame_data["npc1"]["box"] = npc_box_2d.tolist()

            # ===== PERCEPTION OBJECTS =====
            # Perception runs at a lower frequency than groundtruth; align by nearest timestamp.
            if perc_timestamps:
                nearest_ts = find_nearest_timestamp(current_timestamp, perc_timestamps)
                for perc_obj in perc_by_ts.get(nearest_ts) or []:
                    try:
                        perc_pos = perc_obj['pose']['position']
                        perc_rot = perc_obj['pose']['rotation']

                        perc_velocity = np.array([np.nan, np.nan, np.nan])
                        velocity_source = "none"

                        if 'twist' in perc_obj:
                            perc_velocity = get_velocity_from_twist(perc_obj['twist'])
                            velocity_source = "twist"
                        elif 'velocity' in perc_obj:
                            vel = perc_obj['velocity']
                            if isinstance(vel, dict) and all(k in vel for k in ['x', 'y', 'z']):
                                perc_velocity = np.array([vel['x'], vel['y'], vel['z']])
                                velocity_source = "direct"

                        perc_extent = None
                        if 'shape' in perc_obj and perc_obj['shape'] is not None and 'size' in perc_obj['shape']:
                            s = perc_obj['shape']['size']
                            perc_extent = {
                                'x': s['x'] / 2,
                                'y': s['y'] / 2,
                                'z': s['z'] / 2
                            }
                        elif 'dimensions' in perc_obj:
                            dims = perc_obj['dimensions']
                            if all(k in dims for k in ['width', 'length', 'height']):
                                perc_extent = {
                                    'x': dims['length'] / 2,
                                    'y': dims['width'] / 2,
                                    'z': dims['height'] / 2
                                }

                        if perc_extent is None:
                            continue

                        perc_box_2d = create_2d_oriented_box(perc_pos, perc_rot, perc_extent)
                        magnitude = np.linalg.norm(perc_velocity) if not np.isnan(perc_velocity).any() else np.nan

                        frame_data["perception_objects"].append({
                            "box": perc_box_2d.tolist(),
                            "velocity": {
                                "x": nf(perc_velocity[0]),
                                "y": nf(perc_velocity[1]),
                                "z": nf(perc_velocity[2]),
                                "magnitude": nf(magnitude),
                                "source": velocity_source
                            },
                            "position": {
                                "x": float(perc_pos['x']),
                                "y": float(perc_pos['y']),
                                "z": float(perc_pos['z'])
                            }
                        })
                    except Exception:
                        continue

            # ===== CONTROL COMMANDS =====
            if ctrl_timestamps:
                nearest_ts = find_nearest_timestamp(current_timestamp, ctrl_timestamps)
                cmd = ctrl_by_ts[nearest_ts]
                frame_data["control_cmds"] = {
                    "lateral": {
                        "steering_tire_angle": cmd['lateral'].get('steering_tire_angle'),
                        "steering_tire_rotation_rate": cmd['lateral'].get('steering_tire_rotation_rate')
                    },
                    "longitudinal": {
                        "velocity": cmd['longitudinal'].get('velocity'),
                        "acceleration": cmd['longitudinal'].get('acceleration'),
                        "jerk": cmd['longitudinal'].get('jerk')
                    }
                }

            bbox_data["frames"].append(frame_data)
            previous_state = state
            previous_timestamp = current_timestamp

        with open(output_path, 'w') as f:
            json.dump(bbox_data, f, indent=2, cls=NpEncoder)

        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing data for {output_path}: {str(e)}")
        return None


def process_yaml_file_optimized(yaml_path):
    """Process a single YAML file to JSON"""
    try:
        basename = os.path.basename(yaml_path)
        output_path = os.path.join(JSON_OUTPUT_DIR, f"{os.path.splitext(basename)[0]}_data.json")

        if os.path.exists(output_path):
            return {
                'yaml_path': yaml_path,
                'json_path': output_path,
                'success': True,
                'skipped': True
            }

        yaml_data = load_yaml_data(yaml_path)
        if yaml_data is None:
            return None

        json_path = extract_data_from_yaml_optimized(yaml_data, output_path)

        del yaml_data
        gc.collect()

        return {
            'yaml_path': yaml_path,
            'json_path': json_path,
            'success': json_path is not None,
            'skipped': False
        }

    except Exception as e:
        print(f"Error processing {yaml_path}: {str(e)}")
        return None


def main():
    """Main function to process all YAML files in parallel."""
    if not os.path.exists(INPUT_YAML_DIR):
        print(f"Input directory not found: {INPUT_YAML_DIR}")
        return

    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    yaml_files = [f for f in os.listdir(INPUT_YAML_DIR) if f.endswith('.yaml')]
    yaml_paths = [os.path.join(INPUT_YAML_DIR, f) for f in yaml_files]

    print(f"Found {len(yaml_paths)} YAML files total")

    existing_json = set()
    if os.path.exists(JSON_OUTPUT_DIR):
        existing_json = set(os.listdir(JSON_OUTPUT_DIR))

    files_to_process = []
    for yaml_path in yaml_paths:
        basename = os.path.basename(yaml_path)
        expected_json = f"{os.path.splitext(basename)[0]}_data.json"
        if expected_json not in existing_json:
            files_to_process.append(yaml_path)

    print(f"Found {len(existing_json)} existing JSON files")
    print(f"Need to process {len(files_to_process)} YAML files")

    if not files_to_process:
        print("All files already processed!")
        return

    n_workers = min(multiprocessing.cpu_count(), len(files_to_process))
    print(f"Using {n_workers} parallel workers")

    total_successful = 0
    total_skipped = 0
    total_failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(
            executor.map(process_yaml_file_optimized, files_to_process),
            total=len(files_to_process),
            desc="Processing YAML files"
        ):
            if result is None:
                total_failed += 1
            elif result.get('skipped'):
                total_skipped += 1
            elif result['success']:
                total_successful += 1
            else:
                total_failed += 1

    print(f"\n=== FINAL RESULTS ===")
    print(f"Total files processed: {len(files_to_process)}")
    print(f"Successful conversions: {total_successful}")
    print(f"Skipped (already existed): {total_skipped}")
    print(f"Failed conversions: {total_failed}")
    print(f"JSON files saved to: {JSON_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
