import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import cv2
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from tqdm import tqdm
import torch
from lxml import etree
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import box_utils, frame_utils, range_image_utils
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset import label_pb2

import multiprocessing as mp
import pprint

"""
    Modified from waymo_open_dataset/utils/frame_utils.py

    Original function returns dict of arrays, we want to save the LiDAR and images to disk rather than returning.
"""


def convert_frame_to_dict(
    tfrecord_path,
    frame,
    paths_dict: Dict[str, Path],
    objectid_to_label: Dict,
    objectid_to_color: Dict,
) -> Dict[str, Any]:
    """Convert the frame proto into a dict of numpy arrays and save data to disk.

    Arguments:
        frame: open dataset frame
        paths_dict: Dict[str, Path] containing paths for different data types:
            paths_dict = {
                "objectid_to_label": Path("data/objectid_to_label_level1.json"),
                "objectid_to_color": Path("data/objectid_to_color_level1.json"),
                "scene_infos": output_path / "scene_infos",
                "object_infos": output_path / "object_infos",
                "frame_infos": output_path / "frame_infos",
                "camera_images": output_path / "camera_images",
                "point_clouds": output_path / "point_clouds",
                "object_lists": output_path / "object_lists",  # Added new path for object lists
            }

    Returns:
        Dict with metadata about saved files
    """
    scene_id = frame.context.name
    timestamp = frame.timestamp_micros
    tfrecord_name = Path(tfrecord_path).name

    # Parse range images and convert to point clouds
    range_images, camera_projections, seg_labels, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose
    )

    # Initialize dictionaries for different types of data
    scene_info = {}
    frame_info = {}
    object_infos = []
    saved_files = {
        "scene_info": None,
        "frame_info": None,
        "object_infos": [],
        "camera_images": [],
        "point_clouds": [],
        "object_lists": [],  # Added for tracking object lists
    }

    # Add basic scene and frame info
    scene_info["scene_id"] = scene_id
    scene_info["tfrecord_name"] = tfrecord_name

    # Add basic frame information
    frame_info["scene_id"] = scene_id
    frame_info["timestamp"] = timestamp
    frame_info["time_of_day"] = frame.context.stats.time_of_day
    frame_info["weather"] = frame.context.stats.weather
    frame_info["location"] = frame.context.stats.location

    frame_info["pose"] = np.reshape(
        np.array(frame.pose.transform, np.float32), (4, 4)
    ).tolist()

    # Save camera calibrations in scene_info
    scene_info["camera_calibrations"] = []
    for c in frame.context.camera_calibrations:
        cam_name_str = open_dataset.CameraName.Name.Name(c.name)
        scene_info["camera_calibrations"].append(
            {
                "name": cam_name_str,
                "intrinsic": np.array(c.intrinsic, np.float32).tolist(),
                "extrinsic": np.reshape(
                    np.array(c.extrinsic.transform, np.float32), [4, 4]
                ).tolist(),
                "width": int(c.width),
                "height": int(c.height),
                "rolling_shutter_direction": int(c.rolling_shutter_direction),
            }
        )

    # Save laser calibrations in scene_info
    scene_info["laser_calibrations"] = []
    for c in frame.context.laser_calibrations:
        laser_name_str = open_dataset.LaserName.Name.Name(c.name)

        if len(c.beam_inclinations) == 0:
            beam_inclinations = (
                range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_images[c.name][0].shape.dims[0],
                )
                .numpy()
                .tolist()
            )
        else:
            beam_inclinations = np.array(c.beam_inclinations, np.float32).tolist()

        scene_info["laser_calibrations"].append(
            {
                "name": laser_name_str,
                "beam_inclinations": beam_inclinations,
                "extrinsic": np.reshape(
                    np.array(c.extrinsic.transform, np.float32), [4, 4]
                ).tolist(),
            }
        )

    # Save camera images and their metadata in frame_info
    frame_info["images"] = []
    visible_cameras = {}  # Track which cameras each object is visible in

    for im in frame.images:
        cam_name_str = open_dataset.CameraName.Name.Name(im.name)

        # Decode and save image
        img = tf.io.decode_jpeg(im.image).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_path = (
            paths_dict["camera_images"] / f"{scene_id}_{timestamp}_{cam_name_str}.jpg"
        )
        cv2.imwrite(str(img_path), img)
        saved_files["camera_images"].append(str(img_path))

        calibration = None
        for calib in frame.context.camera_calibrations:
            if calib.name == im.name:
                calibration = calib
                break

        assert calibration is not None

        # Add image metadata to frame_info
        frame_info["images"].append(
            {
                "name": cam_name_str,
                "path": str(img_path.relative_to(paths_dict["camera_images"].parent)),
                "velocity": {
                    "v_x": float(im.velocity.v_x),
                    "v_y": float(im.velocity.v_y),
                    "v_z": float(im.velocity.v_z),
                    "w_x": float(im.velocity.w_x),
                    "w_y": float(im.velocity.w_y),
                    "w_z": float(im.velocity.w_z),
                },
                "intrinsic": np.array(calibration.intrinsic, np.float32).tolist(),
                "extrinsic": np.reshape(
                    np.array(calibration.extrinsic.transform, np.float32), [4, 4]
                ).tolist(),
                "metadata": {
                    "width": int(calibration.width),
                    "height": int(calibration.height),
                    "rolling_shutter_direction": int(
                        calibration.rolling_shutter_direction
                    ),
                },
                "pose": np.reshape(
                    np.array(im.pose.transform, np.float32), (4, 4)
                ).tolist(),
                "pose_timestamp": float(im.pose_timestamp),
                "shutter": float(im.shutter),
                "camera_trigger_time": float(im.camera_trigger_time),
                "camera_readout_done_time": float(im.camera_readout_done_time),
            }
        )

    # Save point clouds for each lidar
    frame_info["point_clouds"] = []

    # Concatenate points from all lidars and save as one file
    points_all = np.concatenate(points, axis=0)
    pc_path = paths_dict["point_clouds"] / f"{scene_id}_{timestamp}_ALL.npy"
    np.save(str(pc_path), points_all)
    saved_files["point_clouds"].append(str(pc_path))

    frame_info["point_clouds"].append(
        {
            "name": "ALL",
            "path": str(pc_path.relative_to(paths_dict["point_clouds"].parent)),
            "num_points": points_all.shape[0],
        }
    )

    # Save individual lidar point clouds if needed
    for i, lidar_points in enumerate(points):
        if i < len(frame.context.laser_calibrations):
            laser_name_str = open_dataset.LaserName.Name.Name(
                frame.context.laser_calibrations[i].name
            )
            pc_path = (
                paths_dict["point_clouds"]
                / f"{scene_id}_{timestamp}_{laser_name_str}.npy"
            )
            np.save(str(pc_path), lidar_points)
            saved_files["point_clouds"].append(str(pc_path))

            frame_info["point_clouds"].append(
                {
                    "name": laser_name_str,
                    "path": str(pc_path.relative_to(paths_dict["point_clouds"].parent)),
                    "num_points": lidar_points.shape[0],
                }
            )

    # Save range image top pose in frame_info
    frame_info["range_image_top_pose"] = np.reshape(
        np.array(range_image_top_pose.data, np.float32), range_image_top_pose.shape.dims
    ).tolist()

    # Save camera projections in frame_info
    frame_info["camera_projections"] = {}
    for laser_name, cp_list in camera_projections.items():
        laser_name_str = open_dataset.LaserName.Name.Name(laser_name)
        frame_info["camera_projections"][laser_name_str] = []

        for i, cp in enumerate(cp_list):
            return_name = "FIRST_RETURN" if i == 0 else "SECOND_RETURN"
            frame_info["camera_projections"][laser_name_str].append(
                {
                    "return_type": return_name,
                    "shape": [int(dim) for dim in cp.shape.dims],
                }
            )

    # Track objects with different properties for creating filtered lists
    objects_with_cvat = []
    objects_without_cvat = []
    objects_with_color = []
    objects_without_color = []

    # Save objects and their metadata
    frame_info["objects"] = []
    frame_object_table = []  # Table for just this frame's objects

    for label in frame.laser_labels:
        object_id = label.id

        # Check if object is labeled in CVAT
        in_cvat = object_id in objectid_to_label
        cvat_label = objectid_to_label.get(object_id, None)
        cvat_color = objectid_to_color.get(object_id, None)

        # Track objects for filtered lists
        if in_cvat:
            objects_with_cvat.append(object_id)
        else:
            objects_without_cvat.append(object_id)

        if cvat_color is not None:
            objects_with_color.append(object_id)
        else:
            objects_without_color.append(object_id)

        # Determine visible cameras for this object
        visible_cameras_for_obj = []
        if (
            hasattr(label, "most_visible_camera_name")
            and label.most_visible_camera_name
        ):
            visible_cameras_for_obj.append(label.most_visible_camera_name)

        if (
            len(label.most_visible_camera_name) == 0
        ):  # not visible on camera -> don't bother extracting
            continue

        obj_info = {
            "id": object_id,
            "scene_id": scene_id,
            "timestamp": timestamp,
            "in_cvat": in_cvat,
            "cvat_label": cvat_label,
            "cvat_color": cvat_color,
            "visible_cameras": visible_cameras_for_obj,  # New field for tracking cameras
            "frames": [timestamp],  # New field to track frames where object appears
            "box": {
                "center_x": float(label.box.center_x),
                "center_y": float(label.box.center_y),
                "center_z": float(label.box.center_z),
                "length": float(label.box.length),
                "width": float(label.box.width),
                "height": float(label.box.height),
                "heading": float(label.box.heading),
            },
            "type": label_pb2.Label.Type.Name(label.type),
            "detection_difficulty_level": int(label.detection_difficulty_level),
            "tracking_difficulty_level": int(label.tracking_difficulty_level),
            "num_lidar_points_in_box": int(label.num_lidar_points_in_box),
        }

        # Add camera_synced_box if it exists
        if hasattr(label, "camera_synced_box"):
            obj_info["camera_synced_box"] = {
                "center_x": float(label.camera_synced_box.center_x),
                "center_y": float(label.camera_synced_box.center_y),
                "center_z": float(label.camera_synced_box.center_z),
                "length": float(label.camera_synced_box.length),
                "width": float(label.camera_synced_box.width),
                "height": float(label.camera_synced_box.height),
                "heading": float(label.camera_synced_box.heading),
            }

        # Add most_visible_camera_name if it exists
        if hasattr(label, "most_visible_camera_name"):
            obj_info["most_visible_camera_name"] = label.most_visible_camera_name

        # Add num_top_lidar_points_in_box if it exists
        if hasattr(label, "num_top_lidar_points_in_box"):
            obj_info["num_top_lidar_points_in_box"] = int(
                label.num_top_lidar_points_in_box
            )

        # Add metadata fields if they exist
        if hasattr(label, "metadata"):
            obj_info["metadata"] = {}
            if label.metadata.HasField("speed_x"):
                obj_info["metadata"]["speed_x"] = float(label.metadata.speed_x)
                obj_info["metadata"]["speed_y"] = float(label.metadata.speed_y)
                obj_info["metadata"]["accel_x"] = float(label.metadata.accel_x)
                obj_info["metadata"]["accel_y"] = float(label.metadata.accel_y)

            # Add speed_z and accel_z if they exist
            if hasattr(label.metadata, "speed_z"):
                obj_info["metadata"]["speed_z"] = float(label.metadata.speed_z)
            if hasattr(label.metadata, "accel_z"):
                obj_info["metadata"]["accel_z"] = float(label.metadata.accel_z)

        # Add to frame's object table
        frame_object_table.append(
            {
                "id": object_id,
                "scene_id": scene_id,
                "type": label_pb2.Label.Type.Name(label.type),
                "in_cvat": in_cvat,
                "cvat_label": cvat_label,
                "has_color": cvat_color is not None,
                "cvat_color": cvat_color,
            }
        )

        # Check if the object file already exists (from previous frames)
        object_path_pattern = f"object_{object_id}_*.json"
        existing_files = list(paths_dict["object_infos"].glob(object_path_pattern))

        # if existing_files:
        #     # Load the existing object info and update it
        #     existing_file = existing_files[0]
        #     with open(existing_file, "r") as f:
        #         existing_obj_info = json.load(f)

        #     # Update frames list
        #     if timestamp not in existing_obj_info["frames"]:
        #         existing_obj_info["frames"].append(timestamp)

        #     # Update visible cameras list
        #     for cam in visible_cameras_for_obj:
        #         if cam not in existing_obj_info["visible_cameras"]:
        #             existing_obj_info["visible_cameras"].append(cam)

        #     # Keep the existing file but update its contents
        #     obj_path = existing_file
        #     with open(obj_path, "w") as f:
        #         json.dump(existing_obj_info, f, indent=2)
        # else:

        # Create a new file with scene ID and timestamp in the name
        obj_path = (
            paths_dict["object_infos"]
            / f"object_{object_id}_{scene_id}_{timestamp}.json"
        )
        with open(obj_path, "w") as f:
            json.dump(obj_info, f, indent=2)

        saved_files["object_infos"].append(str(obj_path))

        # Add reference to object in frame_info
        frame_info["objects"].append(
            {
                "id": object_id,
                "path": str(obj_path.relative_to(paths_dict["object_infos"].parent)),
            }
        )

    # Create object_lists directory if it doesn't exist
    paths_dict["object_lists"].mkdir(exist_ok=True)

    # Save timestamp-specific filtered object lists to txt files
    cvat_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_{timestamp}_cvat_objects.txt"
    )
    with open(cvat_objects_path, "w") as f:
        for obj_id in objects_with_cvat:
            f.write(f"{obj_id}\n")
    saved_files["object_lists"].append(str(cvat_objects_path))

    non_cvat_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_{timestamp}_non_cvat_objects.txt"
    )
    with open(non_cvat_objects_path, "w") as f:
        for obj_id in objects_without_cvat:
            f.write(f"{obj_id}\n")
    saved_files["object_lists"].append(str(non_cvat_objects_path))

    color_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_{timestamp}_color_objects.txt"
    )
    with open(color_objects_path, "w") as f:
        for obj_id in objects_with_color:
            f.write(f"{obj_id}\n")
    saved_files["object_lists"].append(str(color_objects_path))

    no_color_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_{timestamp}_no_color_objects.txt"
    )
    with open(no_color_objects_path, "w") as f:
        for obj_id in objects_without_color:
            f.write(f"{obj_id}\n")
    saved_files["object_lists"].append(str(no_color_objects_path))

    # Create a timestamp-specific summary table with basic object info for quick filtering
    object_table_path = (
        paths_dict["object_lists"] / f"{scene_id}_{timestamp}_object_table.json"
    )
    with open(object_table_path, "w") as f:
        json.dump(frame_object_table, f, indent=2)
    saved_files["object_lists"].append(str(object_table_path))

    # Also create/update scene-level lists that aggregate all objects seen in this scene
    # These files track all objects seen in the scene across all frames processed so far

    # First, check if scene-level lists already exist and read them
    scene_cvat_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_all_cvat_objects.txt"
    )
    scene_cvat_objects = set()
    if scene_cvat_objects_path.exists():
        with open(scene_cvat_objects_path, "r") as f:
            scene_cvat_objects = set(line.strip() for line in f if line.strip())

    # Add new objects from this frame
    scene_cvat_objects.update(objects_with_cvat)

    # Write back the updated list
    with open(scene_cvat_objects_path, "w") as f:
        for obj_id in sorted(scene_cvat_objects):
            f.write(f"{obj_id}\n")

    if not scene_cvat_objects_path in saved_files["object_lists"]:
        saved_files["object_lists"].append(str(scene_cvat_objects_path))

    # Same for non-CVAT objects
    scene_non_cvat_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_all_non_cvat_objects.txt"
    )
    scene_non_cvat_objects = set()
    if scene_non_cvat_objects_path.exists():
        with open(scene_non_cvat_objects_path, "r") as f:
            scene_non_cvat_objects = set(line.strip() for line in f if line.strip())

    scene_non_cvat_objects.update(objects_without_cvat)

    with open(scene_non_cvat_objects_path, "w") as f:
        for obj_id in sorted(scene_non_cvat_objects):
            f.write(f"{obj_id}\n")

    if not scene_non_cvat_objects_path in saved_files["object_lists"]:
        saved_files["object_lists"].append(str(scene_non_cvat_objects_path))

    # Same for color objects
    scene_color_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_all_color_objects.txt"
    )
    scene_color_objects = set()
    if scene_color_objects_path.exists():
        with open(scene_color_objects_path, "r") as f:
            scene_color_objects = set(line.strip() for line in f if line.strip())

    scene_color_objects.update(objects_with_color)

    with open(scene_color_objects_path, "w") as f:
        for obj_id in sorted(scene_color_objects):
            f.write(f"{obj_id}\n")

    if not scene_color_objects_path in saved_files["object_lists"]:
        saved_files["object_lists"].append(str(scene_color_objects_path))

    # Same for no-color objects
    scene_no_color_objects_path = (
        paths_dict["object_lists"] / f"{scene_id}_all_no_color_objects.txt"
    )
    scene_no_color_objects = set()
    if scene_no_color_objects_path.exists():
        with open(scene_no_color_objects_path, "r") as f:
            scene_no_color_objects = set(line.strip() for line in f if line.strip())

    scene_no_color_objects.update(objects_without_color)

    with open(scene_no_color_objects_path, "w") as f:
        for obj_id in sorted(scene_no_color_objects):
            f.write(f"{obj_id}\n")

    if not scene_no_color_objects_path in saved_files["object_lists"]:
        saved_files["object_lists"].append(str(scene_no_color_objects_path))

    # Create/update the scene-level object table
    scene_object_table_path = (
        paths_dict["object_lists"] / f"{scene_id}_all_object_table.json"
    )
    scene_object_table = []

    if scene_object_table_path.exists():
        try:
            with open(scene_object_table_path, "r") as f:
                scene_object_table = json.load(f)
        except json.JSONDecodeError:
            # If the file is corrupted, start with an empty list
            scene_object_table = []

    # Create a dictionary of existing objects in the scene table for faster lookups
    existing_objects = {entry["id"]: i for i, entry in enumerate(scene_object_table)}

    # Update the scene object table with the frame's objects
    for obj_entry in frame_object_table:
        obj_id = obj_entry["id"]

        if obj_id in existing_objects:
            # Update existing entry
            index = existing_objects[obj_id]
            entry = scene_object_table[index]

            # Update timestamps - add new timestamp if not already present
            if "timestamps" not in entry:
                entry["timestamps"] = []

            if timestamp not in entry["timestamps"]:
                entry["timestamps"].append(timestamp)
        else:
            # Add new entry
            new_entry = obj_entry.copy()
            new_entry["timestamps"] = [timestamp]

            scene_object_table.append(new_entry)
            existing_objects[obj_id] = len(scene_object_table) - 1

    # Write the updated scene object table
    with open(scene_object_table_path, "w") as f:
        json.dump(scene_object_table, f, indent=2)

    if not scene_object_table_path in saved_files["object_lists"]:
        saved_files["object_lists"].append(str(scene_object_table_path))

    # Save scene info
    scene_info_path = paths_dict["scene_infos"] / f"{scene_id}.json"
    with open(scene_info_path, "w") as f:
        json.dump(scene_info, f, indent=2)
    saved_files["scene_info"] = str(scene_info_path)

    # Save frame info
    frame_info_path = paths_dict["frame_infos"] / f"{scene_id}_{timestamp}.json"
    with open(frame_info_path, "w") as f:
        json.dump(frame_info, f, indent=2)
    saved_files["frame_info"] = str(frame_info_path)

    return saved_files


def process_tfrecord(
    tfrecord_path: Path,
    paths_dict: Dict[str, Path],
    objectid_to_label: Dict,
    objectid_to_color: Dict,
) -> int:
    process_id = mp.current_process().name
    print(f"Process {process_id} starting on {tfrecord_path.name}")

    # Reset TensorFlow session and graph for each process
    tf.keras.backend.clear_session()

    try:
        # Use tf.io.gfile to open the tfrecord file
        print(f"Process {process_id}: Opening TFRecord file")
        dataset = tf.data.TFRecordDataset(
            str(tfrecord_path),
            compression_type="",
            buffer_size=1024 * 1024,
            num_parallel_reads=1,
        )

        # Force dataset initialization
        dataset = dataset.prefetch(1)
        iterator = iter(dataset)

        frame_idx = 0
        print(f"Process {process_id}: Starting to read frames")

        # Loop directly with an iterator to avoid TF dataset issues
        while True:
            try:
                if frame_idx % 10 == 0:
                    print(
                        f"Process {process_id}: Processing frame {frame_idx} from {tfrecord_path.name}"
                    )

                data = next(iterator)
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))

                convert_frame_to_dict(
                    tfrecord_path,
                    frame,
                    paths_dict,
                    objectid_to_label,
                    objectid_to_color,
                )

                if frame_idx == 0:
                    print("frame.context.stats.location", frame.context.stats.location)

                frame_idx += 1
            except StopIteration:
                break
            except Exception as e:
                print(
                    f"Process {process_id}: Error processing frame {frame_idx}: {str(e)}"
                )
                import traceback

                traceback.print_exc()
                continue

        print(
            f"Process {process_id}: Completed all {frame_idx} frames from {tfrecord_path.name}"
        )
        return 1
    except Exception as e:
        print(
            f"Process {process_id} failed on {tfrecord_path.name} with error: {str(e)}"
        )
        import traceback

        traceback.print_exc()
        return 0


def setup_processing_paths(
    output_path: Path, data_dir: Path = Path("data")
) -> Dict[str, Path]:
    """
    Set up and validate all required paths for processing.

    Args:
        output_path: Base path for output files
        data_dir: Directory containing reference data files

    Returns:
        Dictionary of path configurations
    """
    # Define required reference files
    required_files = [
        data_dir / "objectid_to_color_level1.json",
        data_dir / "objectid_to_label_level1.json",
    ]

    # Check if all required files exist
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required reference files: {', '.join(str(f) for f in missing_files)}\n"
            f"Please ensure all files exist in the {data_dir} directory."
        )

    # Create output directories
    output_dirs = [
        "scene_infos",
        "object_infos",
        "frame_infos",
        "camera_images",
        "point_clouds",
        "object_lists",
    ]

    # Create paths dictionary with both reference and output paths
    paths_dict = {
        # Reference files
        "objectid_to_label": data_dir / "objectid_to_label_level1.json",
        "objectid_to_color": data_dir / "objectid_to_color_level1.json",
        # Output directories (created if they don't exist)
        **{dir_name: output_path / dir_name for dir_name in output_dirs},
    }

    # Create output directories
    for dir_path in (paths_dict[k] for k in output_dirs):
        dir_path.mkdir(exist_ok=True, parents=True)

    return paths_dict


def export_data(validation_dir, output_dir, num_processes):
    """Process a TFRecord file and visualize frames with CVAT labels"""
    mp_context = mp.get_context("spawn")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    tfrecord_files = list(Path(validation_dir).rglob("*.tfrecord"))

    assert (
        len(tfrecord_files) == 202
    ), f"There should be 202 validation tfrecords, got {len(tfrecord_files)}."

    # Load our custom labels
    # TODO: add some env variable for loading these or something

    required_files = [
        "data/objectid_to_color_level1.json",
        "data/objectid_to_label_level1.json",
    ]
    required_files = [Path(x) for x in required_files]  # TODO: config path for ./data

    # assert all(x.exists() for x in required_files), f'All required files must exist! required_files existing: {','.join([f'{x} -> {x.exists()}' for x in required_files])}'

    # Setup paths
    paths_dict = setup_processing_paths(output_path)

    # Load dictionaries once before creating processes
    with open(paths_dict["objectid_to_label"], "r") as f:
        objectid_to_label = json.load(f)

    with open(paths_dict["objectid_to_color"], "r") as f:
        objectid_to_color = json.load(f)

    if num_processes == 0:
        for tfrecord_path in tfrecord_files:
            process_tfrecord(
                tfrecord_path, paths_dict, objectid_to_label, objectid_to_color
            )
    else:
        # Update the process args to include the dictionaries
        process_args = [
            (x, paths_dict, objectid_to_label, objectid_to_color)
            for x in tfrecord_files
        ]

        with mp_context.Pool(processes=num_processes) as pool:
            results = []
            for result in tqdm(
                pool.starmap(process_tfrecord, process_args),
                total=len(process_args),
                desc="Processing datasets",
            ):
                results.append(result)

        assert sum(results) == len(process_args)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Waymo dataset with CVAT labels"
    )
    parser.add_argument(
        "--validation-path",
        type=str,
        required=True,
        default="/media/local-data/uqdetche/waymo_open_dataset_v_1_4_3/validation/",
        help="Path to TFRecord validation files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--num-processes", type=int, default=1, help="Number of processes"
    )

    args = parser.parse_args()

    with tf.device("/CPU:0"):
        export_data(args.validation_path, args.output_dir, args.num_processes)


if __name__ == "__main__":
    main()
