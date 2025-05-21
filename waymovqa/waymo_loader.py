from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo

class WaymoDatasetLoader:
    """Helper class to load and query dataset files."""
    object_id_to_scene_id: Dict = None

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)

        assert self.base_path.exists(), f'{base_path} does not exist!'

        self.scene_infos_path = self.base_path / "scene_infos"
        self.frame_infos_path = self.base_path / "frame_infos"
        self.object_infos_path = self.base_path / "object_infos"
        self.camera_images_path = self.base_path / "camera_images"
        self.point_clouds_path = self.base_path / "point_clouds"
        self.object_lists_path = self.base_path / "object_lists"

        self.scenes = {}  # Cache for loaded scenes
        self.frames = {}  # Cache for loaded frames
        self.objects = {}  # Cache for loaded objects

    def get_scene_ids(self) -> List[str]:
        """Get list of all scene IDs in the dataset."""
        scene_files = list(self.scene_infos_path.glob("*.json"))
        return [file.stem for file in scene_files]

    def load_scene(self, scene_id: str) -> SceneInfo:
        """Load scene by ID."""
        if scene_id in self.scenes:
            return self.scenes[scene_id]

        scene_path = self.scene_infos_path / f"{scene_id}.json"
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene {scene_id} not found at {scene_path}")

        scene = SceneInfo.load(scene_path)
        self.scenes[scene_id] = scene
        return scene

    def load_frame(self, scene_id: str, timestamp: int) -> FrameInfo:
        """Load frame by scene ID and timestamp."""
        frame_key = f"{scene_id}_{timestamp}"
        if frame_key in self.frames:
            return self.frames[frame_key]

        frame_path = self.frame_infos_path / f"{frame_key}.json"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame {frame_key} not found at {frame_path}")

        frame = FrameInfo.load(frame_path)
        self.frames[frame_key] = frame
        return frame

    def load_object(self, object_id: str, scene_id: str, timestamp: int) -> ObjectInfo:
        """Load object by ID and scene ID."""
        obj_key = f"{object_id}_{scene_id}_{timestamp}"
        # if obj_key in self.objects:
            # return self.objects[obj_key]

        obj_path = self.object_infos_path / f"object_{obj_key}.json"
        if not obj_path.exists():
            raise FileNotFoundError(f"Object {obj_key} not found at {obj_path}")

        obj = ObjectInfo.load(obj_path)
        self.objects[obj_key] = obj
        return obj

    def load_all_objects_with_cvat_ids(self) -> List[str]:
        """Load all objects with CVAT labels for a scene."""
        cvat_paths = list(self.object_lists_path.rglob("*_all_cvat_objects.txt"))
        
        assert len(cvat_paths) == 202, f'got {len(cvat_paths)} cvat paths'

        # Track object_ids over all scenes with a set as there might be duplicates
        object_ids = set()
        object_id_to_scene_id = {}

        for cvat_path in cvat_paths:
            if not cvat_path.exists():
                continue

            scene_id = cvat_path.name.replace('_all_cvat_objects.txt', '')

            with open(cvat_path, "r") as f:
                scene_object_ids = [line.strip() for line in f if line.strip()]

            for scene_object_id in scene_object_ids:
                object_id_to_scene_id[scene_object_id] = scene_id

            object_ids.update(scene_object_ids)

        self.object_id_to_scene_id = object_id_to_scene_id

        # Save for future
        object_id_to_scene_id_path = self.base_path / 'object_id_to_scene_id.json'
        if not object_id_to_scene_id_path.exists():
            with open(object_id_to_scene_id_path, 'w') as f:
                json.dump(object_id_to_scene_id, f)

        object_ids = list(object_ids)

        return object_ids
    
    def get_object_id_to_scene_id(self) -> Dict[str, str]:
        """Gets mapping from object ids to scene ids."""
        if self.object_id_to_scene_id is not None:
            return self.object_id_to_scene_id
        
        object_id_to_scene_id_path = self.base_path / 'object_id_to_scene_id.json'

        if not object_id_to_scene_id_path.exists():
            self.load_all_objects_with_cvat_ids()

            return self.object_id_to_scene_id
        else:
            with open(object_id_to_scene_id_path, 'r') as f:
                self.object_id_to_scene_id = json.load(f)

            return self.object_id_to_scene_id

    def load_objects_with_cvat(self, scene_id: str) -> List[ObjectInfo]:
        """Load all objects with CVAT labels for a scene."""
        cvat_path = self.object_lists_path / f"{scene_id}_all_cvat_objects.txt"
        if not cvat_path.exists():
            return []

        with open(cvat_path, "r") as f:
            object_ids = [line.strip() for line in f if line.strip()]

        objects = []
        for obj_id in object_ids:
            try:
                obj = self.load_object(obj_id, scene_id)
                objects.append(obj)
            except FileNotFoundError:
                # Skip objects that can't be found
                continue

        return objects

    def load_objects_with_color(self, scene_id: str) -> List[ObjectInfo]:
        """Load all objects with color information for a scene."""
        color_path = self.object_lists_path / f"{scene_id}_all_color_objects.txt"
        if not color_path.exists():
            return []

        with open(color_path, "r") as f:
            object_ids = [line.strip() for line in f if line.strip()]

        objects = []
        for obj_id in object_ids:
            try:
                obj = self.load_object(obj_id, scene_id)
                objects.append(obj)
            except FileNotFoundError:
                # Skip objects that can't be found
                continue

        return objects

    def load_frame_object_table(
        self, scene_id: str, timestamp: int
    ) -> List[Dict[str, Any]]:
        """Load the object table for a specific frame."""
        table_path = (
            self.object_lists_path / f"{scene_id}_{timestamp}_object_table.json"
        )
        if not table_path.exists():
            return []

        with open(table_path, "r") as f:
            return json.load(f)

    def load_scene_object_table(self, scene_id: str) -> List[Dict[str, Any]]:
        """Load the aggregated object table for an entire scene."""
        table_path = self.object_lists_path / f"{scene_id}_all_object_table.json"
        if not table_path.exists():
            return []

        with open(table_path, "r") as f:
            return json.load(f)

    def get_frame_timestamps(self, scene_id: str) -> List[int]:
        """Get all frame timestamps for a scene."""
        prefix = f"{scene_id}_"
        frame_files = [f for f in self.frame_infos_path.glob(f"{prefix}*.json")]
        return [int(f.stem.replace(prefix, "")) for f in frame_files]

    def get_image_path(self, camera_info: CameraInfo, timestamp: int = None) -> str:
        """Load camera image from path."""
        if timestamp is not None:
            # Construct image path based on scene_id, timestamp, and camera name
            img_path = (
                self.camera_images_path
                / f"{camera_info.scene_id}_{timestamp}_{camera_info.name}.jpg"
            )
        else:
            # Use the path from camera_info
            if not camera_info.image_path:
                raise ValueError("Camera info does not have an image path")
            img_path = self.camera_images_path.parent / camera_info.image_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        return str(img_path)

    def load_image(self, camera_info: CameraInfo, timestamp: int = None) -> np.ndarray:
        """Load camera image from path."""
        img_path = self.get_image_path(camera_info, timestamp)

        return cv2.imread(str(img_path))

    def load_point_cloud(
        self, laser_info: LaserInfo, timestamp: int = None
    ) -> np.ndarray:
        """Load point cloud from path."""
        if timestamp is not None:
            # Construct point cloud path based on scene_id, timestamp, and laser name
            pc_path = (
                self.point_clouds_path
                / f"{laser_info.scene_id}_{timestamp}_{laser_info.name}.npy"
            )
        else:
            # Use the path from laser_info
            if not laser_info.point_cloud_path:
                raise ValueError("Laser info does not have a point cloud path")
            pc_path = self.point_clouds_path.parent / laser_info.point_cloud_path

        if not pc_path.exists():
            raise FileNotFoundError(f"Point cloud not found at {pc_path}")

        return np.load(str(pc_path))
