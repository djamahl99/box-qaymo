import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
import cv2
import pytz
import numpy as np
from collections import defaultdict
import logging

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist


from waymovqa.data import *
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.frame_info import FrameInfo, TimeOfDayType, WeatherType, LocationType
from waymovqa.data.object_info import (
    OBJECT_SPEED_CATS,
    OBJECT_SPEED_THRESH,
    WAYMO_TYPE_MAPPING,
    HeadingType,
    MovementType,
    ObjectInfo,
)
from waymovqa.prompt_generators import BasePromptGenerator, register_prompt_generator

from waymovqa.questions.multi_image_multi_choice import (
    MultipleImageMultipleChoiceQuestion,
)
from waymovqa.questions.single_image_multi_choice import (
    SingleImageMultipleChoiceQuestion,
)
from waymovqa.questions.base import BaseQuestion
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.metrics.multiple_choice import MultipleChoiceMetric
from waymovqa.metrics.base import BaseMetric
from waymovqa.primitives import CHOICES_OPTIONS


# Default config values that can be overridden
DEFAULT_CONFIG = {
    "distance_threshold": 30.0,  # Threshold for similar trajectories
    "min_variance": 0.5,  # Minimum velocity for moving objects of 0.5 metres/second
    "min_trajectory_points": 3,  # Minimum number of points needed for a valid trajectory
    "max_objects_per_scene": 10,  # Maximum objects to consider per scene
    "consider_stationary_objects": True,  # Whether to include stationary objects in some questions
    "max_camera_distance": 800,  # pixels between query objects
}

HZ = 10.0
FRAME_DELTA = 1.0 / HZ


def did_pedestrian_cross_lane(
    ped_trajectory, left_lane_positions, right_lane_positions
):
    """
    Determine if pedestrian trajectory crosses the lane defined by left/right boundaries.

    Args:
        ped_trajectory: List of [x, y, z] positions for pedestrian
        left_lane_positions: List of [x, y, z] positions for left lane boundary
        right_lane_positions: List of [x, y, z] positions for right lane boundary

    Returns:
        dict with crossing info
    """
    ped_trajectory = np.array(ped_trajectory)
    left_lane = np.array(left_lane_positions)
    right_lane = np.array(right_lane_positions)

    crossing_events = []

    # Method 1: Check if pedestrian path intersects lane boundaries
    for i in range(len(ped_trajectory) - 1):
        ped_start = ped_trajectory[i]
        ped_end = ped_trajectory[i + 1]

        # Check intersection with left lane boundary
        left_intersections = find_line_intersections(ped_start, ped_end, left_lane)
        right_intersections = find_line_intersections(ped_start, ped_end, right_lane)

        if left_intersections or right_intersections:
            crossing_events.append(
                {
                    "frame_idx": i,
                    "position": ped_start,
                    "intersected_left": bool(left_intersections),
                    "intersected_right": bool(right_intersections),
                }
            )

    # Method 2: Check if pedestrian goes from one side of lane to the other
    side_changes = detect_lane_side_changes(ped_trajectory, left_lane, right_lane)

    return {
        "crossed_lane": len(crossing_events) > 0 or len(side_changes) > 0,
        "crossing_events": crossing_events,
        "side_changes": side_changes,
        "crossing_frames": [event["frame_idx"] for event in crossing_events],
    }


def find_line_intersections(point1, point2, lane_boundary):
    """Find where line segment (point1, point2) intersects with lane boundary."""
    intersections = []

    # Check intersection with each segment of the lane boundary
    for i in range(len(lane_boundary) - 1):
        lane_start = lane_boundary[i]
        lane_end = lane_boundary[i + 1]

        intersection = line_segment_intersection_2d(
            point1[:2], point2[:2], lane_start[:2], lane_end[:2]
        )

        if intersection is not None:
            intersections.append(intersection)

    return intersections


def line_segment_intersection_2d(p1, p2, p3, p4):
    """
    Find intersection point of two line segments in 2D.
    Returns None if no intersection, otherwise returns [x, y].
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:  # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y])

    return None


def detect_lane_side_changes(ped_trajectory, left_lane, right_lane):
    """
    Detect when pedestrian changes from one side of the lane to the other.
    """
    side_changes = []
    previous_side = None

    for i, ped_pos in enumerate(ped_trajectory):
        current_side = determine_lane_side(ped_pos, left_lane, right_lane)

        if (
            previous_side is not None
            and current_side != previous_side
            and current_side is not None
        ):
            side_changes.append(
                {
                    "frame_idx": i,
                    "from_side": previous_side,
                    "to_side": current_side,
                    "position": ped_pos,
                }
            )

        previous_side = current_side

    return side_changes


def determine_lane_side(point, left_lane, right_lane):
    """
    Determine which side of the lane a point is on.
    Returns 'left', 'right', 'inside', or None
    """
    # Find closest points on each lane boundary
    left_distances = cdist([point[:2]], left_lane[:, :2])[0]
    right_distances = cdist([point[:2]], right_lane[:, :2])[0]

    closest_left_idx = np.argmin(left_distances)
    closest_right_idx = np.argmin(right_distances)

    closest_left_point = left_lane[closest_left_idx]
    closest_right_point = right_lane[closest_right_idx]

    # Create a vector from right to left boundary (across the lane)
    lane_vector = closest_left_point[:2] - closest_right_point[:2]

    # Vector from right boundary to the point
    point_vector = point[:2] - closest_right_point[:2]

    # Cross product to determine side
    cross_product = np.cross(lane_vector, point_vector)

    # Also check if point is between the boundaries
    dist_to_left = np.linalg.norm(point[:2] - closest_left_point[:2])
    dist_to_right = np.linalg.norm(point[:2] - closest_right_point[:2])
    lane_width = np.linalg.norm(closest_left_point[:2] - closest_right_point[:2])

    if dist_to_left + dist_to_right <= lane_width * 1.1:  # Small tolerance
        return "inside"
    elif cross_product > 0:
        return "left"
    else:
        return "right"


@register_prompt_generator
class EgoRelativeObjectTrajectoryPromptGenerator(BasePromptGenerator):
    """
    Generates questions about object trajectories and their relationships to the ego vehicle.
    """

    negative_sample_ratio: float = 0.5

    def __init__(self, config=None):
        """
        Initialize the prompt generator with optional configuration.

        Args:
            config: Dictionary with configuration parameters that override defaults
        """
        super().__init__()
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.prompt_functions = [
            self._prompt_faster_than_ego,
            self._prompt_moving_towards_ego,
            self._prompt_parallel_motion,
            self._prompt_approaching_stop_sign,
            self._prompt_ego_following,
        ]

    def _balance_neg_pos(
        self, negative_samples: List[Dict], positive_samples: List[Dict]
    ):
        """Balance the negative and positive samples according to the target ratio."""
        num_neg = len(negative_samples)
        num_pos = len(positive_samples)

        if num_neg == 0 and num_pos == 0:
            return []

        if num_neg == 0:
            return positive_samples  # Can't balance without negatives

        if num_pos == 0:
            return negative_samples  # Can't balance without positives

        # Calculate the maximum total samples we can have while maintaining the ratio
        # Given the constraints of available samples

        # If we use all negatives, how many positives do we need?
        pos_needed_for_all_neg = int(
            num_neg * (1.0 - self.negative_sample_ratio) / self.negative_sample_ratio
        )

        # If we use all positives, how many negatives do we need?
        neg_needed_for_all_pos = int(
            num_pos * self.negative_sample_ratio / (1.0 - self.negative_sample_ratio)
        )

        # Choose the scenario that maximizes total samples while respecting constraints
        if pos_needed_for_all_neg <= num_pos:
            # We can use all negatives and sample the required positives
            final_negatives = negative_samples
            final_positives = random.sample(positive_samples, pos_needed_for_all_neg)
            print(
                f"Using all {num_neg} negatives, sampling {pos_needed_for_all_neg} positives from {num_pos} available"
            )

        elif neg_needed_for_all_pos <= num_neg:
            # We can use all positives and sample the required negatives
            final_positives = positive_samples
            final_negatives = random.sample(negative_samples, neg_needed_for_all_pos)
            print(
                f"Using all {num_pos} positives, sampling {neg_needed_for_all_pos} negatives from {num_neg} available"
            )

        else:
            # Neither scenario works perfectly, so we need to find the best balance
            # Calculate how many of each we can use while maintaining the ratio

            # Try to maximize total samples
            max_total_with_neg_constraint = num_neg / self.negative_sample_ratio
            max_total_with_pos_constraint = num_pos / (1.0 - self.negative_sample_ratio)

            # Use the smaller total (the limiting constraint)
            target_total = min(
                max_total_with_neg_constraint, max_total_with_pos_constraint
            )

            target_neg = int(target_total * self.negative_sample_ratio)
            target_pos = int(target_total * (1.0 - self.negative_sample_ratio))

            # Ensure we don't exceed available samples
            target_neg = min(target_neg, num_neg)
            target_pos = min(target_pos, num_pos)

            final_negatives = random.sample(negative_samples, target_neg)
            final_positives = random.sample(positive_samples, target_pos)
            print(
                f"Balanced sampling: {target_neg} negatives from {num_neg}, {target_pos} positives from {num_pos}"
            )

        # Combine and shuffle the results
        result = final_negatives + final_positives
        random.shuffle(result)

        actual_ratio = len(final_negatives) / len(result) if result else 0
        print(
            f"Final ratio: {actual_ratio:.3f} (target: {self.negative_sample_ratio:.3f})"
        )

        return result

    def _prompt_parallel_motion(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        angle_threshold=20,  # degrees for parallel/perpendicular classification
        min_speed=2.0,  # minimum speed to consider motion meaningful
    ):
        results = []
        choices = ["parallel", "perpendicular"]

        templates = [
            "Is the {} moving parallel or perpendicular to the ego vehicle?",
            "What is the relative motion direction of the {} compared to ego?",
        ]

        for obj_id in trajectories.keys():
            if obj_types[obj_id] not in ["Vehicle", "Pedestrian", "Cyclist"]:
                continue
            obj_trajectory = trajectories[obj_id]
            obj_speeds = speeds[obj_id]

            # Calculate relative distances over time
            relative_positions = obj_trajectory - ego_positions
            distances = np.linalg.norm(relative_positions, axis=1)

            template = random.choice(templates)

            # Find valid timestamps (close enough, both moving)
            valid_mask = (
                (distances < DEFAULT_CONFIG["distance_threshold"])
                & (obj_speeds > min_speed)
                & (ego_speeds > min_speed)
            )

            valid_indices = np.where(valid_mask)[0]
            # Need at least 2 points to calculate direction
            valid_indices = valid_indices[valid_indices < len(timestamps) - 1]

            if len(valid_indices) == 0:
                continue

            best_results = {"parallel": [], "perpendicular": []}

            for i in valid_indices:
                # Calculate ego motion vector
                if i + 1 < len(ego_positions):
                    ego_motion = ego_positions[i + 1] - ego_positions[i]
                else:
                    ego_motion = ego_vectors[i]  # Use velocity vector if at end

                # Calculate object motion vector
                if i + 1 < len(obj_trajectory):
                    obj_motion = obj_trajectory[i + 1] - obj_trajectory[i]
                else:
                    continue

                # Skip if either vector is too small
                if (
                    np.linalg.norm(ego_motion) < 0.01
                    or np.linalg.norm(obj_motion) < 0.01
                ):
                    continue

                # Calculate angle between motion vectors
                cos_angle = np.dot(ego_motion, obj_motion) / (
                    np.linalg.norm(ego_motion) * np.linalg.norm(obj_motion)
                )
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                # Handle obtuse angles (consider absolute angle for parallel/perpendicular)
                angle_deg = min(angle_deg, 180 - angle_deg)

                obj = next((obj for obj in frames[i].objects if obj.id == obj_id), None)
                if obj is None or obj.cvat_label is None:
                    continue

                if angle_deg < angle_threshold:  # Parallel motion
                    best_results["parallel"].append(
                        {
                            "object_id": obj_id,
                            "timestamp": int(timestamps[i]),
                            "question": template.format(obj.get_object_description()),
                            "answer_description": f"parallel motion at {angle_deg:.1f}° angle",
                            "answer": "parallel",
                            "confidence": float(
                                (angle_threshold - angle_deg) / angle_threshold
                            ),
                        }
                    )
                elif angle_deg > (90 - angle_threshold):  # Perpendicular motion
                    best_results["perpendicular"].append(
                        {
                            "object_id": obj_id,
                            "timestamp": int(timestamps[i]),
                            "question": template.format(obj.get_object_description()),
                            "answer_description": f"perpendicular motion at {angle_deg:.1f}° angle",
                            "answer": "perpendicular",
                            "confidence": float(
                                (angle_deg - (90 - angle_threshold)) / angle_threshold
                            ),
                        }
                    )

            # Add best result for each category if available
            for motion_type, motion_results in best_results.items():
                if len(motion_results) > 0:
                    best_result = sorted(
                        motion_results, key=lambda x: x["confidence"], reverse=True
                    )[0]
                    results.append(best_result)

        return choices, results

    def _prompt_ego_following(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        correlation_threshold=0.7,  # Correlation threshold for similar paths
        temporal_lag_max=10,
        min_trajectory_length=10,
        min_speed=5.0,
    ):
        results = []
        choices = ["yes", "no"]

        templates = [
            "Is the ego vehicle following the {}?",
            "Is the ego vehicle behind and following the {}?",
        ]

        for obj_id in trajectories.keys():
            if obj_types[obj_id] not in ["Vehicle", "Pedestrian", "Cyclist"]:
                continue
            obj_trajectory = trajectories[obj_id]
            obj_speeds = speeds[obj_id]

            template = random.choice(templates)

            # Filter for moving segments
            moving_mask = (obj_speeds > min_speed) & (ego_speeds > min_speed)
            if np.sum(moving_mask) < min_trajectory_length:
                continue

            best_correlation = -1
            best_lag = 0
            best_timestamp_idx = 0

            # Try different lags
            for lag in range(
                0, min(temporal_lag_max, len(obj_trajectory) - min_trajectory_length)
            ):
                obj_segment = obj_trajectory[lag : lag + min_trajectory_length]
                ego_segment = ego_positions[:min_trajectory_length]

                if len(obj_segment) != len(ego_segment):
                    continue

                # Calculate directional consistency first
                obj_velocity = np.diff(obj_segment, axis=0)  # Velocity vectors
                ego_velocity = np.diff(ego_segment, axis=0)  # Velocity vectors

                # Check if velocities are generally in same direction
                directional_consistency = []
                for i in range(len(obj_velocity)):
                    if (
                        np.linalg.norm(obj_velocity[i]) > 0.1
                        and np.linalg.norm(ego_velocity[i]) > 0.1
                    ):
                        # Dot product of normalized velocities
                        dot_product = np.dot(obj_velocity[i], ego_velocity[i]) / (
                            np.linalg.norm(obj_velocity[i])
                            * np.linalg.norm(ego_velocity[i])
                        )
                        directional_consistency.append(dot_product)

                if len(directional_consistency) == 0:
                    continue

                mean_directional_consistency = np.mean(directional_consistency)

                # Skip if moving in opposite directions (negative correlation)
                if mean_directional_consistency < 0.3:  # Threshold for same direction
                    continue

                # Now calculate path correlation (but don't use absolute value)
                correlations = []
                for dim in range(3):  # x, y, z
                    obj_dim = obj_segment[:, dim]
                    ego_dim = ego_segment[:, dim]

                    # Remove mean to focus on path shape
                    obj_centered = obj_dim - np.mean(obj_dim)
                    ego_centered = ego_dim - np.mean(ego_dim)

                    if np.std(obj_centered) > 0.1 and np.std(ego_centered) > 0.1:
                        corr = np.corrcoef(obj_centered, ego_centered)[0, 1]
                        if (
                            not np.isnan(corr) and corr > 0
                        ):  # Only positive correlations
                            correlations.append(corr)

                if len(correlations) > 0:
                    mean_correlation = np.mean(correlations)
                    # Combine correlation with directional consistency
                    combined_score = mean_correlation * mean_directional_consistency

                    if combined_score > best_correlation:
                        best_correlation = combined_score
                        best_lag = lag
                        best_timestamp_idx = lag + min_trajectory_length // 2

            if best_correlation > 0:
                timestamp = timestamps[min(best_timestamp_idx, len(timestamps) - 1)]

                obj = next(
                    (
                        obj
                        for obj in frames[best_timestamp_idx].objects
                        if obj.id == obj_id
                    ),
                    None,
                )
                if obj is None:
                    continue

                if best_correlation > correlation_threshold:
                    results.append(
                        {
                            "object_id": obj_id,
                            "timestamp": int(timestamp),
                            "question": template.format(obj.get_object_description()),
                            "answer_description": f"following similar path (score: {best_correlation:.2f})",
                            "answer": "yes",
                            "confidence": float(best_correlation),
                        }
                    )
                else:
                    results.append(
                        {
                            "object_id": obj_id,
                            "timestamp": int(timestamp),
                            "question": template.format(obj.get_object_description()),
                            "answer_description": f"different path/direction (score: {best_correlation:.2f})",
                            "answer": "no",
                            "confidence": float(1.0 - best_correlation),
                        }
                    )

        return choices, results

    def _prompt_approaching_stop_sign(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
    ):
        results = []
        choices = ["yes", "no"]

        templates = [
            "Is the ego vehicle approaching a stop sign?",
            "Is there a stop sign ahead that the ego vehicle is approaching?",
            "Should the ego vehicle come to a stop?",
        ]

        def _handle_sample(
            frame_idx: int,
            object_id: str,
            distance: float,
            relative_speed: float,
            positive: bool,
        ):
            """Handles generating a question for a positive/negative situation."""
            frame = frames[frame_idx]

            obj = next((obj for obj in frame.objects if obj.id == object_id), None)

            if obj is None:
                return None

            cam = next(
                (
                    cam
                    for cam in frame.cameras
                    if obj.get_camera_heading_direction(frame, cam, ignore_sign=False)
                    == HeadingType.TOWARDS
                    and obj.is_visible_on_camera(frame, cam)
                ),
                None,
            )

            if cam is None:
                return None

            template = random.choice(templates)

            if positive:
                answer_description = f"approaching stop sign at {relative_speed:.2f} m/s, currently {distance:.2f} metres away."
            else:
                answer_description = f"driving away from stop sign at {relative_speed:.2f} m/s, currently {distance:.2f} metres away."

            # DEBUGGING #############################
            img_vis = cv2.imread(cam.image_path)  # type: ignore
            x1, y1, x2, y2 = obj.get_object_bbox_2d(frame, cam)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 5)  # type: ignore
            cv2.putText(
                img_vis,
                f"Q: {template}",
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_vis,
                f"answer_description: {answer_description}",
                (0, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_vis,
                f"A: {'yes' if positive else 'no'}",
                (0, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_vis,
                "Camera: {cam.name}",
                (0, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite("stop_sign.jpg", img_vis)
            # DEBUGGING #############################

            return {
                "object_id": object_id,
                "positive": positive,
                "timestamp": int(frame.timestamp),
                "question": template,
                "answer_description": answer_description,
                "answer": "yes" if positive else "no",
                "confidence": float(
                    abs(relative_speed) / OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
                ),
            }

        positive_samples = []
        negative_samples = []

        # Find any stop signs
        stop_sign_object_ids = [
            object_id
            for object_id, cvat_label in obj_cvat_types.items()
            if cvat_label == "Stop Sign"
        ]

        print("stop_sign_object_ids", stop_sign_object_ids)

        # TODO: add frames where there are no stop signs (maybe do after, can then select a reasonable amount)

        for obj_id in stop_sign_object_ids:
            sign_positions = trajectories[obj_id]

            # vectors from ego -> sign
            relative_positions = (sign_positions - ego_positions).reshape(-1, 3)
            distances = np.linalg.norm(relative_positions, axis=1)
            distance_mask = distances < DEFAULT_CONFIG["distance_threshold"]

            # Check if approaching
            diff = np.concatenate(
                [np.zeros((1,), dtype=float), np.diff(distances)], axis=0
            )
            # relative speeds (speed between ego and sign)
            relative_speed = diff / FRAME_DELTA

            # is the difference in positions decreasing at the vehicle "moving" speed?
            is_approaching = relative_speed < (
                (-1) * OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
            )
            # is the ego moving away at a reasonable speed?
            is_moving_away = relative_speed > OBJECT_SPEED_THRESH["TYPE_VEHICLE"]

            # find where we are close to the sign and approaching
            valid_mask = distance_mask & is_approaching
            positive_idx = np.where(valid_mask)[0]

            # find negative samples where we are moving away
            negative_mask = is_moving_away
            negative_idx = np.where(negative_mask)[0]

            for frame_idx in positive_idx:
                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    positive=True,
                )

                if sample is not None:
                    positive_samples.append(sample)

            for frame_idx in negative_idx:
                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    positive=False,
                )

                if sample is not None:
                    negative_samples.append(sample)
        # Combine neg + pos (balanced)
        results = self._balance_neg_pos(negative_samples, positive_samples)

        return choices, results

    def _prompt_vehicle_same_lane(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
    ):
        results = []
        choices = ["yes", "no"]

        templates = []

        def _handle_sample(
            frame_idx: int,
            object_id: str,
            distance: float,
            relative_speed: float,
            positive: bool,
        ):
            """Handles generating a question for a positive/negative situation."""
            frame = frames[frame_idx]

            obj = next((obj for obj in frame.objects if obj.id == object_id), None)

            if obj is None:
                return None

            cam = next(
                (
                    cam
                    for cam in frame.cameras
                    if obj.get_camera_heading_direction(frame, cam, ignore_sign=False)
                    == HeadingType.TOWARDS
                    and obj.is_visible_on_camera(frame, cam)
                ),
                None,
            )

            if cam is None:
                return None

            template = random.choice(templates)

            if positive:
                answer_description = f"approaching stop sign at {relative_speed:.2f} m/s, currently {distance:.2f} metres away."
            else:
                answer_description = f"driving away from stop sign at {relative_speed:.2f} m/s, currently {distance:.2f} metres away."

            # DEBUGGING #############################
            img_vis = cv2.imread(cam.image_path)  # type: ignore
            x1, y1, x2, y2 = obj.get_object_bbox_2d(frame, cam)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 5)  # type: ignore
            cv2.putText(
                img_vis,
                f"Q: {template}",
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_vis,
                f"answer_description: {answer_description}",
                (0, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_vis,
                f"A: {'yes' if positive else 'no'}",
                (0, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_vis,
                "Camera: {cam.name}",
                (0, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite("same_lane.jpg", img_vis)
            # DEBUGGING #############################

            return {
                "object_id": object_id,
                "positive": positive,
                "timestamp": int(frame.timestamp),
                "question": template,
                "answer_description": answer_description,
                "answer": "yes" if positive else "no",
                "confidence": float(
                    abs(relative_speed) / OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
                ),
            }

        positive_samples = []
        negative_samples = []

        # Find any Pedestrians
        pedestrian_object_ids = [
            object_id for object_id, label in obj_types.items() if label == "Pedestrian"
        ]

        ego_position_gradient = np.gradient(ego_positions, axis=0)

        # Calculate the "road" based on the ego trajectory.
        road_left = []
        road_right = []

        for frame_idx, frame in enumerate(frames):
            world_pose = np.array(frame.pose, dtype=float).reshape(4, 4)

            ego_dir = world_pose[:3, 0]
            ego_y_axis = world_pose[:3, 1]

            left_edge = ego_positions[frame_idx] + 1.5 * ego_y_axis
            right_edge = ego_positions[frame_idx] - 1.5 * ego_y_axis

            road_left.append(left_edge)
            road_right.append(right_edge)

        road_left = np.stack(road_left, axis=0)
        road_right = np.stack(road_right, axis=0)

        for obj_id in pedestrian_object_ids:
            pedestrian_positions = trajectories[obj_id]

            # vectors from ego -> pedestrian
            relative_positions = (pedestrian_positions - ego_positions).reshape(-1, 3)
            distances = np.linalg.norm(relative_positions, axis=1)
            distance_mask = distances < DEFAULT_CONFIG["distance_threshold"]

            # Check if approaching
            diff = np.concatenate(
                [np.zeros((1,), dtype=float), np.diff(distances)], axis=0
            )
            # relative speeds (speed between ego and sign)
            relative_speed = diff / FRAME_DELTA

            # is the difference in positions decreasing at the vehicle "moving" speed?
            is_approaching = relative_speed < (
                (-1) * OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
            )
            # is the ego moving away at a reasonable speed?
            is_moving_away = relative_speed > OBJECT_SPEED_THRESH["TYPE_VEHICLE"]

            # find where we are close to the sign and approaching
            valid_mask = distance_mask & is_approaching
            positive_idx = np.where(valid_mask)[0]

            # find negative samples where we are moving away
            negative_mask = is_moving_away
            negative_idx = np.where(negative_mask)[0]

            for frame_idx in positive_idx:
                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    positive=True,
                )

                if sample is not None:
                    positive_samples.append(sample)

            for frame_idx in negative_idx:
                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    positive=False,
                )

                if sample is not None:
                    negative_samples.append(sample)
        # Combine neg + pos (balanced)
        results = self._balance_neg_pos(negative_samples, positive_samples)

        return choices, results

    def _prompt_faster_than_ego(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        threshold_factor=1.5,
    ):
        results = []
        choices = ["yes", "no"]

        templates = [
            "Is the {} moving faster than the ego vehicle?",
        ]

        for obj_id in trajectories.keys():
            if obj_types[obj_id] not in ["Vehicle", "Pedestrian", "Cyclist"]:
                continue
            obj_speeds = speeds[obj_id]  # Speed array over time
            obj_trajectory = trajectories[obj_id]

            # Calculate relative distances over time
            relative_positions = obj_trajectory - ego_positions
            distances = np.linalg.norm(relative_positions, axis=1)

            # Find timestamps where object is significantly faster
            speed_ratios = obj_speeds / (ego_speeds + 1e-6)
            faster_mask = speed_ratios > threshold_factor
            valid_mask = distances < DEFAULT_CONFIG["distance_threshold"]

            template = random.choice(templates)

            if np.any(
                faster_mask & valid_mask & (obj_speeds > 1.0)
            ):  # Only if moving reasonably fast
                # Pick best timestamp - highest speed difference
                best_idx = np.argmax(speed_ratios * (faster_mask & valid_mask))
                timestamp = timestamps[best_idx]
                speed_diff = obj_speeds[best_idx] - ego_speeds[best_idx]

                obj = next(
                    (obj for obj in frames[best_idx].objects if obj.id == obj_id), None
                )

                if obj is None:
                    continue

                results.append(
                    {
                        "object_id": obj_id,
                        "timestamp": int(timestamp),
                        "question": template.format(obj.get_object_description()),
                        "answer_description": f"faster by {speed_diff:.1f} m/s",
                        "answer": "yes",
                    }
                )

            # negative samples
            if np.any((~faster_mask) & valid_mask):
                # Pick best timestamp - highest speed difference
                best_idx = np.argmax(
                    (ego_speeds / (obj_speeds + 1e-6)) * ((~faster_mask) & valid_mask)
                )
                timestamp = timestamps[best_idx]
                speed_diff = ego_speeds[best_idx] - obj_speeds[best_idx]

                obj = next(
                    (obj for obj in frames[best_idx].objects if obj.id == obj_id), None
                )

                if obj is None:
                    continue

                results.append(
                    {
                        "object_id": obj_id,
                        "timestamp": int(timestamp),
                        "question": template.format(obj.get_object_description()),
                        "answer_description": f"slower by {speed_diff:.1f} m/s",
                        "answer": "no",
                    }
                )

        return choices, results

    def _prompt_moving_towards_ego(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        window_size=5,
    ):
        results = []
        choices = ["yes", "no"]

        def _get_scenario_and_templates(
            convergence_factor, obj_heading_to_ego, ego_heading_to_obj
        ):
            """Determine the specific scenario and return appropriate templates."""

            # Head-on collision scenario (very negative cosine similarity + strict conditions)
            if (
                convergence_factor < -0.95
                and obj_heading_to_ego > 0.95
                and ego_heading_to_obj > 0.95
            ):
                scenario = "head_on"
                templates = [
                    "Are the ego vehicle and the {} traveling directly towards each other?",
                    "Is the {} on a direct collision course with the ego vehicle?",
                    "Are the ego vehicle and the {} moving head-on towards each other?",
                ]
            # Object primarily approaching ego
            elif obj_heading_to_ego > 0.6 and ego_heading_to_obj < 0.3:
                scenario = "object_approaching"
                templates = [
                    "Is the {} moving towards the ego vehicle?",
                    "Is the {} approaching the ego vehicle?",
                    "Is the {} heading towards the ego vehicle?",
                ]
            # Ego primarily approaching object
            elif ego_heading_to_obj > 0.6 and obj_heading_to_ego < 0.3:
                scenario = "ego_approaching"
                templates = [
                    "Is the ego vehicle moving towards the {}?",
                    "Is the ego vehicle approaching the {}?",
                    "Is the ego vehicle heading towards the {}?",
                ]
            # General convergence
            else:
                scenario = "general_convergence"
                templates = [
                    "Are the ego vehicle and the {} getting closer to each other?",
                    "Is the distance between the ego vehicle and the {} decreasing?",
                    "Are the ego vehicle and the {} converging?",
                ]

            return scenario, templates

        def _get_divergence_templates(
            convergence_factor, obj_heading_to_ego, ego_heading_to_obj
        ):
            """Get templates for diverging scenarios."""

            # Moving in same direction (parallel)
            if convergence_factor > 0.7:
                scenario = "parallel_diverging"
                templates = [
                    "Are the ego vehicle and the {} moving in the same direction away from each other?",
                    "Is the {} moving away from the ego vehicle in the same direction?",
                ]
            # Object moving away from ego
            elif obj_heading_to_ego < -0.5:
                scenario = "object_diverging"
                templates = [
                    "Is the {} moving away from the ego vehicle?",
                    "Is the {} heading away from the ego vehicle?",
                ]
            # Ego moving away from object
            elif ego_heading_to_obj < -0.5:
                scenario = "ego_diverging"
                templates = [
                    "Is the ego vehicle moving away from the {}?",
                    "Is the ego vehicle heading away from the {}?",
                ]
            # General divergence
            else:
                scenario = "general_divergence"
                templates = [
                    "Are the ego vehicle and the {} moving apart?",
                    "Is the distance between the ego vehicle and the {} increasing?",
                ]

            return scenario, templates

        def _find_pivotal_frame(
            frame_indices,
            distances,
            relative_speeds,
            direction_dots,
            obj_headings,
            ego_headings,
        ):
            """Find the most representative frame from the available indices."""
            if len(frame_indices) == 0:
                return None

            # For positive samples, find frame with strongest convergence signal
            convergence_scores = []
            for idx in frame_indices:
                # Combine multiple factors for convergence strength
                distance_factor = max(
                    0, (50.0 - distances[idx]) / 50.0
                )  # Closer = higher score
                speed_factor = min(
                    abs(relative_speeds[idx]) / 5.0, 1.0
                )  # Faster = higher score
                direction_factor = max(
                    0, -direction_dots[idx]
                )  # More opposite = higher score
                heading_factor = max(
                    obj_headings[idx], ego_headings[idx]
                )  # Stronger heading = higher score

                total_score = (
                    distance_factor + speed_factor + direction_factor + heading_factor
                )
                convergence_scores.append(total_score)

            # Return index with highest convergence score
            best_idx_pos = np.argmax(convergence_scores)
            return frame_indices[best_idx_pos]

        def _handle_sample(
            frame_idx: int,
            object_id: str,
            distance: float,
            relative_speed: float,
            convergence_factor: float,
            obj_heading_to_ego: float,
            ego_heading_to_obj: float,
            positive: bool,
        ):
            """Handles generating a question for a positive/negative situation."""
            frame = frames[frame_idx]
            obj = next((obj for obj in frame.objects if obj.id == object_id), None)

            if obj is None:
                return None

            cam = next(
                (cam for cam in frame.cameras if obj.is_visible_on_camera(frame, cam)),
                None,
            )

            if cam is None:
                return None

            # Get scenario-specific templates
            if positive:
                scenario, templates = _get_scenario_and_templates(
                    convergence_factor, obj_heading_to_ego, ego_heading_to_obj
                )
                answer = "yes"
                answer_description = f"approaching {obj.get_object_description()} at {abs(relative_speed):.2f} m/s, currently {distance:.2f} metres away. Scenario: {scenario}"
            else:
                scenario, templates = _get_divergence_templates(
                    convergence_factor, obj_heading_to_ego, ego_heading_to_obj
                )
                answer = "no"
                answer_description = f"moving away from {obj.get_object_description()} at {abs(relative_speed):.2f} m/s, currently {distance:.2f} metres away. Scenario: {scenario}"

            template = random.choice(templates)

            # ENHANCED DEBUGGING VISUALIZATION #############################
            img_vis = cv2.imread(cam.image_path)  # type: ignore
            x1, y1, x2, y2 = obj.get_object_bbox_2d(frame, cam)

            # Color code by scenario
            scenario_colors = {
                "head_on": (0, 0, 255),  # Red
                "object_approaching": (255, 0, 0),  # Blue
                "ego_approaching": (0, 255, 0),  # Green
                "general_convergence": (0, 255, 255),  # Yellow
                "parallel_diverging": (255, 0, 255),  # Magenta
                "object_diverging": (128, 0, 128),  # Purple
                "ego_diverging": (0, 128, 128),  # Teal
                "general_divergence": (128, 128, 128),  # Gray
            }

            color = scenario_colors.get(scenario, (255, 255, 255))
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 5)  # type: ignore

            # Add comprehensive debugging info
            debug_info = [
                f"Scenario: {scenario}",
                f"Answer: {answer}",
                answer_description,
                f"Cosine similarity: {convergence_factor:.3f}",
                f"Obj->Ego heading: {obj_heading_to_ego:.3f}",
                f"Ego->Obj heading: {ego_heading_to_obj:.3f}",
                f"Distance: {distance:.1f}m",
                f"Rel speed: {relative_speed:.2f} m/s",
                f"Camera: {cam.name}",
                f"Frame: {frame_idx}",
            ]

            for i, info in enumerate(debug_info):
                cv2.putText(
                    img_vis,
                    info,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Add question at the bottom
            cv2.putText(
                img_vis,
                f"Q: {template.format(obj.get_object_description())}",
                (10, img_vis.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            filename = f"/home/djamahl/debug_images_ego_approaching/ego_approaching_{scenario}_{frame_idx}_{object_id[:8]}.jpg"
            print("filename", filename)
            cv2.imwrite(filename, img_vis)
            cv2.imwrite("ego_approaching.jpg", img_vis)
            print(f"Saved debug image: {filename}")
            # ENHANCED DEBUGGING VISUALIZATION #############################

            return {
                "object_id": object_id,
                "positive": positive,
                "scenario": scenario,
                "timestamp": int(frame.timestamp),
                "question": template.format(obj.get_object_description()),
                "answer_description": answer_description,
                "answer": answer,
                "confidence": float(
                    abs(relative_speed) / OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
                ),
                "convergence_factor": float(convergence_factor),
                "obj_heading_to_ego": float(obj_heading_to_ego),
                "ego_heading_to_obj": float(ego_heading_to_obj),
                "distance": float(distance),
                "relative_speed": float(relative_speed),
            }

        positive_samples = []
        negative_samples = []

        for obj_id in trajectories.keys():
            obj_type = obj_types[obj_id]
            if obj_type not in ["Vehicle", "Pedestrian", "Cyclist"]:
                continue

            if not np.any(speeds[obj_id] > 1.0):
                continue

            if obj_cvat_types[obj_id] is None:
                continue

            obj_trajectory = trajectories[obj_id]

            # Calculate relative distances over time
            relative_positions = obj_trajectory - ego_positions
            distances = np.linalg.norm(relative_positions, axis=1)

            # Use object-specific distance threshold
            distance_threshold = DEFAULT_CONFIG["distance_threshold"]
            distance_mask = distances < distance_threshold

            # Calculate relative speed (negative = approaching, positive = moving away)
            diff = np.concatenate(
                [np.zeros((1,), dtype=float), np.diff(distances)], axis=0
            )
            relative_speed = diff / FRAME_DELTA

            # Calculate movement directions for path convergence analysis
            obj_velocity = np.gradient(obj_trajectory, axis=0) / FRAME_DELTA
            ego_velocity = np.gradient(ego_positions, axis=0) / FRAME_DELTA

            # Calculate RELATIVE velocity (this is key!)
            relative_velocity = obj_velocity - ego_velocity

            # Calculate direction vectors (normalized)
            obj_direction = obj_velocity / (
                np.linalg.norm(obj_velocity, axis=1, keepdims=True) + 1e-8
            )
            ego_direction = ego_velocity / (
                np.linalg.norm(ego_velocity, axis=1, keepdims=True) + 1e-8
            )

            # BEST APPROACH: Use relative velocity direction
            relative_direction = relative_velocity / (
                np.linalg.norm(relative_velocity, axis=1, keepdims=True) + 1e-8
            )
            # Alternative: Check if relative velocity is pointing toward the other object
            obj_to_ego_vector = ego_positions - obj_trajectory
            obj_to_ego_normalized = obj_to_ego_vector / (
                np.linalg.norm(obj_to_ego_vector, axis=1, keepdims=True) + 1e-8
            )

            # Positive means relative velocity is pointing from obj toward ego (converging)
            convergence_factor = np.sum(
                relative_direction * obj_to_ego_normalized, axis=1
            )
            # bring back original direction
            convergence_factor = convergence_factor * -1

            print(f"Convergence factor: {convergence_factor}")
            print(f"Positive = converging, Negative = diverging")

            # Use this instead of the original convergence_factor
            paths_converging = convergence_factor < -0.3  # Objects getting closer
            paths_diverging = convergence_factor > 0.3  # Objects moving apart

            # Calculate vector from object to ego
            obj_to_ego = ego_positions - obj_trajectory
            obj_to_ego_normalized = obj_to_ego / (
                np.linalg.norm(obj_to_ego, axis=1, keepdims=True) + 1e-8
            )

            # Check if object is moving towards ego (dot product of obj direction with obj->ego vector)
            obj_heading_to_ego = np.sum(obj_direction * obj_to_ego_normalized, axis=1)

            # Check if ego is moving towards object (dot product of ego direction with ego->obj vector)
            ego_to_obj_normalized = -obj_to_ego_normalized
            ego_heading_to_obj = np.sum(ego_direction * ego_to_obj_normalized, axis=1)

            # Use object-specific speed thresholds
            speed_threshold = OBJECT_SPEED_THRESH["TYPE_VEHICLE"]

            # Enhanced approach detection combining distance and directional analysis
            distance_approaching = relative_speed <= (-speed_threshold)

            # Paths are converging if:
            # 1. Objects are moving in somewhat opposite directions (negative cosine similarity)
            # 2. OR one object is heading towards the other
            paths_converging = (
                (convergence_factor < -0.1)  # Moving in opposite-ish directions
                | (obj_heading_to_ego > 0.3)  # Object heading towards ego
                | (ego_heading_to_obj > 0.3)  # Ego heading towards object
            )

            is_approaching = distance_approaching & paths_converging

            # For moving away: distance increasing AND paths diverging
            distance_moving_away = relative_speed >= speed_threshold
            paths_diverging = (
                (convergence_factor > 0.5)  # Moving in similar directions
                & (obj_heading_to_ego < -0.1)  # Object heading away from ego
                & (ego_heading_to_obj < -0.1)  # Ego heading away from object
            )

            is_moving_away = distance_moving_away & paths_diverging

            # Additional filters for better quality samples

            # Object visibility and size requirements
            def _is_object_visible_and_significant(frame_idx, obj_id):
                frame = frames[frame_idx]
                obj = next((obj for obj in frame.objects if obj.id == obj_id), None)
                if obj is None:
                    return False

                if obj.num_lidar_points_in_box < 10:
                    return False

                # Find camera where object is visible
                visible_cameras = [
                    cam for cam in frame.cameras if obj.is_visible_on_camera(frame, cam)
                ]
                if not visible_cameras:
                    return False

                return True

            # Distance-based filtering by object type and scenario
            def _get_reasonable_distance_bounds(obj_type, scenario):
                if scenario == "head_on":
                    # Head-on scenarios should be closer and more urgent
                    if obj_type == "Pedestrian":
                        return (3.0, 25.0)  # Very close for pedestrians
                    elif obj_type == "Cyclist":
                        return (5.0, 35.0)
                    else:  # Vehicle
                        return (8.0, 50.0)
                else:
                    # Other scenarios can be further
                    if obj_type == "Pedestrian":
                        return (2.0, 30.0)
                    elif obj_type == "Cyclist":
                        return (3.0, 40.0)
                    else:  # Vehicle
                        return (5.0, 60.0)

            approach_consistency = (
                np.cumsum(is_approaching.astype(int)) >= 3
            )  # Reduced for stricter filtering
            away_consistency = np.cumsum(is_moving_away.astype(int)) >= 3

            # Filter out very slow movements (likely noise)
            significant_speed = np.abs(relative_speed) > (
                speed_threshold * 0.8
            )  # Increased threshold

            # Get scenario for this object trajectory to determine distance bounds
            temp_scenarios = []
            for idx in range(len(distances)):
                if is_approaching[idx]:
                    scenario, _ = _get_scenario_and_templates(
                        convergence_factor[idx],
                        obj_heading_to_ego[idx],
                        ego_heading_to_obj[idx],
                    )
                    temp_scenarios.append(scenario)

            # Use most common scenario or default
            most_common_scenario = (
                max(set(temp_scenarios), key=temp_scenarios.count)
                if temp_scenarios
                else "general_convergence"
            )
            min_dist, max_dist = _get_reasonable_distance_bounds(
                obj_type, most_common_scenario
            )

            # Ensure reasonable distance bounds for each object type and scenario
            reasonable_distance = (distances >= min_dist) & (distances <= max_dist)

            # Find positive samples (approaching and close)
            positive_mask = (
                distance_mask
                & is_approaching
                & approach_consistency
                & significant_speed
                & reasonable_distance
            )
            positive_idx = np.where(positive_mask)[0]

            # Find negative samples (moving away, any reasonable distance)
            negative_mask = (
                is_moving_away
                & away_consistency
                & significant_speed
                & (distances > 2.0)  # Not too close
                & (distances < distance_threshold * 2)  # Not too far
            )
            negative_idx = np.where(negative_mask)[0]

            # Find pivotal frames for each sample type
            pivotal_positive_idx = _find_pivotal_frame(
                positive_idx,
                distances,
                relative_speed,
                convergence_factor,
                obj_heading_to_ego,
                ego_heading_to_obj,
            )

            pivotal_negative_idx = _find_pivotal_frame(
                negative_idx,
                distances,
                relative_speed,
                convergence_factor,
                obj_heading_to_ego,
                ego_heading_to_obj,
            )

            # Process pivotal positive sample with visibility check
            if pivotal_positive_idx is not None:
                frame_idx = pivotal_positive_idx

                # Check if object is visible and significant enough
                if not _is_object_visible_and_significant(frame_idx, obj_id):
                    print(
                        f"Skipping frame {frame_idx} for object {obj_id} - not visible/significant enough"
                    )
                    continue

                print(f"\n=== POSITIVE SAMPLE (Frame {frame_idx}, Object {obj_id}) ===")
                print(f"convergence_factor: {convergence_factor[frame_idx]:.3f}")
                print(f"obj_heading_to_ego: {obj_heading_to_ego[frame_idx]:.3f}")
                print(f"ego_heading_to_obj: {ego_heading_to_obj[frame_idx]:.3f}")
                print(f"distance_approaching: {distance_approaching[frame_idx]}")
                print(f"distance: {distances[frame_idx]:.2f}m")
                print(f"relative_speed: {relative_speed[frame_idx]:.2f} m/s")

                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    convergence_factor[frame_idx],
                    obj_heading_to_ego[frame_idx],
                    ego_heading_to_obj[frame_idx],
                    positive=True,
                )

                if sample is not None:
                    positive_samples.append(sample)

            # Process pivotal negative sample with visibility check
            if pivotal_negative_idx is not None:
                frame_idx = pivotal_negative_idx

                # Check if object is visible and significant enough
                if not _is_object_visible_and_significant(frame_idx, obj_id):
                    print(
                        f"Skipping negative frame {frame_idx} for object {obj_id} - not visible/significant enough"
                    )
                    continue

                print(f"\n=== NEGATIVE SAMPLE (Frame {frame_idx}, Object {obj_id}) ===")
                print(f"convergence_factor: {convergence_factor[frame_idx]:.3f}")
                print(f"obj_heading_to_ego: {obj_heading_to_ego[frame_idx]:.3f}")
                print(f"ego_heading_to_obj: {ego_heading_to_obj[frame_idx]:.3f}")
                print(f"distance: {distances[frame_idx]:.2f}m")
                print(f"relative_speed: {relative_speed[frame_idx]:.2f} m/s")

                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    convergence_factor[frame_idx],
                    obj_heading_to_ego[frame_idx],
                    ego_heading_to_obj[frame_idx],
                    positive=False,
                )

                if sample is not None:
                    negative_samples.append(sample)

        print(
            f"\nGenerated {len(positive_samples)} positive and {len(negative_samples)} negative samples"
        )
        for sample in positive_samples:
            print(f"Positive - {sample['scenario']}: {sample['question']}")
        for sample in negative_samples:
            print(f"Negative - {sample['scenario']}: {sample['question']}")

        # Combine neg + pos (balanced)
        results = self._balance_neg_pos(negative_samples, positive_samples)

        return choices, results

    def generate(self, frames):
        """
        Generate questions about object trajectories.

        Args:
            frames: List of FrameInfo objects

        Returns:
            List of (question, answer) tuples
        """
        if not frames or len(frames) < self.config["min_trajectory_points"]:
            return []

        samples = []

        timestamps = np.array([frame.timestamp for frame in frames], dtype=int)

        (trajectories, speeds, accels, ego_positions, ego_speeds, ego_vectors) = (
            self._calculate_trajectories(frames, timestamps)
        )

        obj_types = {
            obj.id: obj.get_simple_type() for frame in frames for obj in frame.objects
        }

        obj_cvat_types = {
            obj.id: obj.cvat_label for frame in frames for obj in frame.objects
        }

        for prompt_func in self.prompt_functions:
            prompt_out = prompt_func(
                trajectories,
                speeds,
                accels,
                ego_positions,
                ego_speeds,
                ego_vectors,
                timestamps,
                frames,
                obj_types,
                obj_cvat_types,
            )

            #
            choices, questions = prompt_out

            for question_dict in questions:
                timestamp = question_dict["timestamp"]
                object_id = question_dict["object_id"]
                question_txt = question_dict["question"]
                answer_txt = question_dict["answer"]

                frame_idx = next(
                    (idx for idx, time in enumerate(timestamps) if time == timestamp),
                    None,
                )

                if frame_idx is None:
                    continue

                frame = frames[frame_idx]
                obj = next((obj for obj in frame.objects if obj.id == object_id), None)

                if obj is None:
                    continue

                assert obj.id == object_id

                camera = next(
                    (
                        cam
                        for cam in frame.cameras
                        if obj.most_visible_camera_name == cam.name
                    ),
                    None,
                )

                if camera is None:
                    continue

                if not obj.is_visible_on_camera(frame, camera):
                    continue

                if len(choices) > CHOICES_OPTIONS:
                    # Remove correct answer from choices
                    distractors = [x for x in choices if x != answer_txt]

                    # Sample (CHOICES_OPTIONS - 1) distractors
                    sampled = np.random.choice(
                        distractors, CHOICES_OPTIONS - 1, replace=False
                    ).tolist()

                    # Add the correct answer back in
                    choices = sampled + [answer_txt]

                    # (Optional) Shuffle the final choices
                    np.random.shuffle(choices)

                if len(choices) < 2:
                    print(f"Received only {len(choices)} choices from {prompt_func}")
                    continue

                if answer_txt is None or answer_txt not in choices:
                    print(
                        f"answer_txt={answer_txt} is none or not in choices={choices}"
                    )
                    continue

                question = MultipleImageMultipleChoiceQuestion(
                    image_paths=[camera.image_path for camera in frame.cameras],
                    question=question_txt,
                    choices=choices,
                    scene_id=obj.scene_id,
                    timestamp=frame.timestamp,
                    camera_names=[camera.name for camera in frame.cameras],
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                    data=dict(
                        question_dict=question_dict,
                        best_camera_name=camera.name,
                        object_id=object_id,
                        bbox=obj.get_object_bbox_2d(frame, camera),
                    ),
                )

                answer = MultipleChoiceAnswer(
                    choices=choices,
                    answer=answer_txt,
                )

                samples.append((question, answer))

        random.shuffle(samples)

        return samples

    def _calculate_trajectories(self, frames, timestamps):
        object_positions = defaultdict(dict)
        object_speeds = defaultdict(dict)
        object_accelerations = defaultdict(dict)
        ego_positions = []
        # Build object position lookup per timestamp
        for frame in frames:
            world_pose = np.array(frame.pose)

            ego_position = world_pose[:3, -1]
            ego_positions.append(ego_position)

            for obj in frame.objects:
                if (
                    obj
                    and obj.id
                    and (
                        obj.get_simple_type()
                        in ["Vehicle", "Pedestrian", "Cyclist", "Sign"]
                    )  # include sign for stop sign question
                ):  # Ensure object exists and has ID
                    try:
                        object_positions[obj.id][frame.timestamp] = (
                            obj.get_world_centre(frame)
                        )
                        object_speeds[obj.id][frame.timestamp] = obj.get_speed()
                        object_accelerations[obj.id][frame.timestamp] = obj.get_accel()
                    except (AttributeError, ValueError) as e:
                        pass

        ego_positions = np.stack(ego_positions, axis=0)
        ego_speeds = [0.0]
        ego_vectors = [np.zeros((3,), dtype=float)]
        last_pos = ego_positions[0]

        for pos in ego_positions[1:]:
            displacement = pos - last_pos  # Displacement vector (meters)
            distance = np.linalg.norm(displacement)  # Distance (meters)
            speed = float(distance / FRAME_DELTA)  # Speed (m/s)

            # Vector should be normalized displacement, not divided by time
            if distance > 1e-6:  # Avoid division by zero
                direction_vector = displacement / distance  # Unit direction vector
            else:
                direction_vector = np.zeros((3,), dtype=float)

            ego_speeds.append(speed)
            ego_vectors.append(
                direction_vector
            )  # Or just use displacement if you want magnitude
            last_pos = pos

        ego_speeds = np.array(ego_speeds).reshape(-1)
        ego_vectors = np.stack(ego_vectors, axis=0)

        # Fill in trajectories, holding the last known position
        full_object_trajectories = {}
        full_object_speeds = {}
        full_object_accels = {}
        for obj_id, position_by_time in object_positions.items():
            object_ts = np.array(
                [float(x) for x in position_by_time.keys()], dtype=float
            )

            object_positions_array = np.array(
                [x for x in position_by_time.values()], dtype=float
            )

            object_speed_array = np.array(
                [x for x in object_speeds[obj_id].values()], dtype=float
            )

            object_accel_array = np.array(
                [x for x in object_accelerations[obj_id].values()], dtype=float
            )

            trajectory = []
            accel = []
            for i in range(3):
                vals = np.interp(timestamps, object_ts, object_positions_array[:, i])
                trajectory.append(vals)

                if i < object_accel_array.shape[1]:
                    vals = np.interp(timestamps, object_ts, object_accel_array[:, i])
                    accel.append(vals)

            trajectory = np.stack(trajectory, axis=1)
            assert trajectory.shape[0] == len(timestamps)

            full_object_trajectories[obj_id] = trajectory
            full_object_speeds[obj_id] = np.interp(
                timestamps, object_ts, object_speed_array.reshape(-1)
            )
            full_object_accels[obj_id] = np.stack(accel, axis=1)

        return (
            full_object_trajectories,
            full_object_speeds,
            full_object_accels,
            ego_positions,
            ego_speeds,
            ego_vectors,
        )

    def visualize_trajectories(
        self,
        object_id,
        trajectories,
        ego_positions,
        frame,
        camera,
        timestamps: np.ndarray,
        save_path=None,
        question_txt="",
        answer_txt="",
        answer_description="",
    ):
        """
        Visualize object trajectories for debugging purposes.

        Creates a figure with two subplots:
        1. 3D plot showing the trajectories in world coordinates
        2. 2D plot showing the trajectories projected onto a camera image

        Args:
            object_id: ID of the main object
            trajectories: Dictionary mapping object IDs to trajectory arrays
            frame: frameinfo chosen
            timestamps: timestamps
            save_path: Optional path to save the figure, if None the figure is displayed

        Returns:
            matplotlib.figure.Figure object
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.image as mpimg
        import os
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np

        # frame, camera, obj = best_object_views[object_id]
        obj = next(obj for obj in frame.objects if obj.id == object_id)

        timestamp_idx = next(
            idx for idx, time in enumerate(timestamps) if time == frame.timestamp
        )

        # Get frame and camera for visualization
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 8))

        # Get trajectories
        traj1 = trajectories[object_id]
        traj2 = ego_positions

        # Get object descriptions for labels
        obj1_desc = obj.get_object_description()
        obj2_desc = "Ego"

        # 1. 3D plot in world coordinates
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")

        # Create custom colormaps for fading effect
        n_points = len(traj1)
        colors1 = plt.cm.viridis(np.linspace(0, 1, n_points))
        colors2 = plt.cm.plasma(np.linspace(0, 1, n_points))

        # Plot trajectories with fading colors to show time progression
        for i in range(len(traj1) - 1):
            # Object 1 trajectory
            ax1.plot(
                traj1[i : i + 2, 0],
                traj1[i : i + 2, 1],
                traj1[i : i + 2, 2] if traj1.shape[1] > 2 else [0, 0],
                color=colors1[i],
                linewidth=2,
            )

            # Object 2 trajectory
            ax1.plot(
                traj2[i : i + 2, 0],
                traj2[i : i + 2, 1],
                traj2[i : i + 2, 2] if traj2.shape[1] > 2 else [0, 0],
                color=colors2[i],
                linewidth=2,
            )

        # Add start and end markers
        ax1.scatter(
            traj1[0, 0],
            traj1[0, 1],
            traj1[0, 2] if traj1.shape[1] > 2 else 0,
            color="green",
            s=100,
            marker="o",
            label=f"{obj1_desc} (start)",
        )
        ax1.scatter(
            traj1[timestamp_idx, 0],
            traj1[timestamp_idx, 1],
            traj1[timestamp_idx, 2] if traj1.shape[1] > 2 else 0,
            color="blue",
            s=100,
            marker="v",
            label=f"{obj1_desc} (current)",
        )
        ax1.scatter(
            traj1[-1, 0],
            traj1[-1, 1],
            traj1[-1, 2] if traj1.shape[1] > 2 else 0,
            color="darkgreen",
            s=100,
            marker="s",
            label=f"{obj1_desc} (end)",
        )

        ax1.scatter(
            traj2[0, 0],
            traj2[0, 1],
            traj2[0, 2] if traj2.shape[1] > 2 else 0,
            color="red",
            s=100,
            marker="o",
            label=f"{obj2_desc} (start)",
        )
        ax1.scatter(
            traj2[timestamp_idx, 0],
            traj2[timestamp_idx, 1],
            traj2[timestamp_idx, 2] if traj1.shape[1] > 2 else 0,
            color="red",
            s=100,
            marker="v",
            label=f"{obj1_desc} (current)",
        )
        ax1.scatter(
            traj2[-1, 0],
            traj2[-1, 1],
            traj2[-1, 2] if traj2.shape[1] > 2 else 0,
            color="darkred",
            s=100,
            marker="s",
            label=f"{obj2_desc} (end)",
        )

        # Set labels and title
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Object Trajectories in 3D Space")
        ax1.legend()

        # Equal aspect ratio for all axes
        max_range = max(
            [
                np.max([np.ptp(traj1[:, 0]), np.ptp(traj2[:, 0])]),
                np.max([np.ptp(traj1[:, 1]), np.ptp(traj2[:, 1])]),
                np.max(
                    [
                        np.ptp(
                            traj1[:, 2] if traj1.shape[1] > 2 else np.zeros(len(traj1))
                        ),
                        np.ptp(
                            traj2[:, 2] if traj2.shape[1] > 2 else np.zeros(len(traj2))
                        ),
                    ]
                ),
            ]
        )

        mid_x = (np.mean(traj1[:, 0]) + np.mean(traj2[:, 0])) / 2
        mid_y = (np.mean(traj1[:, 1]) + np.mean(traj2[:, 1])) / 2
        mid_z = (
            np.mean(traj1[:, 2] if traj1.shape[1] > 2 else np.zeros(len(traj1)))
            + np.mean(traj2[:, 2] if traj2.shape[1] > 2 else np.zeros(len(traj2)))
        ) / 2

        ax1.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax1.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax1.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # 2. Image projection plot
        ax2 = fig.add_subplot(1, 2, 2)

        # Load the camera image
        image_path = camera.image_path
        try:
            if os.path.exists(image_path):
                img = mpimg.imread(image_path)
                ax2.imshow(img)
            else:
                # If image doesn't exist, create a blank image
                ax2.set_facecolor("black")
                print(f"Warning: Image file not found at {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            ax2.set_facecolor("black")

        # Project 3D trajectories onto the image
        try:
            pose_matrix_inv = np.linalg.inv(np.array(frame.pose))

            traj1_ego = traj1.copy().reshape(-1, 3)
            traj1_ego = np.hstack([traj1_ego, np.ones((traj1_ego.shape[0], 1))])
            traj1_ego = np.matmul(traj1_ego, pose_matrix_inv.T)
            traj1_ego = traj1_ego[:, :3]

            traj2_ego = traj2.copy().reshape(-1, 3)
            traj2_ego = np.hstack([traj2_ego, np.ones((traj2_ego.shape[0], 1))])
            traj2_ego = np.matmul(traj2_ego, pose_matrix_inv.T)
            traj2_ego = traj2_ego[:, :3]

            # Project to image
            projected_points1 = camera.project_to_image(traj1_ego, frame)
            projected_points2 = camera.project_to_image(traj2_ego, frame)

            # Extract x, y coordinates (and depth if available)
            if projected_points1.shape[1] > 2:  # If depth is returned
                img_points1 = projected_points1[:, :2]
                img_points2 = projected_points2[:, :2]

                # Adjust colors based on depth
                colors1 = plt.cm.viridis(np.linspace(0, 1, len(img_points1)))
                colors2 = plt.cm.plasma(np.linspace(0, 1, len(img_points2)))
            else:
                img_points1 = projected_points1
                img_points2 = projected_points2

            # Plot trajectory points on image
            for i in range(len(img_points1) - 1):
                ax2.plot(
                    img_points1[i : i + 2, 0],
                    img_points1[i : i + 2, 1],
                    color=colors1[i],
                    linewidth=2,
                    alpha=0.8,
                )

            for i in range(len(img_points2) - 1):
                ax2.plot(
                    img_points2[i : i + 2, 0],
                    img_points2[i : i + 2, 1],
                    color=colors2[i],
                    linewidth=2,
                    alpha=0.8,
                )

            # Add start and end markers
            if len(img_points1) > 0:
                ax2.scatter(
                    img_points1[0, 0],
                    img_points1[0, 1],
                    color="green",
                    s=100,
                    marker="o",
                    label=f"{obj1_desc} (start)",
                )
                ax2.scatter(
                    img_points1[timestamp_idx, 0],
                    img_points1[timestamp_idx, 1],
                    color="green",
                    s=100,
                    marker="v",
                    label=f"{obj1_desc} (current)",
                )
                ax2.scatter(
                    img_points1[-1, 0],
                    img_points1[-1, 1],
                    color="darkgreen",
                    s=100,
                    marker="s",
                    label=f"{obj1_desc} (end)",
                )

            if len(img_points2) > 0:
                ax2.scatter(
                    img_points2[0, 0],
                    img_points2[0, 1],
                    color="red",
                    s=100,
                    marker="o",
                    label=f"{obj2_desc} (start)",
                )
                ax2.scatter(
                    img_points2[timestamp_idx, 0],
                    img_points2[timestamp_idx, 1],
                    color="red",
                    s=100,
                    marker="v",
                    label=f"{obj2_desc} (current)",
                )
                ax2.scatter(
                    img_points2[-1, 0],
                    img_points2[-1, 1],
                    color="darkred",
                    s=100,
                    marker="s",
                    label=f"{obj2_desc} (end)",
                )

        except Exception as e:
            print(f"Error projecting points to image: {e}")

        # Set title and legend
        ax2.set_title(f"Trajectories Projected on {camera.name} Camera")
        ax2.legend(loc="upper right")

        # Set axes limits to match image dimensions if available
        if hasattr(camera, "width") and hasattr(camera, "height"):
            ax2.set_xlim(0, camera.width)
            ax2.set_ylim(camera.height, 0)  # Inverted y-axis for image coordinates

        # Add metadata as text
        info_text = (
            f"Scene ID: {obj.scene_id}\n"
            f"Timestamp: {frame.timestamp}\n"
            f"Object: {obj1_desc}\n"
            f"Camera: {camera.name}\n"
            f"Question: {question_txt}\n"
            f"Answer: {answer_txt}\n"
            f"answer_description: {answer_description}\n"
        )

        plt.figtext(
            0.01,
            0.01,
            info_text,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )

        # Adjust layout
        plt.tight_layout()

        # Save or show figure
        if save_path:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    def visualise_sample(
        self,
        question_obj: MultipleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        figsize=(12, 8),
        box_color="green",
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Visualises trajectories etc.

        Args:
            question_obj: Question object with image_path and question text
            answer_obj: MultipleChoiceAnswer
            save_path: Path to save the visualization
            frames:
            figsize: Figure size (width, height)
            box_color: Color for bounding boxes
            text_fontsize: Font size for question/answer text
            title_fontsize: Font size for the title
            dpi: DPI for saved image
        """
        if question_obj.data is None:
            raise ValueError("question_obj.data should not be none")

        object_id = question_obj.data["object_id"]
        best_camera_name = question_obj.data["best_camera_name"]
        answer_description = question_obj.data["question_dict"]["answer_description"]

        timestamps = np.array([frame.timestamp for frame in frames], dtype=int)

        (trajectories, speeds, accels, ego_positions, ego_speeds, ego_vectors) = (
            self._calculate_trajectories(frames, timestamps)
        )

        frame = next(
            frame for frame in frames if frame.timestamp == question_obj.timestamp
        )

        best_camera = next(
            camera for camera in frame.cameras if camera.name == best_camera_name
        )

        question_txt = question_obj.question
        answer_txt = answer_obj.answer

        self.visualize_trajectories(
            object_id,
            trajectories,
            ego_positions,
            frame,
            best_camera,
            timestamps,
            save_path=save_path,
            question_txt=question_txt,
            answer_txt=answer_txt,
            answer_description=answer_description,
        )

    def get_question_type(self):
        """Return the type of questions generated by this generator."""
        return MultipleImageMultipleChoiceQuestion

    def get_answer_type(self):
        """Return the type of answers generated by this generator."""
        return MultipleChoiceAnswer

    def get_metric_class(self):
        """Return the metric class used to evaluate answers to these questions."""
        return MultipleChoiceMetric
