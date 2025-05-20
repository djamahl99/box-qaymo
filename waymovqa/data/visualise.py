import cv2
import numpy as np

faces = [
    [0, 1, 2, 3],  # bottom
    [4, 5, 6, 7],  # top
    [0, 1, 5, 4],  # front
    [2, 3, 7, 6],  # back
    [0, 3, 7, 4],  # left
    [1, 2, 6, 5],  # right
]

lines = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def draw_3d_wireframe_box_cv(img, u, v, color, thickness=3):
    """Draws 3D wireframe bounding boxes onto the given image."""
    # List of lines to interconnect. Allows for various forms of connectivity.
    # Four lines each describe bottom face, top face, and vertical connectors.

    for point_idx1, point_idx2 in lines:

        pt1 = (u[point_idx1], v[point_idx1])
        pt2 = (u[point_idx2], v[point_idx2])

        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        img = cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    return img

def is_face_visible(face, verts_cam):
    a, b, c = [verts_cam[i] for i in face[:3]]
    normal = np.cross(b - a, c - a)
    view_dir = a  # assuming camera is at origin
    return np.dot(normal, view_dir) < 0


def colorize_and_save_depth_buffer(depth_buffer, output_path, colormap=cv2.COLORMAP_JET):
    """
    Colorizes a depth buffer and saves it as an image.

    Args:
        depth_buffer (np.ndarray): The depth buffer (2D array) to be colorized.
        output_path (str): The file path to save the colorized depth image.
        colormap (int): OpenCV colormap to apply (default is cv2.COLORMAP_JET).

    Returns:
        colorized_depth (np.ndarray): The colorized depth image.
    """
    # Replace infinite values with a large number (for visualization purposes)
    if np.isfinite(depth_buffer).sum() > 0:
        depth_buffer[np.isinf(depth_buffer)] = np.max(depth_buffer[np.isfinite(depth_buffer)])
    else:
        # all inf
        depth_buffer[:, :] = 0.0

    # Normalize the depth buffer to the range [0, 255]
    normalized_depth = cv2.normalize(depth_buffer, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap
    colorized_depth = cv2.applyColorMap(normalized_depth, colormap)

    # Save the colorized depth image
    cv2.imwrite(output_path, colorized_depth)

    return colorized_depth

def generate_object_depth_buffer(frame, camera):
    depth_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float32)  # Z-buffer for occlusion handling

    for obj in frame.objects:
        # obj projected to the camera
        uvdok = obj.project_to_image(frame, camera, return_depth=True)
        uvs, depth, ok = uvdok[..., 0:2], uvdok[..., 2], uvdok[..., 3]
        ok = ok.astype(bool)

        uvs_ok = uvs[ok]
        depth_ok = depth[ok]
        depth_mean = depth_ok.mean()

        # Convert to int and proper format for OpenCV
        hull_input = uvs_ok.astype(np.int32)
        if len(hull_input) < 3:
            continue

        hull = cv2.convexHull(hull_input.reshape(-1, 1, 2))

        # Create a temporary mask and apply depth
        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)

        # Apply mean depth to masked region only if it's closer (Z-buffering)
        update_mask = (mask == 255) & (depth_buffer > depth_mean)
        depth_buffer[update_mask] = depth_mean

    print('depth_buffer', depth_buffer.shape)

    colorize_and_save_depth_buffer(depth_buffer, 'depth.png')

    return depth_buffer


                