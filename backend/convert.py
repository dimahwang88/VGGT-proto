"""Turn raw VGGT outputs into transport formats: cameras JSON + a GLB point cloud."""

from __future__ import annotations

import io

import numpy as np
import trimesh


def cameras_to_json(extr: np.ndarray, intr: np.ndarray) -> list[dict]:
    """Per-image camera parameters.

    ``extr``: (N, 3, 4) world-to-camera (OpenCV convention) from VGGT.
    ``intr``: (N, 3, 3) pixel intrinsics for the preprocessed frame size.

    For each image we return the 3x3 intrinsics, the 4x4 world-to-camera
    matrix, and its inverse (camera-to-world) so the viewer can place a
    frustum without doing the matrix inverse itself.
    """
    out: list[dict] = []
    for i in range(extr.shape[0]):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = extr[i]
        cam2world = np.linalg.inv(w2c)
        out.append(
            {
                "image_id": i,
                "intrinsic": intr[i].astype(float).tolist(),       # 3x3
                "extrinsic": w2c.astype(float).tolist(),           # 4x4 world->cam
                "cam_to_world": cam2world.astype(float).tolist(),  # 4x4 cam->world
            }
        )
    return out


def build_glb(points_xyz: np.ndarray, points_rgb: np.ndarray) -> bytes:
    """Colored point cloud -> binary glTF (.glb) bytes.

    trimesh exports a PointCloud as glTF POINTS, which three.js loads directly.
    """
    if points_xyz.shape[0] == 0:
        # Degenerate: a single origin point keeps the loader happy.
        points_xyz = np.zeros((1, 3), dtype=np.float32)
        points_rgb = np.zeros((1, 3), dtype=np.uint8)

    rgba = np.concatenate(
        [points_rgb.astype(np.uint8),
         np.full((points_rgb.shape[0], 1), 255, dtype=np.uint8)],
        axis=1,
    )
    cloud = trimesh.PointCloud(vertices=points_xyz.astype(np.float32), colors=rgba)

    buf = io.BytesIO()
    cloud.export(buf, file_type="glb")
    return buf.getvalue()
