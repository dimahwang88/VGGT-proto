"""Thin wrapper around the official VGGT model.

We deliberately reuse VGGT's own modules/utils and do not re-derive any
geometry. The aggregator tokens are computed once per scene and kept so the
tracking call can reuse the backbone (fast path when the reference frame is
the first image).
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

# Official VGGT code (installed from github.com/facebookresearch/vggt).
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

MODEL_ID = "facebook/VGGT-1B"  # non-commercial checkpoint

# Cap exported point count so the GLB stays small enough for a browser viewer.
MAX_POINTS = 600_000

_model: VGGT | None = None
_model_lock = threading.Lock()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16  # Ada (RTX 4090) etc.
    return torch.float16


def get_model() -> VGGT:
    """Lazy, process-wide singleton. First call downloads weights to HF_HOME."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                m = VGGT.from_pretrained(MODEL_ID).to(_device())
                m.eval()
                _model = m
    return _model


@dataclass
class Scene:
    """Everything a reconstruction produced, plus state reused by tracking."""

    # Serializable outputs
    num_images: int
    cameras: list[dict]            # per-image extrinsic/intrinsic (see convert.py)
    points_xyz: np.ndarray         # (P, 3) float32
    points_rgb: np.ndarray         # (P, 3) uint8
    frames_png: list[bytes]        # preprocessed frames, what the UI displays/clicks on
    depth_png: list[bytes]         # per-frame colorized depth maps (same size as frames)
    frame_size: tuple[int, int]    # (height, width) of the preprocessed frames

    # GPU state retained for the tracking head (fast path: ref frame == 0)
    _tokens: list = field(repr=False, default=None)
    _ps_idx: object = field(repr=False, default=None)
    _images: torch.Tensor = field(repr=False, default=None)  # (1, N, 3, H, W)

    def release(self) -> None:
        self._tokens = None
        self._ps_idx = None
        self._images = None


def _tensor_to_png(img_chw: torch.Tensor) -> bytes:
    """(3, H, W) float in [0,1] -> PNG bytes."""
    arr = (img_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# Inferno-style colormap anchors (perceptual, dependency-free via np.interp).
_CMAP_T = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_CMAP_R = np.array([0, 87, 188, 249, 252])
_CMAP_G = np.array([0, 16, 55, 142, 255])
_CMAP_B = np.array([4, 110, 84, 9, 164])


def _depth_to_png(depth_hw: np.ndarray, conf_hw: np.ndarray) -> bytes:
    """(H, W) depth + (H, W) confidence -> colorized PNG bytes.

    Near = warm, far = dark. Robust 2/98 percentile stretch over the
    confident, finite pixels; invalid pixels render black.
    """
    d = depth_hw.astype(np.float64)
    valid = np.isfinite(d) & (conf_hw > 0)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    t = 1.0 - norm  # invert so closer surfaces are warm/bright

    rgb = np.zeros((*d.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.interp(t, _CMAP_T, _CMAP_R)
    rgb[..., 1] = np.interp(t, _CMAP_T, _CMAP_G)
    rgb[..., 2] = np.interp(t, _CMAP_T, _CMAP_B)
    rgb[~valid] = 0

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def reconstruct(image_paths: list[str]) -> Scene:
    """Run aggregator + camera/depth heads and build a colored point cloud."""
    from .convert import cameras_to_json  # local import avoids a cycle

    model = get_model()
    device, dtype = _device(), _dtype()

    images = load_and_preprocess_images(image_paths).to(device)  # (N, 3, H, W)
    n, _, h, w = images.shape
    images_b = images[None]  # (1, N, 3, H, W)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            tokens, ps_idx = model.aggregator(images_b)
            pose_enc = model.camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_b.shape[-2:]
            )
            depth_map, depth_conf = model.depth_head(tokens, images_b, ps_idx)

    # World points by unprojecting depth (the convention used by the official
    # demos — more stable than the raw point head for visualization).
    extr = extrinsic.squeeze(0).float().cpu().numpy()   # (N, 3, 4)
    intr = intrinsic.squeeze(0).float().cpu().numpy()    # (N, 3, 3)
    depth_np = depth_map.squeeze(0).float().cpu().numpy()  # (N, H, W, 1)
    conf_np = depth_conf.squeeze(0).float().cpu().numpy()  # (N, H, W)

    depth_png = [_depth_to_png(depth_np[i, :, :, 0], conf_np[i]) for i in range(n)]

    world = unproject_depth_map_to_point_map(depth_np, extr, intr)  # (N, H, W, 3)
    colors = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    world = world.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    conf = conf_np.reshape(-1)

    # Keep only confident, finite points; drop a confidence floor at the
    # median so flat/sky regions don't dominate.
    finite = np.isfinite(world).all(axis=1)
    thresh = max(1e-3, float(np.quantile(conf[finite], 0.5))) if finite.any() else 0.0
    keep = finite & (conf >= thresh)
    pts = world[keep]
    rgb = colors[keep]

    if pts.shape[0] > MAX_POINTS:
        sel = np.random.choice(pts.shape[0], MAX_POINTS, replace=False)
        pts, rgb = pts[sel], rgb[sel]

    frames_png = [_tensor_to_png(images[i]) for i in range(n)]

    return Scene(
        num_images=n,
        cameras=cameras_to_json(extr, intr),
        points_xyz=pts.astype(np.float32),
        points_rgb=rgb.astype(np.uint8),
        frames_png=frames_png,
        depth_png=depth_png,
        frame_size=(h, w),
        _tokens=tokens,
        _ps_idx=ps_idx,
        _images=images_b,
    )


def track(scene: Scene, ref_idx: int, query_xy: list[list[float]]) -> list[list[list[float]]]:
    """Track ``query_xy`` (pixel coords on frame ``ref_idx``) across all frames.

    Returns ``tracks[q] = [[x, y, visible], ... one per frame]``.

    VGGT's tracking head expects query points on frame 0. When the reference
    frame is the first image we reuse the cached backbone; otherwise we
    recompute the aggregator on a frame-reordered batch and map the result
    back to the original order.
    """
    model = get_model()
    device, dtype = _device(), _dtype()
    n = scene.num_images

    q = torch.tensor(query_xy, dtype=torch.float32, device=device)  # (Q, 2)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            if ref_idx == 0 and scene._tokens is not None:
                tokens, ps_idx, images_b = scene._tokens, scene._ps_idx, scene._images
                order = list(range(n))
            else:
                order = [ref_idx] + [i for i in range(n) if i != ref_idx]
                images_b = scene._images[:, order, ...]
                tokens, ps_idx = model.aggregator(images_b)

            track_list, vis_score, _ = model.track_head(
                tokens, images_b, ps_idx, query_points=q[None]
            )

    tracks = track_list[-1].squeeze(0).float().cpu().numpy()   # (N, Q, 2)
    vis = vis_score.squeeze(0).float().cpu().numpy()           # (N, Q)

    # Undo the frame permutation so frame index == original upload order.
    inv = [0] * n
    for new_i, orig_i in enumerate(order):
        inv[orig_i] = new_i

    n_q = tracks.shape[1]
    out: list[list[list[float]]] = []
    for qi in range(n_q):
        per_frame = []
        for orig_i in range(n):
            ni = inv[orig_i]
            x, y = tracks[ni, qi]
            visible = bool(vis[ni, qi] >= 0.5)
            per_frame.append([float(x), float(y), float(visible)])
        out.append(per_frame)
    return out
