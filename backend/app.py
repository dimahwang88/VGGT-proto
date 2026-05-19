"""FastAPI service wrapping VGGT.

One GPU -> one model -> inference is serialized behind an asyncio lock and
run in a worker thread so the event loop stays responsive. Scenes live in
memory with a TTL and a hard cap (each retains GPU tensors for tracking).
"""

from __future__ import annotations

import asyncio
import io
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .convert import build_glb
from .schemas import ReconstructResponse, TrackRequest, TrackResponse
from .vggt_runner import Scene, reconstruct, track

MAX_IMAGES = 50          # UI/VRAM guard (24 GB RTX 4090, 518px frames)
MIN_IMAGES = 1
SESSION_TTL = 30 * 60    # seconds
MAX_SESSIONS = 4         # bound retained GPU memory

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="VGGT 3D Reconstruction")
_gpu_lock = asyncio.Lock()


@dataclass
class _Entry:
    scene: Scene
    glb: bytes
    created: float


_sessions: dict[str, _Entry] = {}


def _evict() -> None:
    now = time.time()
    stale = [k for k, e in _sessions.items() if now - e.created > SESSION_TTL]
    while len(_sessions) - len(stale) >= MAX_SESSIONS:
        oldest = min(
            (k for k in _sessions if k not in stale),
            key=lambda k: _sessions[k].created,
        )
        stale.append(oldest)
    for k in stale:
        _sessions.pop(k).scene.release()


def _get(session_id: str) -> _Entry:
    entry = _sessions.get(session_id)
    if entry is None:
        raise HTTPException(404, "Unknown or expired session")
    return entry


@app.post("/api/reconstruct", response_model=ReconstructResponse)
async def api_reconstruct(images: list[UploadFile] = File(...)) -> ReconstructResponse:
    if not (MIN_IMAGES <= len(images) <= MAX_IMAGES):
        raise HTTPException(
            400, f"Provide between {MIN_IMAGES} and {MAX_IMAGES} images"
        )

    with tempfile.TemporaryDirectory() as tmp:
        paths: list[str] = []
        for idx, up in enumerate(images):
            data = await up.read()
            # Bake EXIF orientation into pixels — VGGT's PIL loader ignores
            # the EXIF rotate tag, so phone portraits would otherwise be fed
            # (and reconstructed) sideways.
            p = Path(tmp) / f"{idx:04d}.png"
            try:
                img = ImageOps.exif_transpose(
                    Image.open(io.BytesIO(data))
                ).convert("RGB")
                img.save(p, format="PNG")
            except Exception:
                p = Path(tmp) / f"{idx:04d}_{Path(up.filename or 'img').name}"
                p.write_bytes(data)
            paths.append(str(p))

        async with _gpu_lock:
            try:
                scene = await asyncio.to_thread(reconstruct, paths)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise HTTPException(
                        507, "GPU out of memory — use fewer images"
                    ) from exc
                raise HTTPException(500, f"Reconstruction failed: {exc}") from exc

    glb = build_glb(scene.points_xyz, scene.points_rgb)

    sid = uuid.uuid4().hex
    _evict()
    _sessions[sid] = _Entry(scene=scene, glb=glb, created=time.time())

    h, w = scene.frame_size
    return ReconstructResponse(
        session_id=sid,
        num_images=scene.num_images,
        frame_height=h,
        frame_width=w,
        cameras=scene.cameras,
        pointcloud_url=f"/api/asset/{sid}.glb",
        frame_urls=[f"/api/asset/{sid}/frame/{i}.png" for i in range(scene.num_images)],
    )


@app.get("/api/asset/{session_id}.glb")
async def api_glb(session_id: str) -> Response:
    return Response(_get(session_id).glb, media_type="model/gltf-binary")


@app.get("/api/asset/{session_id}/frame/{idx}.png")
async def api_frame(session_id: str, idx: int) -> Response:
    scene = _get(session_id).scene
    if not (0 <= idx < scene.num_images):
        raise HTTPException(404, "Frame out of range")
    return Response(scene.frames_png[idx], media_type="image/png")


@app.post("/api/track", response_model=TrackResponse)
async def api_track(req: TrackRequest) -> TrackResponse:
    scene = _get(req.session_id).scene
    if not (0 <= req.ref_image_id < scene.num_images):
        raise HTTPException(400, "ref_image_id out of range")

    async with _gpu_lock:
        try:
            tracks = await asyncio.to_thread(
                track, scene, req.ref_image_id, req.query_points
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise HTTPException(507, "GPU out of memory") from exc
            raise HTTPException(500, f"Tracking failed: {exc}") from exc
    return TrackResponse(tracks=tracks)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "sessions": len(_sessions)}


# Serve the static UI at the root. Mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
