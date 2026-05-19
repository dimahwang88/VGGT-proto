"""Request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Camera(BaseModel):
    image_id: int
    intrinsic: list[list[float]]      # 3x3
    extrinsic: list[list[float]]      # 4x4 world -> camera
    cam_to_world: list[list[float]]   # 4x4 camera -> world


class ReconstructResponse(BaseModel):
    session_id: str
    num_images: int
    frame_height: int
    frame_width: int
    cameras: list[Camera]
    pointcloud_url: str
    frame_urls: list[str]
    depth_urls: list[str]


class TrackRequest(BaseModel):
    session_id: str
    ref_image_id: int = Field(0, ge=0)
    query_points: list[list[float]] = Field(..., min_length=1)


class TrackResponse(BaseModel):
    # tracks[q] = [[x, y, visible(0/1)], ... one entry per frame]
    tracks: list[list[list[float]]]


class IntrospectResponse(BaseModel):
    layer: int
    num_layers: int
    layer_kind: str                 # heuristic: "frame-wise" / "global"
    grid: list[int]                 # [rows, cols] of the patch-token grid
    frame: int
    num_frames: int
    query_patch: list[int]          # [row, col] of the picked patch
    cross_frame_mix: list[float]    # query-token mean similarity per frame
    tokennorm_png: str              # data URI (inferno heatmap)
    attention_png: str              # data URI (inferno heatmap)
