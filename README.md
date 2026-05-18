# VGGT 3D Reconstruction Web App

A minimal web app around **VGGT** (Wang et al., *Visual Geometry Grounded
Transformer*, CVPR 2025, arXiv 2503.11651). Upload multiple images, run one
feed-forward pass, and get back:

- a **3D point cloud** (interactive Three.js viewer),
- per-image **camera intrinsics + extrinsics**,
- **point tracking** from user-clicked query points across all views.

This wraps the official `facebookresearch/vggt` model + the non-commercial
`facebook/VGGT-1B` checkpoint behind a FastAPI service. It does **not**
re-implement or re-train VGGT.

## Architecture

```
web/            static UI (vanilla JS + Three.js, no build step)
backend/
  app.py        FastAPI: routes, single-GPU lock, in-memory session store
  vggt_runner.py lazy VGGT singleton; reconstruct() and track()
  convert.py    predictions -> GLB point cloud + cameras JSON
  schemas.py    pydantic request/response models
Dockerfile      RunPod/CUDA base image
scripts/        provisioning notes (RunPod default, Hetzner alternative)
```

Flow: `POST /api/reconstruct` runs the aggregator + camera/depth heads once,
caches the backbone tokens in an in-memory session, and returns cameras +
a GLB URL + preprocessed frame URLs. `POST /api/track` reuses the cached
backbone (when the reference frame is the first image) to run the tracking
head on clicked query points.

## Deploy on RunPod (recommended)

A local GPU is **not** required (and not used). Everything runs on a RunPod
RTX 4090 pod; your machine only edits code and opens the pod's proxy URL.

### One-time setup

1. Create a RunPod account and add pay-as-you-go credit.
2. Create a **Network Volume** (~25 GB) in a region that has RTX 4090 stock;
   note the region (pods must match it to attach the volume). This persists
   the model cache across pods.
3. (Optional) Build and push the image so sessions reduce to one command:
   `docker build -t <registry>/vggt-proto . && docker push <registry>/vggt-proto`.

### Each session (on-demand loop)

1. Deploy a Pod → GPU **RTX 4090 (24 GB)**, On-Demand, image
   `<registry>/vggt-proto` (or the stock
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` template).
2. Attach the Network Volume (mounts at `/workspace`).
3. Set env `HF_HOME=/workspace/hf`.
4. Expose **HTTP port 8000** → app is at
   `https://<pod-id>-8000.proxy.runpod.net`.
5. Start: `uvicorn backend.app:app --host 0.0.0.0 --port 8000`.
6. Open the proxy URL → use the UI.
7. **Stop or Terminate the pod when done** to halt GPU billing. The Network
   Volume (with weights) persists either way.

### Model setup on the pod (stock template, no prebuilt image)

Run in the pod's web terminal / SSH:

```bash
# 0. Sanity: GPU visible, torch sees CUDA
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1. Persist caches on the Network Volume
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

# 2. Get the project code onto the volume
cd /workspace
git clone <this-repo-url> vggt-proto      # or rsync your local working copy
cd vggt-proto

# 3. Install deps WITHOUT clobbering the image's CUDA torch
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir --no-deps "git+https://github.com/facebookresearch/vggt.git"
python -c "import vggt, torchvision; print('vggt + tv OK', torchvision.__version__)"

# 4. Pre-download the non-commercial weights to the volume (one time, ~5 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/VGGT-1B')"

# 5. Smoke-test the model loads on GPU
python -c "import torch; from vggt.models.vggt import VGGT; \
m=VGGT.from_pretrained('facebook/VGGT-1B').to('cuda').eval(); \
print('model loaded', sum(p.numel() for p in m.parameters())/1e9, 'B params')"

# 6. Launch the app (bind 0.0.0.0 so the RunPod proxy can reach it)
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

On a **fresh pod the weights are already on the volume** — skip steps 0 & 4,
re-run 1–3, then 6. Baking the Docker image collapses every session to step 6.

See `scripts/provision_runpod.md` for more detail and
`scripts/provision_hetzner.md` for the fixed-cost Hetzner alternative.

## Limitations

- **VRAM scales with #images × tokens** (global attention is quadratic in
  total tokens). On a 24 GB RTX 4090 the practical limit is a few tens of
  518 px frames; the API caps input at `MAX_IMAGES = 50` and returns
  HTTP 507 on GPU OOM.
- **No metric scale / fixed coordinate frame** — outputs are relative,
  anchored to the first input image.
- **Static-scene assumption** — moving content and very large viewpoint
  changes degrade depth, point maps, and tracks.
- **Resolution** — inputs are preprocessed to ~518 px; fine detail is lost.
  The UI displays/clicks on these preprocessed frames so tracking coordinates
  line up exactly.
- **License** — uses the **non-commercial** `facebook/VGGT-1B` checkpoint.
  Commercial use needs the separate `VGGT-1B-Commercial` checkpoint and an
  application (out of scope here).
- **Single GPU** — requests are serialized; this is a prototype, not a
  multi-tenant service. Sessions are in-memory (`MAX_SESSIONS`, TTL).
- **Cold start** — first run downloads ~5 GB of weights unless pre-cached on
  the Network Volume.

## Local API shape

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/reconstruct` | multipart `images` → cameras + point cloud + frames |
| GET  | `/api/asset/{sid}.glb` | point cloud (binary glTF) |
| GET  | `/api/asset/{sid}/frame/{i}.png` | preprocessed frame `i` |
| POST | `/api/track` | `{session_id, ref_image_id, query_points}` → tracks |
| GET  | `/healthz` | liveness + session count |
