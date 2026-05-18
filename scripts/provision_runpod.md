# Provision on RunPod (recommended, on-demand)

VGGT needs a CUDA GPU with ≥24 GB VRAM ideally. RTX 4090 (24 GB) is the
cheapest capable option and is billed per second — far cheaper than a
fixed-cost Hetzner dedicated GPU for sporadic use.

## 1. One-time

1. RunPod account + pay-as-you-go credit.
2. **Storage → Network Volumes → New.** ~25 GB, in a datacenter/region that
   lists RTX 4090 availability. Remember the region.
3. (Optional) Build & push the image:
   ```bash
   docker build -t <registry>/vggt-proto .
   docker push <registry>/vggt-proto
   ```

## 2. Deploy a pod

- **Pods → Deploy → GPU: RTX 4090**, On-Demand (or Spot for short cheap
  sessions; Spot can be reclaimed).
- Container image: `<registry>/vggt-proto`, or stock
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- **Attach the Network Volume** → mounts at `/workspace` (must be the volume's
  region).
- Environment: `HF_HOME=/workspace/hf`.
- **Expose HTTP port `8000`.** RunPod gives a proxy URL
  `https://<pod-id>-8000.proxy.runpod.net`.

## 3. Run

- Prebuilt image: nothing to do — the container `CMD` starts uvicorn. The
  first reconstruction triggers the one-time weight download to the volume.
- Stock template: follow the "Model setup on the pod" steps in the project
  `README.md` (clone, install deps with `--no-deps` for vggt, pre-download
  weights, `uvicorn backend.app:app --host 0.0.0.0 --port 8000`).

Open the proxy URL in a browser.

## 4. Teardown (stop billing)

- **Stop** the pod: GPU billing stops; container disk kept (small fee). Fast
  to resume.
- **Terminate** the pod: removes the pod entirely. The **Network Volume
  persists** (with the cached weights), so the next pod skips the download.
- Delete the Network Volume only if you no longer need the cached weights.

## Notes

- ~1 hour of RTX 4090 ≈ $0.34–0.69 total.
- Need hundreds of images per scene? Use a 48 GB GPU (L40S / A40) instead.
- Hands-off scale-to-zero: package `backend/` as a RunPod **Serverless**
  handler (trade ~tens-of-seconds cold start for zero idle cost). Not wired
  up here — the long-lived FastAPI app is the default.
