# Provision on Hetzner (fixed-cost alternative)

Use this only if you want an always-on box on Hetzner specifically. Hetzner
**Cloud has no GPU instances** — GPUs are only on the **dedicated** line, each
order has a one-time €79 setup fee, and provisioning is not instant. For
true on-demand use, prefer RunPod (`provision_runpod.md`).

## Server

- Order a **GEX44** (NVIDIA RTX 4000 SFF Ada, **20 GB**, 64 GB RAM,
  ~€184/mo + €79 setup) in Falkenstein. GEX130 (RTX 6000 Ada, 48 GB) only if
  you need hundreds of images per scene.
- 20 GB < the recommended 24 GB: keep `MAX_IMAGES` conservative (lower it in
  `backend/app.py` if you hit OOM).

## Setup

```bash
# NVIDIA driver + Docker + NVIDIA Container Toolkit, then:
git clone <this-repo-url> vggt-proto && cd vggt-proto
docker build -t vggt-proto .

# Persist the HF cache on the host so re-creates don't re-download weights.
mkdir -p /opt/vggt-hf
docker run -d --restart unless-stopped --gpus all \
  -p 8000:8000 \
  -e HF_HOME=/workspace/hf \
  -v /opt/vggt-hf:/workspace/hf \
  --name vggt vggt-proto
```

Put nginx (or Caddy) in front for TLS if exposing publicly. Optionally manage
the container via a systemd unit instead of `--restart`.

## Teardown

Recurring cost stops only when you **cancel the server** in the Hetzner
robot/console. Stopping the container alone does not stop billing for a
dedicated server.
