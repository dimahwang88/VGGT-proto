# Matches the RunPod GPU pod base (torch 2.4.0 / CUDA 12.4 / py3.11).
# `-devel` keeps nvcc + gcc so any source-built optional dep compiles.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Cache models on the mounted RunPod network volume so weights survive
# pod stop/start and are downloaded only once.
ENV HF_HOME=/workspace/hf \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# Install app deps, then the official VGGT model code with --no-deps so it
# cannot replace the prebuilt CUDA torch from the base image.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps "git+https://github.com/facebookresearch/vggt.git"

COPY backend ./backend
COPY web ./web

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
