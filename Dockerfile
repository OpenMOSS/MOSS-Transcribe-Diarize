# syntax=docker/dockerfile:1.7
#
# Targets:
#   runtime — vLLM only, model from HF cache at runtime (default)
#   serve   — runtime + model baked in
#
# Runtime (default):
#   docker build -t moss-td-vllm:dev .
#
# Serve (reuse local HF cache, no re-download):
#   docker build --target serve \
#     --build-context models=$HOME/.cache/huggingface \
#     --build-arg MODEL_REVISION=e5118b411bf5a77d7a90c4941066bec93c967312 \
#     -t moss-td-serve:1.0.0 .

ARG CUDA_IMAGE=nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    python3.12 \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"
# README: CUDA 13 → cu130 pinned nightly
ARG VLLM_CUDA=cu130
ARG VLLM_WHEEL_COMMIT=68b4a1d582818e67adc903bf1b8fc5a5447da2fa

# Install the CUDA PyTorch wheels explicitly so uv does not resolve to torch+xpu.
# Pin to the 2.11 release family to match the CUDA 13 wheels used by the project docs.
RUN uv pip install --system --break-system-packages -U \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --python /usr/bin/python3.12 \
    --index-url "https://download.pytorch.org/whl/${VLLM_CUDA}"

RUN uv pip install --system --break-system-packages -U "vllm[audio]" \
    --python /usr/bin/python3.12 \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_WHEEL_COMMIT}/${VLLM_CUDA}"

WORKDIR /workspace
EXPOSE 8000
CMD ["vllm", "serve", "OpenMOSS-Team/MOSS-Transcribe-Diarize", "--trust-remote-code", "--host", "0.0.0.0", "--port", "8000"]

# Model baked in; reuses runtime layer cache when only MODEL_REVISION changes.
FROM runtime AS serve

ARG MODEL_REPO=models--OpenMOSS-Team--MOSS-Transcribe-Diarize
ARG MODEL_REVISION=e5118b411bf5a77d7a90c4941066bec93c967312

# HF snapshots are symlinks into hub/blobs; cp -aL dereferences them into real files.
RUN --mount=type=bind,from=models,source=/,target=/hf,ro \
    mkdir -p /models/MOSS-Transcribe-Diarize && \
    cp -aL "/hf/hub/${MODEL_REPO}/snapshots/${MODEL_REVISION}/." /models/MOSS-Transcribe-Diarize/

CMD ["vllm", "serve", "/models/MOSS-Transcribe-Diarize", "--served-model-name", "OpenMOSS-Team/MOSS-Transcribe-Diarize", "--trust-remote-code", "--host", "0.0.0.0", "--port", "8000"]
