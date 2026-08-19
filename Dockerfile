# Build/verify environment for KernelForge.
#
# This is the same recipe the build-cuda CI job runs and passes: a CUDA devel
# image has nvcc but no GPU, so `docker build` proves the kernels compile and the
# shared library loads through ctypes. It does NOT run them - the GPU tests skip
# inside this image exactly as they do in CI.
#
#   docker build -t kernelforge .                 # compile + CPU-side tests
#   docker run --rm --gpus all kernelforge \
#       sh -c "python3 -m pytest && python3 bench/run.py"
#
# The second form needs a real GPU and the NVIDIA container runtime. The measured
# numbers in bench/RESULTS.md come from a Colab/Kaggle T4, not from a container.
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# ARCH must match the GPU this image will run on: sm_75 = T4, sm_80 = A100,
# sm_89 = L4. A mismatch still builds and still loads, then fails at the first
# kernel launch with CUDA error 209.
ARG ARCH=sm_75
RUN make ARCH=${ARCH}

# numpy and pytest only; nothing here may pull in a torch that would replace the
# runtime's own build.
RUN python3 -m pip install --no-cache-dir numpy pytest

CMD ["python3", "-m", "pytest"]
