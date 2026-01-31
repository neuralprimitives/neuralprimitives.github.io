#!/bin/bash
set -e

# =========================
# Config
# =========================
PROJECT_NAME=compod
PYTHON_VERSION=3.10.10

# =========================
# Init conda
# =========================
source "$(conda info --base)/etc/profile.d/conda.sh"

# =========================
# Create env if needed
# =========================
if ! conda info --envs | awk '{print $1}' | grep -qx "${PROJECT_NAME}"; then
    echo "[INFO] Creating conda environment: ${PROJECT_NAME}"
    conda create -y \
        -n "${PROJECT_NAME}" \
        python="${PYTHON_VERSION}"
else
    echo "[INFO] Conda environment ${PROJECT_NAME} already exists"
fi

# =========================
# Activate env
# =========================
conda activate "${PROJECT_NAME}"

# =========================
# Conda dependencies
# =========================
echo "[INFO] Installing conda dependencies..."

conda install -y \
    -c conda-forge \
    -c pytorch \
    -c nvidia \
    gsl \
    libgomp \
    scipy \
    shapely \
    tqdm \
    trimesh \
    treelib \
    colorlog \
    pytorch \
    torchvision \
    torchaudio \
    pytorch-cuda=11.8

# =========================
# Pip dependencies
# =========================
echo "[INFO] Installing pip dependencies..."

pip install --upgrade pip

pip install \
    open3d \
    gco-wrapper

# =========================
# Install project (editable if desired)
# =========================
echo "[INFO] Installing project package..."

pip install .

# pip install -e .   # ← uncomment if you want editable mode

echo "[SUCCESS] Environment ${PROJECT_NAME} is ready."
