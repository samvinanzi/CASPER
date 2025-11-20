#!/bin/bash

# ===========================
# Launch CASPER (or container) with NVIDIA GPU Offload
# ===========================

IMAGE_NAME="casper-webots"
CONTAINER_NAME="casper-container"
SCENE_PATH="/app/CASPER/webots/worlds/kitchen2_Transformed.wbt"

DOCKERFILE_PATH="Dockerfile"
CHECKSUM_FILE=".dockerfile_checksum"

# ---------------------------
# Allow Docker (root) to access X server
# ---------------------------
xhost +local:root


# ---------------------------
# Check Dockerfile changes and rebuild image if needed
# ---------------------------
CURRENT_HASH=$(sha256sum $DOCKERFILE_PATH | awk '{print $1}')
OLD_HASH=""
if [[ -f $CHECKSUM_FILE ]]; then
    OLD_HASH=$(cat $CHECKSUM_FILE)
fi

if [[ "$CURRENT_HASH" != "$OLD_HASH" ]] || [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "🔨 Dockerfile changed or image missing. Building $IMAGE_NAME..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
    echo "$CURRENT_HASH" > $CHECKSUM_FILE
else
    echo "✅ Dockerfile unchanged and image exists. Skipping build."
fi

# ---------------------------
# Stop and remove any existing CASPER container (if still running)
# ---------------------------
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  echo "🛑 Stopping existing $CONTAINER_NAME..."
  docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
  docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

# ---------------------------
# Enable NVIDIA GPU offload for Webots (in Docker or native)
# ---------------------------
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0

# ---------------------------
# Optional for better GL performance
# ---------------------------
export LIBGL_ALWAYS_INDIRECT=0

# If you run Webots directly on host
# webots

# ---------------------------
# Launch CASPER container (add --rm to auto-deletes on exit):
# ---------------------------
docker run -it \
  --gpus all \
  --env DISPLAY=$DISPLAY \
  --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  --env __NV_PRIME_RENDER_OFFLOAD=1 \
  --env __GLX_VENDOR_LIBRARY_NAME=nvidia \
  --env __VK_LAYER_NV_optimus=NVIDIA_only \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --env QT_X11_NO_MITSHM=1 \
  --mount type=bind,source=/home/e10738lb/Development/CASPER,target=/app/CASPER,readonly=false,bind-propagation=rslave \
  --mount type=volume,source=casper_sqlite,target=/app/CASPER/data/sqlite3 \
  -w /app/CASPER \
  --name $CONTAINER_NAME \
  $IMAGE_NAME webots $SCENE_PATH

# Revoke X access for security after exit
xhost -local:root

# ---------------------------
# Copy Data from Docker volume to Host:
# docker cp casper-container:app/CASPER/data/sqlite3/kitchen.sqlite3 ./data/sqlite3
# ---------------------------