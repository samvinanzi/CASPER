# ---- Stage 1: Download Webots ----
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 AS downloader

ARG WEBOTS_VERSION=R2025a
#ARG WEBOTS_VERSION=R2021a
ARG WEBOTS_PACKAGE_PREFIX=

ENV DEBIAN_FRONTEND=noninteractive

# Install required tools
RUN apt-get update && apt-get install -y wget bzip2 && rm -rf /var/lib/apt/lists/*

# Download and extract Webots
WORKDIR /webots
RUN wget https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2 && \
    tar xjf webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2 && \
    rm webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2

# ---- Stage 2: Final Image with Webots + Python ----
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and minimal GUI libs
RUN apt update -y && apt upgrade -y && \
    apt install -y --no-install-recommends \
    curl \
    wget \
    python3-venv \
    python3-pip \
    locales \
    git \
    xvfb \
    libx11-6 \
    x11-apps mesa-utils libgl1 libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb-icccm4 \
    libxrandr2 \
    libxcomposite1 \
    libxcursor1 \
    libxi6 \
    libxtst6 \
    libxss1 \
    libxdamage1 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'


# Create virtual environment 
RUN python3 -m venv /venv 
# Set PATH to use the virtual environment by default 
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements.txt and install Python packages
COPY requirements.txt .
RUN pip3 install  -r requirements.txt
RUN rm requirements.txt


COPY external/strands_qsr_lib /opt/strands_qsr_lib

# Set PYTHONPATH
ENV PYTHONPATH="/opt/strands_qsr_lib/qsr_lib/src:/opt/strands_qsr_lib/qsr_lib/src/qsrlib_qsrs:$PYTHONPATH"

# Install Webots runtime dependencies
RUN wget https://raw.githubusercontent.com/cyberbotics/webots/master/scripts/install/linux_runtime_dependencies.sh && \
    chmod +x linux_runtime_dependencies.sh && ./linux_runtime_dependencies.sh && rm ./linux_runtime_dependencies.sh

# Copy Webots installation
COPY --from=downloader /webots/webots* /usr/local/webots

# Set environment variables for Webots
ENV QTWEBENGINE_DISABLE_SANDBOX=1
ENV WEBOTS_HOME=/usr/local/webots
ENV PATH="${WEBOTS_HOME}:$PATH"

# Enable OpenGL capabilities
ENV NVIDIA_DRIVER_CAPABILITIES graphics,compute,utility

ENV USER=root


# Webots GUI will use the host display
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Set NVIDIA offload environment variables
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

WORKDIR /app

# Copy everything from the current host directory into /app inside the container
#COPY . .

# === Optional Headless Mode ===
# To use Webots in headless mode (no GUI), comment the line above and uncomment below:
# CMD ["xvfb-run", "--auto-servernum", "--", "webots", "--batch", "--no-rendering"]

# Default: open shell with GUI support
#CMD ["/bin/bash"]
#CMD ["webots"]
# ====================================================================================
# === Build ===
#sudo docker build -t webots-gui .

# === Without GPU Acceleration ===
#To run Webots with a graphical user interface in a docker container, you need to enable connections to the X server before starting the docker container:
#xhost +local:root 

# Start the container:
#docker run --gpus=all -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw  webots-gui


#xhost +local:root
#docker run -it \
#  --gpus all \
#  -e DISPLAY=$DISPLAY \
#  -e QT_X11_NO_MITSHM=1 \
#  -e __NV_PRIME_RENDER_OFFLOAD=1 \
#  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
#  -v /tmp/.X11-unix:/tmp/.X11-unix \
#  -v /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia:ro \
#  --device /dev/dri \
#  webots-gpu

# Start bash by default
ENTRYPOINT ["/bin/bash"]
