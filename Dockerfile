# ---- Stage 1: Download Webots ----
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 AS downloader

ARG WEBOTS_VERSION=R2025a
ARG WEBOTS_PACKAGE_PREFIX=

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal tools to download and extract Webots
RUN apt-get update && apt-get install -y wget bzip2 && rm -rf /var/lib/apt/lists/*

WORKDIR /webots
RUN wget https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2 && \
    tar xjf webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2 && \
    rm webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2

# ---- Stage 2: Final Image with Webots + Python ----
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies for Webots and Python
RUN apt-get update && apt upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    wget \
    locales \
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

# Create Python virtual environment
RUN python3 -m venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python packages from requirements.txt
COPY requirements.txt .
RUN pip3 install -r requirements.txt && rm requirements.txt

# Copy any custom Python libraries
COPY external/strands_qsr_lib /opt/strands_qsr_lib
ENV PYTHONPATH="/opt/strands_qsr_lib/qsr_lib/src:/opt/strands_qsr_lib/qsr_lib/src/qsrlib_qsrs:$PYTHONPATH"

# Install Webots runtime dependencies
RUN wget https://raw.githubusercontent.com/cyberbotics/webots/master/scripts/install/linux_runtime_dependencies.sh && \
    chmod +x linux_runtime_dependencies.sh && ./linux_runtime_dependencies.sh && rm ./linux_runtime_dependencies.sh

# Copy Webots installation from downloader stage
COPY --from=downloader /webots/webots* /usr/local/webots

# Set environment variables for Webots
ENV WEBOTS_HOME=/usr/local/webots
ENV PATH="${WEBOTS_HOME}:$PATH"
ENV PYTHONPATH=${PYTHONPATH}:${WEBOTS_HOME}/lib/controller/python

# Enable OpenGL capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# Use host display for Webots GUI
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Set working directory
WORKDIR /app

# Start bash by default
CMD ["/bin/bash"]
