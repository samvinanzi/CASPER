#!/usr/bin/env bash
set -e

WEBOTS_VERSION="R2025a"
WEBOTS_PACKAGE_PREFIX=""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/venv"  # venv inside project folder
DATA_FOLDER="$SCRIPT_DIR/data"

echo "=== Updating system ==="
sudo apt-get update
sudo apt-get upgrade -y
export DEBIAN_FRONTEND=noninteractive

echo "=== Installing system dependencies ==="
sudo apt-get install -y --no-install-recommends \
    openjdk-17-jre \
    python3 python3-venv python3-pip \
    wget bzip2 locales xvfb \
    libx11-6 x11-apps mesa-utils libgl1 libglu1-mesa \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libxcb-icccm4 libxrandr2 libxcomposite1 \
    libxcursor1 libxi6 libxtst6 libxss1 libxdamage1 \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libgtk-3-0 ca-certificates

echo "=== Setting locales ==="
sudo locale-gen en_US.UTF-8
export LANG='en_US.UTF-8'
export LANGUAGE='en_US:en'
export LC_ALL='en_US.UTF-8'

echo "=== Creating Python virtual environment in $VENV_PATH ==="
python3 -m venv "$VENV_PATH"

echo "=== Installing Python packages from requirements.txt ==="
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "ERROR: Missing requirements.txt!"
    exit 1
fi
source "$VENV_PATH/bin/activate"
pip install -r "$SCRIPT_DIR/requirements.txt"
deactivate

echo "=== Installing custom Python library ==="
sudo mkdir -p /opt/strands_qsr_lib
sudo cp -r "$SCRIPT_DIR/external/strands_qsr_lib"/* /opt/strands_qsr_lib/

echo "=== Configuring venv for Docker-like PYTHONPATH ==="
ACTIVATE_HOOK="$VENV_PATH/bin/activate"
cat <<EOF >> "$ACTIVATE_HOOK"

# --- Docker-like PYTHONPATH for Webots ---
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH="/opt/strands_qsr_lib/qsr_lib/src:/opt/strands_qsr_lib/qsr_lib/src/qsrlib_qsrs:\$WEBOTS_HOME/lib/controller/python:\$PYTHONPATH"
export PATH="\$WEBOTS_HOME:\$PATH"
EOF

echo "=== Ensuring venv has full access to project data folder ==="
if [ -d "$DATA_FOLDER" ]; then
    sudo chown -R $(whoami):$(whoami) "$DATA_FOLDER"
    chmod -R u+rwX "$DATA_FOLDER"
    echo "vENV now has full access to $DATA_FOLDER"
else
    echo "Data folder $DATA_FOLDER does not exist. Skipping permissions fix."
fi

echo "=== Installing Webots runtime dependencies ==="
wget https://raw.githubusercontent.com/cyberbotics/webots/master/scripts/install/linux_runtime_dependencies.sh
chmod +x linux_runtime_dependencies.sh
sudo ./linux_runtime_dependencies.sh
rm linux_runtime_dependencies.sh

echo "=== Downloading Webots ${WEBOTS_VERSION} ==="
mkdir -p "$SCRIPT_DIR/webots_download"
cd "$SCRIPT_DIR/webots_download"
wget https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2

echo "=== Extracting Webots ==="
tar xjf webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2
rm webots-${WEBOTS_VERSION}-x86-64${WEBOTS_PACKAGE_PREFIX}.tar.bz2

echo "=== Installing Webots to /usr/local/webots ==="
sudo rm -rf /usr/local/webots
sudo mv webots* /usr/local/webots

echo "=== Cleaning up Webots download folder ==="
rm -rf "$SCRIPT_DIR/webots_download"

echo "=== Adding Docker-like environment variables to ~/.bashrc ==="
BASHRC_UPDATE=false

# WEBOTS_HOME
if ! grep -q "WEBOTS_HOME=/usr/local/webots" ~/.bashrc; then
    echo 'export WEBOTS_HOME=/usr/local/webots' >> ~/.bashrc
    BASHRC_UPDATE=true
fi

# PATH
if ! grep -q 'PATH="\$WEBOTS_HOME:$PATH"' ~/.bashrc; then
    echo 'export PATH="$WEBOTS_HOME:$PATH"' >> ~/.bashrc
    BASHRC_UPDATE=true
fi

# Webots Python controller path
if ! grep -q 'PYTHONPATH="\$PYTHONPATH:$WEBOTS_HOME/lib/controller/python"' ~/.bashrc; then
    echo 'export PYTHONPATH="$PYTHONPATH:$WEBOTS_HOME/lib/controller/python"' >> ~/.bashrc
    BASHRC_UPDATE=true
fi

# Custom library (strands_qsr_lib)
if ! grep -q '/opt/strands_qsr_lib/qsr_lib/src' ~/.bashrc; then
    echo 'export PYTHONPATH="/opt/strands_qsr_lib/qsr_lib/src:/opt/strands_qsr_lib/qsr_lib/src/qsrlib_qsrs:$PYTHONPATH"' >> ~/.bashrc
    BASHRC_UPDATE=true
fi

if [ "$BASHRC_UPDATE" = true ]; then
    echo "Docker-like Webots environment variables added to ~/.bashrc"
fi

echo "=== Installation complete ==="
echo "Virtual environment created at: $VENV_PATH"
echo "Activate it manually with: source \"$VENV_PATH/bin/activate\""
echo "Webots installed in /usr/local/webots"
echo "Run 'source ~/.bashrc' or open a new terminal to use Webots and Python controllers"
