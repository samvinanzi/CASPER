# CASPER + Webots Simulation Environment (Host & Docker)

This project allows you to run the full CASPER + Webots simulation environment **either directly on your host machine** or **inside Docker with GPU acceleration**, ensuring both setups behave identically.

Both workflows provide the same:
- Webots installation
- Python virtual environment
- QSR library setup
- PYTHONPATH configuration
- Controller compatibility
- Shared `data/` folder read/write access

---

## 🐳 Running via Docker (GPU Enabled)
The script `launch_webots_gpu.sh` builds and manages the Docker environment.

### What the script does
```
- Creates or updates the Docker image (only rebuilds when Dockerfile changes)
- Starts the container
- Launches Webots with the correct simulation world
- Ensures GPU acceleration is enabled
- Mounts the project data folder with proper permissions
```

### Usage
```bash
chmod +x launch_webots_gpu.sh
./launch_webots_gpu.sh
```

### Copying data from the Docker container to the host
Example: copying the SQLite database
```bash
docker cp casper-container:app/CASPER/data/sqlite3/kitchen.sqlite3 ./data/sqlite3
```

---

## 🖥️ Running Natively on Host
Install a fully Docker‑equivalent environment on your local machine using:

### Script: `install_local.sh`

### What the script does
```
- Installs Webots into /usr/local/webots
- Creates a Python venv at ./venv
- Installs requirements.txt
- Copies the custom strands_qsr_lib
- Installs all Webots runtime dependencies
- Adds Docker‑like environment variables permanently into ~/.bashrc
- Configures PYTHONPATH so controllers work immediately
- Fixes all permissions in ./data so the venv can read/write without errors
```

### Usage
```bash
chmod +x install_local.sh
./install_local.sh

# Load Webots environment variables
source ~/.bashrc

# Activate the local venv
source ./venv/bin/activate

# Run Webots
webots

# Leave the venv
deactivate
```

---

## 🤖 Running Controllers (Robots & Humans)
Controllers can be executed in **two ways**:

### 1. **Inside Webots**
You can assign a controller directly in the Webots GUI:
- Select the robot
- Set its controller to any available controller script
- Webots will run it internally

### 2. **Extern Mode**
You can set a Webots robot controller to `<extern>`, allowing the agent to be run from VSCode.

The project contains a prepared `.vscode/` folder with tasks that let you:
- Run **robot agents** externally
- Run **human agents** externally
- Run **both** simultaneously
- Automatically connect to Webots via the extern controller interface
- Start each agent in its own VSCode terminal

This allows complete debugging and logging directly from VSCode.

---

## ✅ Summary
You can run the simulation in two interchangeable modes:
- **Docker GPU mode** → fully contained, replicable, guaranteed identical environment
- **Host local mode** → identical setup to Docker but without containers

Both modes support:
- Webots GUI execution
- Extern controllers via VSCode tasks
- Human and robot agents
- Shared data and full write access

---

## 📚 Citation
If you use this code or build upon the CASPER architecture, please cite:

Vinanzi, S., Cangelosi, A. **CASPER: Cognitive Architecture for Social Perception and Engagement in Robots.** *International Journal of Social Robotics* 17, 1979–1997 (2025). https://doi.org/10.1007/s12369-024-01116-2

