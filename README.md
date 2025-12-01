# UR5 Robotic Manipulation with Vision-Based RL in Isaac Lab

<div align="center">
  <img src="gifs/i2r_clemson_ur5.png" width="100%">
  <br><br>
  <img src="gifs/i2r_clemson_ur5.gif" width="80%" autoplay loop>
</div>


This project implements vision-based reinforcement learning for the UR5 robotic manipulator in Isaac Lab, enabling precise object manipulation through camera-guided control. Our framework combines state-of-the-art physics simulation with deep reinforcement learning to achieve robust pick-and-place operations in complex environments.

**Key Features:**
- 🎯 **Vision-Based Control**: Direct camera input for object detection and manipulation
- 🚀 **GPU-Accelerated Training**: Leverage Isaac Sim's parallel simulation capabilities
- 📊 **Real-time Monitoring**: Integrated WandB support for experiment tracking
- 🔧 **Modular Architecture**: Easy to extend and customize for different tasks

**Keywords:** UR5, vision-based RL, Isaac Lab, robotic manipulation, pick-and-place

---

## 📋 Table of Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Prerequisites
- Ubuntu 22.04 or Windows
- NVIDIA GPU with CUDA 11.7+
- Python 3.11

### Source installation: Isaac Sim 5.1.0 and Isaac Lab 2.3.0

This repository was developed and tested with Isaac Sim 5.1.0 and Isaac Lab 2.3.0. The instructions below follow the Isaac Lab source-installation workflow and show how to build Isaac Sim from source and install Isaac Lab from source.

Note: building Isaac Sim from source is an advanced workflow and requires Ubuntu 22.04 LTS or higher, Python 3.11 (for Isaac Sim 5.x) and an up-to-date NVIDIA driver/CUDA toolchain. If you prefer pre-built binaries, refer to the Isaac Lab documentation for the binaries installation instead.

**1) Clone and build Isaac Sim (5.1.0)**

```bash
# Example workspace directory where you keep the sources
cd $HOME

# Clone Isaac Sim
git clone https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim


# Build Isaac Sim from source (Linux)
./build.sh

# After a successful build, set the ISAACSIM_PATH environment variable to the built release
export ISAACSIM_PATH="${PWD}/IsaacSim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Quick verification
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
${ISAACSIM_PATH}/isaac-sim.sh --help
```

**2) Clone Isaac Lab (2.3.0) and link to Isaac Sim**

```bash
# Move to your workspace and clone Isaac Lab
cd $HOME
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab


# Create a symbolic link in Isaac Lab pointing to the Isaac Sim built release
# This makes the Isaac Sim modules and extensions discoverable by Isaac Lab
ln -s ${ISAACSIM_PATH} _isaac_sim
```

**3) Create / activate a Python environment for Isaac Lab**

Recommendation: create a dedicated environment (conda or uv). For Isaac Sim 5.x the Python runtime is 3.11 — ensure your virtual env uses the same Python minor version.

```bash
# Using the helper to create a conda environment (default name: env_isaaclab)
./isaaclab.sh -c


# Activate the environment (conda example)
conda activate env_isaaclab

```

**4) Install Isaac Lab extensions and learning frameworks**

```bash
# Install all extensions (default). This installs the learning frameworks (rl_games, rsl_rl, sb3, skrl, robomimic, ...)
./isaaclab.sh -i

```

**5) Install your project dependencies (editable) and verify**

```bash
# From the root of this repo, with the env active
pip install -e source/RL_UR5

```

**6) Download the 3D assets**

Download the 3D assets from: https://clemson.box.com/s/raeoeb7gcislpjend57gj5im4q5p24h2

Place the downloaded assets in the following folder (replace the existing assets in that folder):

`/RL_UR5_IsaacLab/source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/assets`



Note: the assets must be downloaded and placed into the assets folder above before running the tasks or training — tasks expect these assets to be present.

**7) Weights & Biases (WandB) integration (optional but recommended)**

WandB is used for experiment tracking and visualizing training metrics. To enable WandB integration for this project:

- Install the WandB client in the active Conda environment:

```bash
pip install wandb
```

- Login to WandB (interactive) or provide an API key via environment variable:

```bash
# interactive login (recommended for local use)
wandb login

# or set the API key in CI/headless setups
export WANDB_API_KEY="<your_api_key_here>"
```

- Enable WandB in the project configuration at:

`source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/agents/PPO_skrl_camera.yaml`

Set `experiment.wandb.enabled: true` and set `experiment.wandb.project` / `experiment.wandb.entity` to your project and account. You can also enable offline logging or artifact uploads there.

Note: ensure you have network access and a WandB account (or set `WANDB_API_KEY`) before running training with WandB enabled. If you prefer not to use WandB, leave `experiment.wandb.enabled` set to `false`.




Notes and troubleshooting:
- Ensure OS is Ubuntu 22.04 LTS (required for building Isaac Sim from source).
- Isaac Sim 5.x requires Python 3.11 — the Python interpreter in your virtual environment must match the simulator's Python version.
- If you see `ModuleNotFoundError: No module named 'isaacsim'`, ensure the virtual environment is activated and `_isaac_sim/setup_conda_env.sh` (or the corresponding setup script) has been executed.
- If switching from older Isaac Sim versions, you may want to reset user data after the first run:

```bash
${ISAACSIM_PATH}/isaac-sim.sh --reset-user
```

If you prefer not to build from source, you can use pre-built packages for Isaac Sim (not covered here) or follow the Isaac Lab pip/binaries installation guides linked in the official docs.

---

## 🎯 Training

### Quick Start

Train the UR5 manipulator with vision-based reinforcement learning:

```bash
python scripts/skrl/train.py \
    --task=Isaac-UR5-HuberDirectObj-PPO \
    --num_envs 2 \
    --enable_cameras \
    --headless
```

### Command Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Training environment/task name | Required |
| `--num_envs` | Number of parallel simulation environments | 2 |
| `--enable_cameras` | Enable camera sensors for vision-based control | False |
| `--headless` | Run without GUI rendering (faster training) | False |
| `--seed` | Random seed for reproducibility | 42 |
| `--max_iterations` | Maximum training iterations | 10000 |

### Advanced Training Example

For longer training with more environments:

```bash
python scripts/skrl/train.py \
    --task=Isaac-UR5-HuberDirectObj-PPO \
    --num_envs 64 \
    --enable_cameras \
    --headless \
    --seed 123 \
    --max_iterations 50000
```

### Monitoring Training Progress

Training logs and checkpoints are automatically saved to:
```
logs/skrl/logs/<experiment_name>/<timestamp>/
├── checkpoints/
│   ├── best_agent.pt      # Best performing model
│   └── agent_XXXX.pt      # Periodic checkpoints
├── tensorboard/
└── config.yaml
```

---

## 🎮 Evaluation

### Playing a Trained Model

To visualize and evaluate a trained checkpoint:

```bash
python scripts/skrl/play.py \
    --task=Isaac-UR5-HuberDirectObj-PPO \
    --num_envs 2 \
    --enable_cameras \
    --checkpoint logs/skrl/logs/skrl_camera_pose_tracking/arm_avoidance_v1/checkpoints/best_agent.pt
```

### Evaluation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to trained model checkpoint | Required |
| `--num_envs` | Number of parallel evaluation environments | 2 |
| `--enable_cameras` | Enable camera rendering | False |
| `--record_video` | Record evaluation episodes | False |
| `--video_length` | Number of steps to record | 500 |

### Batch Evaluation

Evaluate multiple checkpoints or conditions:

```bash
# Evaluate with different environment counts
for n in 1 2 4 8; do
    python scripts/skrl/play.py \
        --task=Isaac-UR5-HuberDirectObj-PPO \
        --num_envs $n \
        --enable_cameras \
        --checkpoint <your_checkpoint_path>
done
```

---

## ⚙️ Configuration

### WandB Integration

This project includes Weights & Biases (WandB) integration for experiment tracking. Configuration is located at:

```
source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/agents/PPO_skrl_camera.yaml
```

To enable WandB logging, modify the configuration:

```yaml
# In PPO_skrl_camera.yaml
experiment:
  wandb:
    enabled: true
    project: "ur5-manipulation"
    entity: "your-wandb-username"
    tags: ["ur5", "vision", "ppo"]
```

### Environment Configuration

Customize task parameters in the environment configuration files:

```yaml
# Example configuration structure
sim:
  dt: 0.01                    # Simulation timestep
  substeps: 1                  # Physics substeps
  
env:
  num_envs: 2048              # Number of parallel environments
  episode_length_s: 10.0      # Episode duration in seconds
  
robot:
  controller:
    type: "joint_position"    # Controller type
    stiffness: 800.0
    damping: 40.0
```

---

## 📊 Results

Our trained models achieve:
- **Success Rate**: 90%+ on arm avoidance tasks
- **Training Time**: ~10 hours on RTX 3080 (128 environments with Tiled Camera Data)
- **Sim-to-Real Gap**: Minimal with proper domain randomization

### Visualizations

Training progress and evaluation videos are automatically saved to the logs directory. View them with:

```bash
# TensorBoard visualization
tensorboard --logdir logs/skrl/logs/

# Video playback
python scripts/visualize_results.py --log_dir logs/skrl/logs/<experiment_name>
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce number of environments
python scripts/skrl/train.py --task=Isaac-UR5-HuberDirectObj-PPO --num_envs 1 --enable_cameras
```

**2. Camera not rendering**
- Ensure `--enable_cameras` flag is set
- Check GPU drivers support RTX rendering
- Try running without `--headless` for debugging

**3. Module not found errors**
```bash
# Ensure conda environment is activated
conda activate isaaclab

# Reinstall project dependencies
pip install -e source/distMARL --force-reinstall
```

### Getting Help

- 📚 Check the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab)
- 💬 Open an issue on our GitHub repository

---

## 🚧 Coming Soon: Real Robot Deployment

We're actively working on deploying our trained policies to physical UR5 robots! The upcoming release will include:

### **Planned Features**

- **🤖 Real UR5 Integration**: Direct deployment pipeline from simulation to physical UR5 arm
- **📦 Pre-trained Checkpoints**: Battle-tested models ready for real-world deployment
- **🔌 ROS2 Bridge**: Seamless integration with ROS2 for robot control and sensor data
- **📷 Camera Calibration**: Automated tools for camera-robot calibration
- **🛡️ Safety Layers**: Built-in collision detection and emergency stop mechanisms
- **📊 Real-time Monitoring**: Live visualization of robot state and vision input


## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{ur5_isaac_manipulation,
  author = {Aditya Parameshwaran},
  title = {Vision-Based UR5 Manipulation in Isaac Lab},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ur5-isaac-lab}
}
```

---

## 📄 License

Copyright (c) 2024, [Your Name/Organization]. All rights reserved.

This project is released under the [BSD-3-Clause License](LICENSE). See the [LICENSE](LICENSE) file for full details.

### Third-Party Licenses

This project incorporates code from:
- **Isaac Lab**: BSD-3-Clause License
- **NVIDIA Isaac Sim**: Subject to NVIDIA EULA
- **Python Dependencies**: Various licenses (see `requirements.txt`)

For a complete list of third-party licenses, please refer to the `docs/licenses/` directory.

---

## 🙏 Acknowledgments

This work builds upon:
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) for the simulation framework
- [SKRL](https://github.com/Toni-SM/skrl) for reinforcement learning algorithms
- The robotics research community for continuous inspiration

---

<div align="center">
  <b>Happy Training! 🚀</b>
</div>
