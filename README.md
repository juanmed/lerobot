<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** in this fork is focused on real-world robotics dataset workflows: teleoperation, recording, replay, visualization, editing, and upload. The goal is to keep the data collection toolchain lightweight and practical for hardware-first setups such as a Raspberry Pi 5.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 A dataset-only repository layout with the training, evaluation, RL, and policy-inference surface removed from the codebase.

## Quick Start

This repository is a local fork. Installing `lerobot` from PyPI will install the upstream package, not this fork.

To run this repository, install it from this checkout or run it directly from the repository root.

```bash
git clone <your-fork-url> jlerobot
cd jlerobot
# Optional but recommended on Debian/Bookworm if you want voice prompts:
# sudo apt install speech-dispatcher
pip install -e .
python -m pip install dynamixel-sdk safetensors
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
lerobot-info
```

If you do not want to install the checkout into the environment, you can run commands directly from the repo with:

```bash
cd /path/to/jlerobot
PYTHONPATH=src python -m lerobot.scripts.lerobot_info
```

> [!IMPORTANT]
> If you already have another editable `lerobot` checkout in the same environment, the global console scripts may resolve that other checkout. In that case, either reinstall this fork with `pip install -e .` from this repository or use `PYTHONPATH=src python -m ...` from this repository root.

> [!IMPORTANT]
> This fork is intentionally dataset-only. The upstream training, policy, RL, and simulation instructions do not apply here.

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices for recording and replay workflows.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = {...}  # action from a teleoperator or replayed dataset frame
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can implement the Robot interface to use the same recording, replay, and dataset tooling with your own hardware.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Record new datasets, replay episodes, visualize recordings, upload to the Hub, and edit datasets by deleting episodes, splitting by indices/fractions, adding/removing features, and merging multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## Recording, Visualization, and Upload

This fork is intended for dataset collection workflows. The policy training and inference stack has been removed.

### Record a dataset

Use `lerobot-record` with a robot config, teleoperator config, and dataset destination:

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyUSB0 \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyUSB1 \
  --dataset.repo_id=<hf_user>/<dataset_name> \
  --dataset.single_task="Pick up the cube" \
  --dataset.num_episodes=10 \
  --display_data=true
```

### Visualize a dataset

Use `lerobot-dataset-viz` to inspect a recorded episode locally or save a `.rrd` file for later viewing:

```bash
lerobot-dataset-viz \
  --repo-id <hf_user>/<dataset_name> \
  --episode-index 0
```

You can also point it at a local dataset root:

```bash
lerobot-dataset-viz \
  --repo-id <hf_user>/<dataset_name> \
  --root /path/to/local/dataset \
  --episode-index 0
```

### Upload a dataset

You can upload during recording:

```bash
lerobot-record \
  ... \
  --dataset.push_to_hub=true
```

Or upload/edit an existing dataset with `lerobot-edit-dataset`, for example:

```bash
lerobot-edit-dataset \
  --repo_id <hf_user>/<dataset_name> \
  --operation.type info
```

This fork keeps the Hugging Face dataset upload path intact for recorded datasets.

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** Upstream reference documentation for dataset format and hardware integration. Training and inference sections do not apply to this fork.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.

## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](./CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
