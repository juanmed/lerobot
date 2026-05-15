#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class OmxFollowerArmConfig:
    """Per-arm config for BiOmxFollower. Plain dataclass — NOT a RobotConfig subclass."""

    port: str

    disable_torque_on_disconnect: bool = True

    # Limits the magnitude of the relative positional target vector for safety.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to True for backward compatibility with previous policies/datasets.
    use_degrees: bool = False


@RobotConfig.register_subclass("bi_omx_follower")
@dataclass
class BiOmxFollowerConfig(RobotConfig):
    """Configuration for a bimanual OMX follower robot (two OmxFollower arms)."""

    left_arm_config: OmxFollowerArmConfig
    right_arm_config: OmxFollowerArmConfig
