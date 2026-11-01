# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


def base_height_above_ground_below(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base is closer than ``minimum_height`` [m] to the ground right below it.

    ``root_height_below_minimum`` compares against z = 0, which is meaningless on terrain (e.g. the start
    platform of the stairs lies below z = 0). The ground height comes from a downward height scanner.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_z = scanner.data.ray_hits_w.torch[..., 2]
    # rays that miss return inf; ignore them (fall back to z = 0 if every ray missed)
    valid = torch.isfinite(hits_z)
    ground_z = torch.where(valid, hits_z, torch.zeros_like(hits_z)).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return asset.data.root_pos_w.torch[:, 2] - ground_z < minimum_height
