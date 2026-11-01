# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def terrain_levels_tachikoma(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    promote_ratio: float = 0.8,
    demote_ratio: float = 0.4,
    min_commanded_distance: float = 0.3,
    demote_on_fall: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain curriculum judged against the distance the robot was *commanded* to walk.

    Isaac Lab's ``terrain_levels_vel`` promotes a robot only after it walks half a terrain tile (1.5 m here),
    which a 0.1 m/s robot can hardly reach in one episode. Here a robot moves up when it covers at least
    ``promote_ratio`` of its commanded distance and down when it covers less than ``demote_ratio`` of it.
    Episodes with a near-zero command (standing) change nothing.

    With ``demote_on_fall`` a robot is also moved down when its episode ended by a termination other than the
    time-out. terrain_v1 showed that a high ``demote_ratio`` alone lets the policy escape hard terrain by
    simply walking slowly there, so v2 demotes mainly on falls and keeps only a low distance threshold.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.linalg.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    commanded = torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    judged = commanded > min_commanded_distance
    move_up = judged & (distance > promote_ratio * commanded)
    move_down = judged & (distance < demote_ratio * commanded)
    if demote_on_fall:
        # still valid here: the curriculum runs inside the reset, before the termination buffers are cleared
        fell = env.termination_manager.terminated[env_ids]
        move_up &= ~fell
        move_down |= fell
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term: str = "track_lin_vel_xy",
    promote_ratio: float = 0.75,
    demote_ratio: float = 0.4,
    min_distance: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain curriculum judged by how well the robot tracked its velocity commands over the whole episode.

    The distance-based ``terrain_levels_tachikoma`` turned out to be a poor judge (terrain_v2): the command is
    resampled every 8-12 s but only the last one was used, yaw commands make the path curve on purpose, and the
    +-0.25 m spawn offset alone could cover a slow command's distance. Instead, the episode sum of the velocity
    tracking reward is compared with its maximum (a full episode of perfect tracking):

    - up:   tracking ratio >= ``promote_ratio`` and the robot ended at least ``min_distance`` from the tile
      centre (so a robot that only stood still under zero commands is not promoted)
    - down: the episode ended by a fall, or tracking ratio < ``demote_ratio``
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    # the curriculum runs inside the reset, before the reward manager logs and clears its episode sums
    tracking_sum = env.reward_manager._episode_sums[reward_term][env_ids]
    weight = env.reward_manager.get_term_cfg(reward_term).weight
    tracking_ratio = tracking_sum / (weight * env.max_episode_length_s)
    distance = torch.linalg.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    fell = env.termination_manager.terminated[env_ids]
    move_up = (tracking_ratio >= promote_ratio) & (distance >= min_distance) & ~fell
    move_down = (fell | (tracking_ratio < demote_ratio)) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
