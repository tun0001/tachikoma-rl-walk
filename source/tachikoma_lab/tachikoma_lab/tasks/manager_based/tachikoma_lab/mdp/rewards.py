# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def tachikoma_lin_vel(asset: Articulation) -> torch.Tensor:
    """Base linear velocity in Tachikoma's (forward, left, up) frame.

    The URDF base_link uses +Y = forward and +X = right, so forward = vy and left = -vx.
    """
    vel_b = asset.data.root_lin_vel_b.torch
    return torch.stack((vel_b[:, 1], -vel_b[:, 0], vel_b[:, 2]), dim=1)


def tachikoma_base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Observation: base linear velocity in the same frame as the velocity command."""
    return tachikoma_lin_vel(env.scene[asset_cfg.name])


def track_tachikoma_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track the (forward, left) velocity command in Tachikoma's body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    velocity_error = torch.sum(torch.square(command - tachikoma_lin_vel(asset)[:, :2]), dim=1)
    return torch.exp(-velocity_error / std**2)


def _feet_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float) -> torch.Tensor:
    """Boolean (num_envs, num_feet): foot pushed on the ground in any of the recent sensor samples."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).max(dim=1)[0] > force_threshold


def feet_support_below(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, min_feet: int = 3, force_threshold: float = 1.0
) -> torch.Tensor:
    """1 while fewer than ``min_feet`` feet are on the ground (penalize with a negative weight).

    Keeping three feet down at all times leads to a statically stable crawl gait.
    """
    num_contacts = torch.sum(_feet_in_contact(env, sensor_cfg, force_threshold), dim=1)
    return (num_contacts < min_feet).float()


def feet_tip_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    tip_offset: tuple[float, float, float],
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Horizontal speed [m/s] of the foot tips that are on the ground, summed over feet.

    The tibia link origin is at the knee, ~0.15 m above the tip, so the tip velocity is computed as
    v_tip = v_link + w x (R * tip_offset).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    lin_vel = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids]
    offset = torch.tensor(tip_offset, device=quat.device).expand(quat.shape[:-1] + (3,))
    tip_vel = lin_vel + torch.cross(ang_vel, quat_apply(quat, offset), dim=-1)
    in_contact = _feet_in_contact(env, sensor_cfg, force_threshold)
    return torch.sum(tip_vel[..., :2].norm(dim=-1) * in_contact, dim=1)


def support_margin(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    tip_offset: tuple[float, float, float],
    force_threshold: float = 1.0,
    unsupported_margin: float = -0.03,
) -> torch.Tensor:
    """Static stability margin [m]: distance of the whole-body CoM (projected on the ground) inside the
    polygon spanned by the feet that are on the ground. Negative means the robot would tip over.

    ``sensor_cfg``/``asset_cfg`` must list the feet in cyclic (clockwise from above) order, e.g. fr, rr, rl, fl.
    With fewer than three feet down there is no support polygon and ``unsupported_margin`` is returned.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # whole-body center of mass (xy)
    mass = asset.data.default_mass.torch.to(env.device)
    com_w = asset.data.body_com_pos_w.torch[..., :2]
    com = torch.sum(com_w * mass.unsqueeze(-1), dim=1) / torch.sum(mass, dim=1, keepdim=True)
    # foot tips (xy)
    quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    pos = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    offset = torch.tensor(tip_offset, device=quat.device).expand(quat.shape[:-1] + (3,))
    tips = (pos + quat_apply(quat, offset))[..., :2]
    contact = _feet_in_contact(env, sensor_cfg, force_threshold)
    num_feet = contact.shape[1]
    # for every contacting foot, the next contacting foot in cyclic order closes one polygon edge
    margin = torch.full((env.num_envs,), float("inf"), device=env.device)
    for i in range(num_feet):
        next_idx = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        for off in range(num_feet - 1, 0, -1):
            cand = (i + off) % num_feet
            next_idx = torch.where(contact[:, cand], torch.full_like(next_idx, cand), next_idx)
        valid = contact[:, i] & (next_idx >= 0)
        a = tips[:, i]
        b = torch.gather(tips, 1, next_idx.clamp(min=0).view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        edge = b - a
        # clockwise polygon: the interior is on the right of each edge
        normal = torch.stack((edge[:, 1], -edge[:, 0]), dim=1) / edge.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dist = torch.sum((com - a) * normal, dim=1)
        margin = torch.where(valid, torch.minimum(margin, dist), margin)
    supported = torch.sum(contact, dim=1) >= 3
    return torch.where(supported, margin, torch.full_like(margin, unsupported_margin))


def support_margin_below(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    tip_offset: tuple[float, float, float],
    target: float = 0.01,
) -> torch.Tensor:
    """Shortfall [m] of the static stability margin below ``target`` (penalize with a negative weight)."""
    margin = support_margin(env, sensor_cfg, asset_cfg, tip_offset)
    return torch.clamp(target - margin, min=0.0, max=0.05)


class FootClearance(ManagerTermBase):
    """Penalize moving feet whose tip is not lifted to ``target_height`` above the ground it last stood on.

    Per foot: ((clearance - target) / target)^2 * tanh(k * |v_tip_xy|), summed over feet. A foot sliding on the
    ground is ``target`` short and pays ~1; a planted (still) foot pays nothing; overshooting is penalized too,
    so the legs are not flung high. Only moving feet count, so keeping one foot raised in place earns nothing.

    The ground reference is the tip height at the last moment the foot was on the ground, so the term also
    works on steps and bumps, where the absolute z of the floor differs per foot.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ground_z: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._ground_z is not None:
            # re-initialized from the tips on the next call (the robot is not placed yet during reset)
            self._ground_z[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        tip_offset: tuple[float, float, float],
        target_height: float,
        velocity_gain: float = 10.0,
        force_threshold: float = 1.0,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
        pos = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
        lin_vel = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]
        ang_vel = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids]
        offset = torch.tensor(tip_offset, device=quat.device).expand(quat.shape[:-1] + (3,))
        tip_rel = quat_apply(quat, offset)
        tip_z = (pos + tip_rel)[..., 2]
        tip_speed = (lin_vel + torch.cross(ang_vel, tip_rel, dim=-1))[..., :2].norm(dim=-1)

        in_contact = _feet_in_contact(env, sensor_cfg, force_threshold)
        if self._ground_z is None:
            self._ground_z = tip_z.clone()
        self._ground_z = torch.where(torch.isnan(self._ground_z) | in_contact, tip_z, self._ground_z)

        clearance = tip_z - self._ground_z
        error = torch.square((clearance - target_height) / target_height)
        return torch.sum(error * torch.tanh(velocity_gain * tip_speed), dim=1)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, min_command: float = 0.05
) -> torch.Tensor:
    """Reward steps whose swing lasts longer than ``threshold`` seconds (zero while standing)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > min_command
    return reward

