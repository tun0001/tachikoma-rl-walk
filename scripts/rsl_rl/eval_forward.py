"""Measure whether a trained Tachikoma policy walks forward (not sideways).

Every environment receives the same fixed command (forward speed only), and the script reports the
mean body-frame velocity in Tachikoma's (forward, left) frame plus the heading drift.

Example:
    python scripts/rsl_rl/eval_forward.py --task Tachikoma-Walk-v0 --headless --num_envs 256 --cmd_vx 0.2
"""

import argparse
import math
import importlib.metadata as metadata
import os
import sys

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import quat_apply
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import tachikoma_lab.tasks  # noqa: F401
from tachikoma_lab.tasks.manager_based.tachikoma_lab.mdp.rewards import support_margin, tachikoma_lin_vel
from tachikoma_lab.tasks.manager_based.tachikoma_lab.terrains import make_test_course, make_tachikoma_terrain_importer
from tachikoma_lab.tasks.manager_based.tachikoma_lab.tachikoma_lab_env_cfg import (
    TACHIKOMA_FOOT_LINK_NAMES_CYCLIC,
    TACHIKOMA_FOOT_TIP_OFFSET,
)

parser = argparse.ArgumentParser(description="Evaluate forward walking of a Tachikoma policy.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tachikoma-Walk-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--cmd_vx", type=float, default=0.2, help="Forward speed command [m/s].")
parser.add_argument("--seconds", type=float, default=10.0, help="Evaluation length [s].")
parser.add_argument("--warmup", type=float, default=2.0, help="Seconds ignored at the start.")
parser.add_argument(
    "--terrain", type=str, default="flat", choices=["flat", "step", "rough"], help="Test course to walk on."
)
parser.add_argument("--terrain_height", type=float, default=0.02, help="Step height / max bump height [m].")
parser.add_argument("--video", action="store_true", default=False, help="Record env 0 with a camera that follows it.")
parser.add_argument(
    "--heading_hold",
    type=float,
    default=0.0,
    help="Gain of a heading-hold P controller wrapped around the policy (0 = off): the yaw-rate command becomes"
    " gain * (initial heading - heading), like an IMU-based heading hold on the real robot.",
)
parser.add_argument("--heading_max", type=float, default=0.3, help="Yaw-rate command limit of the heading hold [rad/s].")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # fixed straight-ahead command, no episode timeouts during the evaluation window
        cmd = env_cfg.commands.base_velocity
        cmd.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        # heading-command tasks (v4) would steer towards a random heading; use --heading_hold instead
        cmd.heading_command = False
        cmd.ranges.heading = None
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.episode_length_s = args_cli.seconds + args_cli.warmup + 5.0
        # tasks trained on terrain: evaluate on the chosen course only, with a fixed difficulty
        if getattr(env_cfg, "curriculum", None) is not None:
            env_cfg.curriculum = None
        if args_cli.terrain == "flat" and getattr(env_cfg.scene, "terrain", None) is not None:
            env_cfg.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        if args_cli.terrain != "flat":
            # low_base checks the absolute height; the stairs' start platform lies below z=0, so it would end
            # every episode at once. Falls are still caught by bad_orientation and base_contact.
            env_cfg.terminations.low_base = None
            env_cfg.scene.ground = None
            env_cfg.scene.terrain = make_tachikoma_terrain_importer(
                make_test_course(args_cli.terrain, args_cli.terrain_height)
            )

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[EVAL] checkpoint: {resume_path}")

        robot = env.unwrapped.scene["robot"]
        feet = env.unwrapped.scene.sensors["foot_contact"]
        margin_feet = SceneEntityCfg("foot_contact", body_names=TACHIKOMA_FOOT_LINK_NAMES_CYCLIC, preserve_order=True)
        margin_robot = SceneEntityCfg("robot", body_names=TACHIKOMA_FOOT_LINK_NAMES_CYCLIC, preserve_order=True)
        margin_feet.resolve(env.unwrapped.scene)
        margin_robot.resolve(env.unwrapped.scene)
        # foot tips in the contact sensor's body order, for the foot-lift measurement
        tips_cfg = SceneEntityCfg("robot", body_names=list(feet.body_names), preserve_order=True)
        tips_cfg.resolve(env.unwrapped.scene)
        step_dt = env.unwrapped.step_dt
        warmup_steps = int(args_cli.warmup / step_dt)
        total_steps = warmup_steps + int(args_cli.seconds / step_dt)

        obs = env.get_observations()
        vel_sum = torch.zeros(env.num_envs, 3, device=env.unwrapped.device)
        fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        # gait statistics: how many feet are on the ground, and how long each swing lasts
        support_hist = torch.zeros(env.num_envs, 5, device=env.unwrapped.device)
        air_time_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
        step_count = torch.zeros(env.num_envs, device=env.unwrapped.device)
        margin_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
        unstable_steps = torch.zeros(env.num_envs, device=env.unwrapped.device)
        prev_contact = torch.ones(env.num_envs, 4, dtype=torch.bool, device=env.unwrapped.device)
        air_frames = torch.zeros(env.num_envs, 4, device=env.unwrapped.device)
        # foot lift: highest tip point during each swing, measured from where the foot left the ground
        lift_sum = torch.zeros(env.num_envs, device=env.unwrapped.device)
        lift_start = torch.zeros(env.num_envs, 4, device=env.unwrapped.device)
        swing_max = torch.zeros(env.num_envs, 4, device=env.unwrapped.device)
        climb_max = torch.zeros(env.num_envs, device=env.unwrapped.device)
        start_pos = start_yaw = start_z = None
        frames = []
        if args_cli.video:
            # the video recorder renders through this Kit camera and only places it once
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
        # The kit viewport camera does not track assets in headless mode, so move it ourselves: a side view
        # from the robot's right (body +X) at its initial heading, so forward motion goes across the screen.
        cam_yaw = float(_yaw(robot.data.root_quat_w.torch[:1])[0])
        cam_offset = torch.tensor([0.9 * math.cos(cam_yaw), 0.9 * math.sin(cam_yaw), 0.35])
        velocity_command = env.unwrapped.command_manager.get_term("base_velocity")
        target_yaw = _yaw(robot.data.root_quat_w.torch)
        with torch.inference_mode():
            for step in range(total_steps):
                obs, _, dones, _ = env.step(policy(obs))
                if args_cli.heading_hold > 0.0:
                    # the policy sees the new command from the next step on (one 0.02 s step of delay)
                    yaw_error = _wrap(target_yaw - _yaw(robot.data.root_quat_w.torch))
                    velocity_command.vel_command_b[:, 2] = torch.clamp(
                        args_cli.heading_hold * yaw_error, -args_cli.heading_max, args_cli.heading_max
                    )
                if args_cli.video:
                    root = robot.data.root_pos_w.torch[0].cpu()
                    target = root.clone()
                    target[2] -= 0.05
                    set_kit_renderer_camera_view(eye=(root + cam_offset).tolist(), target=target.tolist())
                    frames.append(env.unwrapped.render())
                fallen |= dones.bool()
                if step == warmup_steps:
                    start_pos = robot.data.root_pos_w.torch[:, :2].clone()
                    start_yaw = _yaw(robot.data.root_quat_w.torch)
                    yaw_at_start = start_yaw.clone()
                    start_z = robot.data.root_pos_w.torch[:, 2].clone()
                    lift_start = _foot_tips_z(robot, tips_cfg)
                    swing_max = lift_start.clone()
                if step >= warmup_steps:
                    vel_sum += tachikoma_lin_vel(robot)
                    in_contact = feet.data.net_forces_w.torch.norm(dim=-1) > 1.0
                    support_hist[torch.arange(env.num_envs), in_contact.sum(dim=1)] += 1.0
                    # a step = foot leaves the ground and lands again; air time counted in policy steps
                    touchdown = in_contact & ~prev_contact
                    liftoff = ~in_contact & prev_contact
                    tip_z = _foot_tips_z(robot, tips_cfg)
                    lift_start = torch.where(liftoff, tip_z, lift_start)
                    swing_max = torch.where(liftoff, tip_z, torch.maximum(swing_max, tip_z))
                    lift_sum += ((swing_max - lift_start) * touchdown).sum(dim=1)
                    climb_max = torch.maximum(climb_max, robot.data.root_pos_w.torch[:, 2] - start_z)
                    air_frames += (~in_contact).float()
                    air_time_sum += (air_frames * step_dt * touchdown).sum(dim=1)
                    step_count += touchdown.sum(dim=1)
                    air_frames *= (~in_contact).float()
                    prev_contact = in_contact
                    margin = support_margin(env.unwrapped, margin_feet, margin_robot, TACHIKOMA_FOOT_TIP_OFFSET)
                    margin_sum += margin
                    unstable_steps += (margin < 0.0).float()

        n = total_steps - warmup_steps
        vel = vel_sum / n
        ok = ~fallen
        heading_drift = torch.rad2deg(_wrap(_yaw(robot.data.root_quat_w.torch) - yaw_at_start))
        # displacement expressed in the start heading's (forward=+Y_body, left=-X_body) frame
        disp_w = robot.data.root_pos_w.torch[:, :2] - start_pos
        c, s = torch.cos(start_yaw), torch.sin(start_yaw)
        disp_bx = c * disp_w[:, 0] + s * disp_w[:, 1]
        disp_by = -s * disp_w[:, 0] + c * disp_w[:, 1]
        print("=" * 60)
        print(f"[EVAL] command forward      : {args_cli.cmd_vx:.3f} m/s")
        print(f"[EVAL] envs without reset   : {int(ok.sum())}/{env.num_envs}")
        print(f"[EVAL] mean forward  vel    : {vel[ok, 0].mean():+.3f} m/s")
        print(f"[EVAL] mean left     vel    : {vel[ok, 1].mean():+.3f} m/s")
        print(f"[EVAL] |left| / |forward|   : {vel[ok, 1].abs().mean() / vel[ok, 0].abs().mean().clamp(min=1e-6):.2f}")
        print(f"[EVAL] forward displacement : {disp_by[ok].mean():+.3f} m in {args_cli.seconds:.0f} s")
        print(f"[EVAL] left    displacement : {-disp_bx[ok].mean():+.3f} m")
        print(f"[EVAL] total distance       : {disp_w[ok].norm(dim=1).mean():.3f} m (heading-independent)")
        print(f"[EVAL] heading drift        : {heading_drift[ok].mean():+.1f} deg (|.| {heading_drift[ok].abs().mean():.1f})")
        support = support_hist[ok].sum(dim=0) / support_hist[ok].sum()
        print("[EVAL] feet on ground       : " + "  ".join(f"{k}:{100 * support[k]:4.1f}%" for k in range(5)))
        print(f"[EVAL] steps per foot       : {step_count[ok].mean() / 4 / args_cli.seconds:.2f} /s")
        print(f"[EVAL] stability margin     : mean {1000 * margin_sum[ok].mean() / n:+.1f} mm, "
              f"CoM outside support {100 * unstable_steps[ok].mean() / n:.1f}% of the time")
        print(f"[EVAL] mean swing (air) time: {air_time_sum[ok].sum() / step_count[ok].sum().clamp(min=1):.3f} s")
        print(f"[EVAL] mean foot lift       : {1000 * lift_sum[ok].sum() / step_count[ok].sum().clamp(min=1):.1f} mm")
        if args_cli.terrain == "step":
            h = args_cli.terrain_height
            print(f"[EVAL] course               : stairs, {1000 * h:.0f} mm per step")
            print(f"[EVAL] height climbed       : mean {1000 * climb_max.mean():.0f} mm "
                  f"(= {climb_max.mean() / h:.1f} steps), max {1000 * climb_max.max():.0f} mm")
            print(f"[EVAL] climbed >= 1 step    : {100 * (climb_max >= 0.8 * h).float().mean():.0f}% of robots "
                  f"(fallen: {int(fallen.sum())}/{env.num_envs})")
        elif args_cli.terrain == "rough":
            print(f"[EVAL] course               : random bumps up to {1000 * args_cli.terrain_height:.0f} mm "
                  f"(fallen: {int(fallen.sum())}/{env.num_envs})")
        print("=" * 60)
        if args_cli.video:
            import imageio

            course_tag = "flat" if args_cli.terrain == "flat" else f"{args_cli.terrain}{1000 * args_cli.terrain_height:.0f}mm"

            video_path = os.path.join(os.path.dirname(resume_path), "videos", f"{course_tag}_{os.path.basename(resume_path)[:-3]}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            # the first frames are rendered before the camera moves; skip them
            imageio.mimsave(video_path, frames[5:], fps=round(1.0 / step_dt))
            print(f"[EVAL] video: {video_path}")
        env.close()


def _foot_tips_z(robot, tips_cfg: SceneEntityCfg) -> torch.Tensor:
    quat = robot.data.body_link_quat_w.torch[:, tips_cfg.body_ids]
    pos = robot.data.body_link_pos_w.torch[:, tips_cfg.body_ids]
    offset = torch.tensor(TACHIKOMA_FOOT_TIP_OFFSET, device=quat.device).expand(quat.shape[:-1] + (3,))
    return (pos + quat_apply(quat, offset))[..., 2]


def _yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    # Isaac Lab 3.x stores quaternions as (x, y, z, w).
    x, y, z, w = quat_xyzw.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


if __name__ == "__main__":
    main()
