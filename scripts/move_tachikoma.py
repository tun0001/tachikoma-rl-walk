import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Move Tachikoma joints one at a time.")
parser.add_argument("--joint_time", type=float, default=1.5, help="Seconds for each joint motion.")
parser.add_argument("--motion_time", type=float, default=1.0, help="Seconds for one joint sweep.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from tachikoma_lab.assets.tachikoma import TACHIKOMA_CFG


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(eye=(3.5, 3.5, 2.2), target=(0.0, 0.0, 0.125))

    ground_cfg = sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.85, 1.0))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(TACHIKOMA_CFG.replace(prim_path="/World/Tachikoma"))
    sim.reset()

    joint_count = len(robot.joint_names)
    print(f"Moving {joint_count} joints one at a time.")
    print(f"Joints: {robot.joint_names}")

    default_pos = robot.data.default_joint_pos.clone()
    limits = robot.data.soft_joint_pos_limits[0]
    sim_dt = sim.get_physics_dt()
    joint_index = 0
    joint_elapsed = 0.0

    while simulation_app.is_running():
        target = default_pos.clone()
        lower, upper = limits[joint_index]
        center = (lower + upper) * 0.5
        amplitude = (upper - lower) * 0.25
        phase = 2.0 * math.pi * (joint_elapsed / args_cli.motion_time)
        target[:, joint_index] = center + amplitude * math.sin(phase)

        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        joint_elapsed += sim_dt
        if joint_elapsed >= args_cli.joint_time:
            print(f"Finished: {robot.joint_names[joint_index]}")
            joint_index = (joint_index + 1) % joint_count
            joint_elapsed = 0.0


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
