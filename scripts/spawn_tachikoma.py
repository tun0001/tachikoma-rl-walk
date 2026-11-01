import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn Tachikoma in Isaac Sim.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from tachikoma_lab.assets.tachikoma import TACHIKOMA_CFG


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(3.5, 3.5, 2.2),
        target=(0.0, 0.0, 0.125),
    )

    ground_cfg = sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(
        intensity=2500.0,
        color=(0.8, 0.85, 1.0),
    )
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = TACHIKOMA_CFG.replace(prim_path="/World/Tachikoma")
    robot = Articulation(robot_cfg)

    sim.reset()
    print(f"Spawned Tachikoma with {len(robot.joint_names)} joints.")
    print(f"Joints: {robot.joint_names}")

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
