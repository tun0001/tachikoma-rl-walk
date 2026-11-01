from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets" / "usd" / "tachikoma"
_TACHIKOMA_USD = _ASSET_ROOT / "tachikoma.usda"


# Body and foot links used by the RL task.
TACHIKOMA_BODY_LINK_NAME = "base_link"
TACHIKOMA_STAND_HEIGHT = 0.125
TACHIKOMA_FOOT_LINK_NAMES = [
    "leg_fr_tibia",
    "leg_fl_tibia",
    "leg_rl_tibia",
    "leg_rr_tibia",
]

# Joint groups and limits taken from assets/tachikoma/tachikoma.urdf.
TACHIKOMA_LEG_JOINT_NAMES = [
    "leg_fr_yaw",
    "leg_fr_pitch",
    "leg_fr_knee",
    "leg_fl_yaw",
    "leg_fl_pitch",
    "leg_fl_knee",
    "leg_rl_yaw",
    "leg_rl_pitch",
    "leg_rl_knee",
    "leg_rr_yaw",
    "leg_rr_pitch",
    "leg_rr_knee",
]

# The RL task currently controls the 12 leg joints only.
TACHIKOMA_ACTUATED_JOINT_NAMES = TACHIKOMA_LEG_JOINT_NAMES

TACHIKOMA_ARM_JOINT_NAMES = [
    "arm_r_yaw",
    "arm_r_pitch",
    "arm_r_elbow",
    "arm_l_yaw",
    "arm_l_pitch",
    "arm_l_elbow",
]

TACHIKOMA_EYE_JOINT_NAMES = [
    "eye_r_roll",
    "eye_l_roll",
]

TACHIKOMA_JOINT_LIMITS = {
    "leg_fr_yaw": (-0.698132, 0.698132),
    "leg_fr_pitch": (-0.785398, 0.959931),
    "leg_fr_knee": (-0.767945, 0.767945),
    "leg_fl_yaw": (-0.698132, 0.698132),
    "leg_fl_pitch": (-0.785398, 0.959931),
    "leg_fl_knee": (-0.767945, 0.767945),
    "leg_rl_yaw": (-0.698132, 0.698132),
    "leg_rl_pitch": (-0.785398, 0.959931),
    "leg_rl_knee": (-0.767945, 0.767945),
    "leg_rr_yaw": (-0.698132, 0.698132),
    "leg_rr_pitch": (-0.785398, 0.959931),
    "leg_rr_knee": (-0.767945, 0.767945),
    "arm_r_yaw": (-0.261799, 0.261799),
    "arm_r_pitch": (-0.785398, 1.483530),
    "arm_r_elbow": (0.0, 1.658063),
    "arm_l_yaw": (-0.261799, 0.261799),
    "arm_l_pitch": (-0.785398, 1.483530),
    "arm_l_elbow": (0.0, 1.658063),
    "eye_r_roll": (-1.396263, 1.396263),
    "eye_l_roll": (-1.396263, 1.396263),
}

# A reasonable standing pose from assets/tachikoma/urdf.md.
TACHIKOMA_STAND_JOINT_POS = {
    "leg_fr_yaw": 0.3141592653589793,
    "leg_fr_pitch": -0.490371,
    "leg_fr_knee": 0.218241,
    "leg_fl_yaw": -0.3141592653589793,
    "leg_fl_pitch": -0.490371,
    "leg_fl_knee": 0.218241,
    "leg_rl_yaw": -0.03490658503988659,
    "leg_rl_pitch": -0.490371,
    "leg_rl_knee": 0.218241,
    "leg_rr_yaw": 0.03490658503988659,
    "leg_rr_pitch": -0.490371,
    "leg_rr_knee": 0.218241,
    "arm_r_yaw": 0.0,
    "arm_r_pitch": 0.0,
    "arm_r_elbow": 0.0,
    "arm_l_yaw": 0.0,
    "arm_l_pitch": 0.0,
    "arm_l_elbow": 0.0,
    "eye_r_roll": 0.0,
    "eye_l_roll": 0.0,
}


TACHIKOMA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Resolved lazily (after Kit starts) because the spawner pulls in pxr.
        func="tachikoma_lab.assets.spawn:spawn_tachikoma",
        usd_path=str(_TACHIKOMA_USD),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The URDF standing pose places base_link about 115 mm above the ground.
        pos=(0.0, 0.0, TACHIKOMA_STAND_HEIGHT),
        joint_pos=TACHIKOMA_STAND_JOINT_POS,
        joint_vel={name: 0.0 for name in TACHIKOMA_STAND_JOINT_POS},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[r"leg_.*_(yaw|pitch|knee)"],
            stiffness=20.0,
            damping=1.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[r"arm_.*_(yaw|pitch|elbow)"],
            stiffness=10.0,
            damping=0.5,
        ),
        "eyes": ImplicitActuatorCfg(
            joint_names_expr=[r"eye_.*_roll"],
            stiffness=5.0,
            damping=0.25,
        ),
    },
)
