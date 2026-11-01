# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass

from tachikoma_lab.assets.tachikoma import (
    TACHIKOMA_ACTUATED_JOINT_NAMES,
    TACHIKOMA_CFG,
    TACHIKOMA_BODY_LINK_NAME,
    TACHIKOMA_FOOT_LINK_NAMES,
    TACHIKOMA_STAND_HEIGHT,
    TACHIKOMA_STAND_JOINT_POS,
)

from . import mdp
from .terrains import TACHIKOMA_TERRAINS_CFG, make_tachikoma_terrain_importer


@configclass
class TachikomaLabSceneCfg(InteractiveSceneCfg):
    """Configuration for the Tachikoma scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = TACHIKOMA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact data for fall detection. The links are nested in the USD, so "Robot/.*" only resolves base_link.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
    )
    # Contact data for the four feet (tibia links), used for the stepping reward. The sensor searches all
    # descendants of the parent path for the leaf name, so the nested tibias are found from base_link.
    foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/leg_.*_tibia",
        history_length=3,
        track_air_time=True,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # The real robot uses hobby position servos, so the policy outputs joint position offsets
    # around the standing pose (target = stand_pose + 0.25 * action).
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=TACHIKOMA_ACTUATED_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """Velocity commands that the policy must learn to follow."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.1,
        heading_command=False,
        # The debug arrows assume +X forward, which is Tachikoma's right side.
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # Commands are (forward, left, yaw) in Tachikoma's frame; see mdp.tachikoma_lin_vel.
            # Speeds are sized for a ~2.9 kg robot with 1.96 N·m hobby servos.
            lin_vel_x=(0.0, 0.3),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.5, 0.5),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.tachikoma_base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset the actuated leg joints with small noise around the configured initial pose
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=TACHIKOMA_ACTUATED_JOINT_NAMES),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.05, 0.05),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_tachikoma_lin_vel_xy_exp,
        weight=2.0,
        # Narrow enough that standing still under a 0.3 m/s command earns almost nothing.
        params={"command_name": "base_velocity", "std": 0.1},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("foot_contact", body_names=TACHIKOMA_FOOT_LINK_NAMES),
            "threshold": 0.15,
        },
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TACHIKOMA_ACTUATED_JOINT_NAMES)},
    )
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": TACHIKOMA_STAND_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TACHIKOMA_ACTUATED_JOINT_NAMES)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.000001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TACHIKOMA_ACTUATED_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Stop an episode when the robot falls. Joint limits are a penalty, not a termination,
    # because position targets near the limits would otherwise end episodes on tiny overshoots.
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    low_base = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.04, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=TACHIKOMA_BODY_LINK_NAME),
            "threshold": 1.0,
        },
    )


@configclass
class TachikomaLabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TachikomaLabSceneCfg = TachikomaLabSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        # follow env 0's robot from its front-right (the robot is only ~30 cm long)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.7, 0.7, 0.45)
        self.viewer.lookat = (0.0, 0.0, 0.05)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.foot_contact.update_period = self.sim.dt


##
# Safe (slow, statically stable) walking variant.
##

# Foot tip in the tibia link frame [m], from the lowest point of leg_*_tibia__col_0.stl.
TACHIKOMA_FOOT_TIP_OFFSET = (0.0, 0.0, -0.15)


@configclass
class SafeCommandsCfg(CommandsCfg):
    """Slower commands: safety matters more than speed."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.1,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.1),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(-0.3, 0.3),
        ),
    )


@configclass
class SafeRewardsCfg(RewardsCfg):
    """Rewards for a slow crawl gait: long deliberate steps, three feet on the ground, no foot sliding.

    The smoothness/safety penalties are large while the policy is still jittery, so staying alive must
    clearly outweigh them; otherwise the policy learns to end episodes by dropping its body (seen in the
    first attempt: episodes shrank to ~11 steps).
    """

    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-20.0)

    track_lin_vel_xy = RewTerm(
        func=mdp.track_tachikoma_lin_vel_xy_exp,
        weight=2.0,
        # scaled down with the slower commands
        params={"command_name": "base_velocity", "std": 0.05},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("foot_contact", body_names=TACHIKOMA_FOOT_LINK_NAMES),
            # steps shorter than this are penalized, so quick shuffling no longer pays
            "threshold": 0.3,
            "min_command": 0.02,
        },
    )
    feet_support = RewTerm(
        func=mdp.feet_support_below,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("foot_contact", body_names=TACHIKOMA_FOOT_LINK_NAMES),
            "min_feet": 3,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_tip_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "foot_contact", body_names=TACHIKOMA_FOOT_LINK_NAMES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=TACHIKOMA_FOOT_LINK_NAMES, preserve_order=True),
            "tip_offset": TACHIKOMA_FOOT_TIP_OFFSET,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)


@configclass
class TachikomaWalkSafeEnvCfg(TachikomaLabEnvCfg):
    commands: SafeCommandsCfg = SafeCommandsCfg()
    rewards: SafeRewardsCfg = SafeRewardsCfg()


##
# Stable walking variant: the safe variant plus a balanced stance and a CoM-in-support-polygon reward.
##

# Feet in clockwise order seen from above (x = right, y = forward), as support_margin expects.
TACHIKOMA_FOOT_LINK_NAMES_CYCLIC = ["leg_fr_tibia", "leg_rr_tibia", "leg_rl_tibia", "leg_fl_tibia"]

# The whole-body CoM sits ~34 mm behind the centre of the feet (heavy rear pod), so with the URDF stance the
# robot tips backward whenever a rear foot is lifted (margin -29 mm) while front lifts have +29 mm to spare.
# Swinging every foot 15 deg toward the rear with the yaw joints balances the two (margin ~ +-1 mm), so the
# policy only needs a small body sway, like the SWAY of the real firmware's gait, before lifting any foot.
TACHIKOMA_STABLE_STANCE_YAW_SHIFT = math.radians(15.0)
TACHIKOMA_STABLE_STAND_JOINT_POS = {
    **TACHIKOMA_STAND_JOINT_POS,
    "leg_fr_yaw": TACHIKOMA_STAND_JOINT_POS["leg_fr_yaw"] - TACHIKOMA_STABLE_STANCE_YAW_SHIFT,
    "leg_rr_yaw": TACHIKOMA_STAND_JOINT_POS["leg_rr_yaw"] - TACHIKOMA_STABLE_STANCE_YAW_SHIFT,
    "leg_fl_yaw": TACHIKOMA_STAND_JOINT_POS["leg_fl_yaw"] + TACHIKOMA_STABLE_STANCE_YAW_SHIFT,
    "leg_rl_yaw": TACHIKOMA_STAND_JOINT_POS["leg_rl_yaw"] + TACHIKOMA_STABLE_STANCE_YAW_SHIFT,
}
# base_link height of the URDF standing pose (assets/tachikoma/urdf.md: 115 mm).
TACHIKOMA_STABLE_BASE_HEIGHT = 0.115


@configclass
class StableRewardsCfg(SafeRewardsCfg):
    """Safe rewards plus a static-stability term; the height target matches the real standing pose."""

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": TACHIKOMA_STABLE_BASE_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
    )
    support_margin = RewTerm(
        func=mdp.support_margin_below,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "foot_contact", body_names=TACHIKOMA_FOOT_LINK_NAMES_CYCLIC, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=TACHIKOMA_FOOT_LINK_NAMES_CYCLIC, preserve_order=True),
            "tip_offset": TACHIKOMA_FOOT_TIP_OFFSET,
            # keep the CoM at least 1 cm inside the polygon of the feet on the ground
            "target": 0.01,
        },
    )


@configclass
class TachikomaWalkStableEnvCfg(TachikomaWalkSafeEnvCfg):
    rewards: StableRewardsCfg = StableRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # balanced stance; the position action is an offset around these default joint positions
        self.scene.robot.init_state.joint_pos = TACHIKOMA_STABLE_STAND_JOINT_POS


##
# Foot-lift variant (stage 1 of the terrain plan): the stable variant plus a swing-foot clearance reward.
##

# stable_v1 slid its feet (mean lift 0.4 mm on flat ground); the legs can lift the tip ~23-34 mm near the
# stance point, so aim for 25 mm.
TACHIKOMA_FOOT_CLEARANCE_TARGET = 0.025


@configclass
class LiftRewardsCfg(StableRewardsCfg):
    foot_clearance = RewTerm(
        func=mdp.FootClearance,
        # kept well below alive (+2.0) so that giving up never pays (see SafeRewardsCfg)
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("foot_contact", body_names=TACHIKOMA_FOOT_LINK_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=TACHIKOMA_FOOT_LINK_NAMES, preserve_order=True),
            "tip_offset": TACHIKOMA_FOOT_TIP_OFFSET,
            "target_height": TACHIKOMA_FOOT_CLEARANCE_TARGET,
        },
    )


@configclass
class TachikomaWalkLiftEnvCfg(TachikomaWalkStableEnvCfg):
    rewards: LiftRewardsCfg = LiftRewardsCfg()


##
# Terrain variant (stage 2): the foot-lift variant trained on small bumps, stairs, blocks and slopes.
##


@configclass
class TerrainSceneCfg(TachikomaLabSceneCfg):
    """Rough terrain instead of the flat plane, plus a downward height scanner under the body.

    The scanner only feeds the height reward/termination; the policy stays blind, like the real robot.
    """

    ground = None
    terrain = make_tachikoma_terrain_importer(TACHIKOMA_TERRAINS_CFG, max_init_terrain_level=2)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        # 5 x 5 rays over the 20 cm square under the body
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class TerrainRewardsCfg(LiftRewardsCfg):
    # body height measured from the ground right below, not from z = 0
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={
            "target_height": TACHIKOMA_STABLE_BASE_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class TerrainTerminationsCfg(TerminationsCfg):
    low_base = DoneTerm(
        func=mdp.base_height_above_ground_below,
        params={"minimum_height": 0.04, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class TerrainCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_tachikoma)


@configclass
class TachikomaWalkTerrainEnvCfg(TachikomaWalkLiftEnvCfg):
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=4.0)
    rewards: TerrainRewardsCfg = TerrainRewardsCfg()
    terminations: TerrainTerminationsCfg = TerrainTerminationsCfg()
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # the terrain's friction/restitution apply to every contact with the ground
        self.sim.physics_material = self.scene.terrain.physics_material


##
# Terrain variant v2: terrain_v1 survived every course but drifted ~10 deg/s clockwise and drifted down to the
# easy rows of the curriculum (walking slowly on hard terrain got it demoted to easier, better-paying terrain).
##


@configclass
class TerrainV2RewardsCfg(TerrainRewardsCfg):
    # 0.5 -> 1.5: linear tracking is judged in the body frame, so a turning robot still earned full tracking
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


@configclass
class TerrainV2CurriculumCfg:
    # demote on falls, and on distance only when the robot barely moved (was: < 40 % of the commanded distance)
    terrain_levels = CurrTerm(func=mdp.terrain_levels_tachikoma, params={"demote_ratio": 0.15, "demote_on_fall": True})


@configclass
class TachikomaWalkTerrainV2EnvCfg(TachikomaWalkTerrainEnvCfg):
    rewards: TerrainV2RewardsCfg = TerrainV2RewardsCfg()
    curriculum: TerrainV2CurriculumCfg = TerrainV2CurriculumCfg()


##
# Terrain variant v3: terrain_v2 still turned ~10 deg/s while walking (and more the faster it walked), and its
# curriculum level (3.85) did not show on the test courses.
##


@configclass
class TerrainV3RewardsCfg(TerrainV2RewardsCfg):
    # std 0.5 -> 0.15 rad/s: with 0.5 a 0.17 rad/s (10 deg/s) turn still kept ~90 % of the reward, so tripling
    # the weight in v2 only tripled a 10 % difference; with 0.15 the same turn keeps ~28 %
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.15},
    )


@configclass
class TerrainV3CurriculumCfg:
    # judged by velocity tracking over the whole episode instead of the distance walked
    terrain_levels = CurrTerm(func=mdp.terrain_levels_tracking)


@configclass
class TachikomaWalkTerrainV3EnvCfg(TachikomaWalkTerrainEnvCfg):
    rewards: TerrainV3RewardsCfg = TerrainV3RewardsCfg()
    curriculum: TerrainV3CurriculumCfg = TerrainV3CurriculumCfg()


##
# Terrain variant v4: v3 still turned right while walking and hardly reacted to yaw-rate commands (an outside
# heading-hold controller saturated at +0.3 rad/s and the robot kept turning). stable_v1 does not have this habit,
# so v4 restarts from it with v3's rewards and curriculum plus heading commands.
##


@configclass
class TerrainV4CommandsCfg(SafeCommandsCfg):
    # Half of the robots get a heading target instead of a yaw rate: their yaw-rate command becomes
    # stiffness * heading error, so any drift turns into a command to turn back, and following it pays.
    # (heading_w is measured along body +X, Tachikoma's right side; the constant offset does not matter
    # because the targets are uniform over the full circle.)
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.1,
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_heading_envs=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.1),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class TachikomaWalkTerrainV4EnvCfg(TachikomaWalkTerrainV3EnvCfg):
    commands: TerrainV4CommandsCfg = TerrainV4CommandsCfg()
