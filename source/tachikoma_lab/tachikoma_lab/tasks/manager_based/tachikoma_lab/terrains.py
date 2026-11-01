# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrains scaled for Tachikoma (body ~115 mm high, feet lift ~23-34 mm near the stance point).

Isaac Lab's stock rough terrain targets ANYmal-sized robots (5-23 cm steps, 10 cm heightfield cells), which is
far beyond what this robot can step over, so everything here is scaled down.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

# heightfield resolution fine enough for ~2 cm feet and centimetre-scale bumps
TACHIKOMA_HF_HORIZONTAL_SCALE = 0.025
TACHIKOMA_HF_VERTICAL_SCALE = 0.001


def make_tachikoma_terrain_importer(generator: TerrainGeneratorCfg, max_init_terrain_level: int | None = None):
    """Wrap a terrain generator in the importer that replaces the flat ground plane."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=max_init_terrain_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
    )


TACHIKOMA_TERRAINS_CFG = TerrainGeneratorCfg(
    # with curriculum=True the rows are difficulty levels: step heights / slopes grow from row 0 to row 9
    curriculum=True,
    size=(3.0, 3.0),
    border_width=5.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=TACHIKOMA_HF_HORIZONTAL_SCALE,
    vertical_scale=TACHIKOMA_HF_VERTICAL_SCALE,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "bumps": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30, noise_range=(0.002, 0.015), noise_step=0.002
        ),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.125, step_height_range=(0.005, 0.025), step_width=0.2, platform_width=1.0
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.125, step_height_range=(0.005, 0.025), step_width=0.2, platform_width=1.0
        ),
        "blocks": terrain_gen.MeshRandomGridTerrainCfg(
            # the grid must not tile the 3 m square exactly (it needs a border), hence 0.14 m and not 0.15 m
            proportion=0.15, grid_width=0.14, grid_height_range=(0.005, 0.02), platform_width=1.0
        ),
        # up to ~10 deg
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.18), platform_width=1.0, inverted=True
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.18), platform_width=1.0
        ),
    },
)
"""Training terrain for stage 2: flat, bumps, small stairs, blocks and gentle slopes, easy to hard."""


def make_test_course(kind: str, height: float) -> TerrainGeneratorCfg:
    """Single-difficulty course used by the evaluation script.

    Args:
        kind: ``"step"`` (stairs rising outward from the start platform, so walking in any direction climbs)
            or ``"rough"`` (random bumps up to ``height``).
        height: Step height or maximum bump height [m].
    """
    if kind == "step":
        sub_terrain = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            # the start platform must hold the whole robot (feet reach ~0.16 m from the base) plus the
            # +-0.25 m reset offset, otherwise feet spawn on the first step edge
            proportion=1.0, step_height_range=(height, height), step_width=0.2, platform_width=1.0
        )
    elif kind == "rough":
        sub_terrain = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, height), noise_step=0.002
        )
    else:
        raise ValueError(f"Unknown test course: {kind}")
    return TerrainGeneratorCfg(
        size=(3.0, 3.0),
        border_width=1.0,
        num_rows=4,
        num_cols=4,
        horizontal_scale=TACHIKOMA_HF_HORIZONTAL_SCALE,
        vertical_scale=TACHIKOMA_HF_VERTICAL_SCALE,
        slope_threshold=None,
        use_cache=False,
        sub_terrains={kind: sub_terrain},
    )
