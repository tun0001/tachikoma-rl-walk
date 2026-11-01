"""Spawner for the Tachikoma USD.

This module imports pxr, so it must only be imported after Kit has started. TACHIKOMA_CFG references
it by string for that reason.
"""

from __future__ import annotations

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim


@clone
def spawn_tachikoma(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the USD and enable contact reporting on every link.

    The converted USD nests the links (base_link/leg_*_coxa/leg_*_femur/leg_*_tibia), but Isaac Lab's
    ``activate_contact_sensors`` stops at the first rigid body, so only base_link would report contacts.
    This runs on the source env before cloning, so every env inherits the schemas.
    """
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    for child in Usd.PrimRange(prim):
        if not child.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        applied = child.GetAppliedSchemas()
        if "PhysxRigidBodyAPI" not in applied:
            child.AddAppliedSchema("PhysxRigidBodyAPI")
        safe_set_attribute_on_usd_prim(child, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
        if "PhysxContactReportAPI" not in applied:
            child.AddAppliedSchema("PhysxContactReportAPI")
        safe_set_attribute_on_usd_prim(child, "physxContactReport:threshold", 0.0, camel_case=False)
    return prim
