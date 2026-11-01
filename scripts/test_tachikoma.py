from pathlib import Path

from tachikoma_lab.assets.tachikoma import (
    TACHIKOMA_ACTUATED_JOINT_NAMES,
    TACHIKOMA_BODY_LINK_NAME,
    TACHIKOMA_CFG,
    TACHIKOMA_FOOT_LINK_NAMES,
    TACHIKOMA_JOINT_LIMITS,
    TACHIKOMA_LEG_JOINT_NAMES,
    TACHIKOMA_STAND_JOINT_POS,
)


def test_tachikoma_asset_exists() -> None:
    usd_path = Path(TACHIKOMA_CFG.spawn.usd_path)
    assert usd_path.name == "tachikoma.usda"
    assert usd_path.is_file()


def test_tachikoma_leg_configuration() -> None:
    assert len(TACHIKOMA_LEG_JOINT_NAMES) == 12
    assert TACHIKOMA_ACTUATED_JOINT_NAMES == TACHIKOMA_LEG_JOINT_NAMES
    assert len(set(TACHIKOMA_LEG_JOINT_NAMES)) == 12


def test_tachikoma_links() -> None:
    assert TACHIKOMA_BODY_LINK_NAME == "base_link"
    assert TACHIKOMA_FOOT_LINK_NAMES == [
        "leg_fr_tibia",
        "leg_fl_tibia",
        "leg_rl_tibia",
        "leg_rr_tibia",
    ]


def test_tachikoma_joint_limits_and_stand_pose() -> None:
    assert set(TACHIKOMA_JOINT_LIMITS) == set(TACHIKOMA_STAND_JOINT_POS)
    assert set(TACHIKOMA_LEG_JOINT_NAMES) <= set(TACHIKOMA_JOINT_LIMITS)
    assert all(lower < upper for lower, upper in TACHIKOMA_JOINT_LIMITS.values())
    assert all(
        lower <= TACHIKOMA_STAND_JOINT_POS[name] <= upper
        for name, (lower, upper) in TACHIKOMA_JOINT_LIMITS.items()
    )


def main() -> None:
    tests = (
        test_tachikoma_asset_exists,
        test_tachikoma_leg_configuration,
        test_tachikoma_links,
        test_tachikoma_joint_limits_and_stand_pose,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"Tachikoma checks passed: {len(tests)}")


if __name__ == "__main__":
    main()
