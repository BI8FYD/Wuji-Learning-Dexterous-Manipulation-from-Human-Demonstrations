"""The bundled IK model must not need an external retargeting checkout."""

from pathlib import Path
from xml.etree import ElementTree

import pytest

VENDOR = (
  Path(__file__).resolve().parents[1]
  / "src/wuji_lfh/_vendor/wuji_retargeting"
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_bundled_kinematic_model(side):
  urdf = VENDOR / f"wuji-description/hand/body/urdf/{side}.urdf"
  model = ElementTree.parse(urdf).getroot()
  movable = [joint for joint in model.findall("joint") if joint.get("type") != "fixed"]
  assert len(movable) == 20
  assert not model.findall(".//mesh")
  assert (VENDOR / "LICENSE").is_file()
  assert (VENDOR / "wuji-description/LICENSE").is_file()


def test_robot_wrapper_uses_bundled_model():
  pytest.importorskip("pinocchio")
  pytest.importorskip("nlopt")
  from wuji_lfh._vendor.wuji_retargeting.robot import RobotWrapper

  urdf = VENDOR / "wuji-description/hand/body/urdf/right.urdf"
  robot = RobotWrapper(str(urdf), hand_side="right")
  assert robot.model.nq == 20
  assert robot.joint_limits.shape == (20, 2)
  assert RobotWrapper.__module__.startswith("wuji_lfh._vendor.")
