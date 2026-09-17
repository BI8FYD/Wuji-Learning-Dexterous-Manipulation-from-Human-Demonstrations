"""Safety-critical contracts shared by simulation export and WujiHand playback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reference import JOINT_NAMES_20, SCHEMA_VERSION, ReferenceTrajectory, sha256_file


@dataclass(frozen=True)
class PolicyReferenceMetadata:
  """The reference-specific fields embedded beside an exported policy."""

  reference_sha256: str
  reference_schema_version: str
  reference_joint_names: tuple[str, ...]
  control_dt: float
  residual_scale: float
  ema_alpha: float

  @classmethod
  def from_mapping(cls, data: dict[str, Any]) -> "PolicyReferenceMetadata":
    required = {
      "reference_sha256", "reference_schema_version", "reference_joint_names",
      "control_dt", "residual_scale", "ema_alpha",
    }
    missing = required.difference(data)
    if missing:
      raise ValueError(f"policy metadata is missing fields: {sorted(missing)}")
    return cls(
      reference_sha256=str(data["reference_sha256"]),
      reference_schema_version=str(data["reference_schema_version"]),
      reference_joint_names=tuple(str(item) for item in data["reference_joint_names"]),
      control_dt=float(data["control_dt"]),
      residual_scale=float(data["residual_scale"]),
      ema_alpha=float(data["ema_alpha"]),
    )


def validate_policy_reference(
  metadata: PolicyReferenceMetadata, reference: ReferenceTrajectory,
) -> None:
  """Reject a policy/reference pair that could command a mismatched trajectory."""
  if metadata.reference_sha256 != sha256_file(reference.path):
    raise ValueError("reference trajectory does not match the exported policy")
  if metadata.reference_schema_version != SCHEMA_VERSION:
    raise ValueError("policy was exported for an unsupported reference schema")
  if metadata.reference_joint_names != JOINT_NAMES_20:
    raise ValueError("policy joint order does not match the WujiHand encoder order")
  if abs(metadata.control_dt - reference.dt) > 1.0e-6:
    raise ValueError("policy control period does not match the reference")
  if metadata.residual_scale <= 0.0 or not 0.0 < metadata.ema_alpha <= 1.0:
    raise ValueError("policy residual parameters are invalid")
