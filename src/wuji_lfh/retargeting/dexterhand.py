"""DexterHand MANO parameters to IK-friendly MediaPipe-style landmarks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for _name, _type in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _type)

import smplx
from smplx.lbs import batch_rodrigues
from smplx.vertex_ids import vertex_ids

from .fingertip_axis import TIP_AXIS_RATIO, virtual_tip_points

# MANO: wrist, index(3), middle(3), pinky(3), ring(3), thumb(3).
# MediaPipe: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4).
MANO_CHAINS = {
    "index": (1, 2, 3), "middle": (4, 5, 6), "pinky": (7, 8, 9),
    "ring": (10, 11, 12), "thumb": (13, 14, 15),
}
MEDIAPIPE_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FOUR_FINGERS = ("index", "middle", "ring", "pinky")


def mirror_left_mano_landmarks_to_right(landmarks: np.ndarray) -> np.ndarray:
    """Mirror DexterHand's left-MANO geometry into right-MANO coordinates."""
    mirrored = np.asarray(landmarks, dtype=np.float32).copy()
    if mirrored.shape[-2:] != (21, 3):
        raise ValueError(f"expected landmarks [...,21,3], got {mirrored.shape}")
    mirrored[..., 0] *= -1.0
    return mirrored


@dataclass(frozen=True)
class FingerCalibration:
    """Shape-specific, fixed geometry for one canonical finger chain."""

    direction: np.ndarray
    flexion_axis: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True)
class RetargetSkeletonCalibration:
    """Fixed canonical geometry shared by every frame in a demonstration."""

    betas: np.ndarray
    fingers: dict[str, FingerCalibration]
    thumb_tip_length: float


def load_dexterhand_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load and validate the official DexterHand NPZ schema."""
    with np.load(path, allow_pickle=True) as archive:
        required = {"metadata", "hand_poses", "hand_shapes"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"DexterHand archive is missing keys: {sorted(missing)}")
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        metadata = archive["metadata"].item()
    poses, shapes = arrays["hand_poses"], arrays["hand_shapes"]
    if poses.ndim != 2 or poses.shape[1] != 45:
        raise ValueError(f"expected hand_poses [T, 45], got {poses.shape}")
    if shapes.shape != (poses.shape[0], 10):
        raise ValueError(f"expected hand_shapes [T, 10], got {shapes.shape}")
    if not np.isfinite(poses).all() or not np.isfinite(shapes).all():
        raise ValueError("MANO pose/shape contains NaN or infinite values")
    return arrays, metadata


def resolve_mano_model_path(model_path: Path, handedness: str) -> Path:
    """Resolve one handed MANO pickle from direct and legacy asset layouts.

    The formal callers consume the returned pickle path, never an SMPL-X asset
    root.  A directory containing the pickle is the canonical project layout;
    the ``mano/`` child lookup remains for existing SMPL-X asset roots.
    """
    hand = str(handedness).lower()
    if hand not in {"left", "right"}:
        raise ValueError(f"MANO handedness must be 'left' or 'right', got {handedness!r}")
    expected = f"MANO_{hand.upper()}.pkl"
    supplied = Path(model_path).expanduser().resolve()
    if supplied.is_file():
        if supplied.name != expected:
            raise ValueError(f"expected {expected} for {hand} hand, got {supplied.name}")
        return supplied
    candidates = (supplied / expected, supplied / "mano" / expected)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"MANO {hand} model not found; tried: {attempted}")


def _create_mano(model_path: Path, batch_size: int, is_rhand: bool = False):
    return smplx.create(
        model_path=str(model_path), model_type="mano", flat_hand_mean=True,
        is_rhand=is_rhand, use_pca=False, batch_size=batch_size,
    )


def _forward_mano(model_path, hand_pose, betas, global_orient, transl, is_rhand=False):
    model = _create_mano(model_path, len(hand_pose), is_rhand=is_rhand)
    params = {
        "global_orient": torch.as_tensor(global_orient, dtype=torch.float32),
        "transl": torch.as_tensor(transl, dtype=torch.float32),
        "hand_pose": torch.as_tensor(hand_pose, dtype=torch.float32),
        "betas": torch.as_tensor(np.array(betas, dtype=np.float32, copy=True)),
    }
    with torch.inference_mode():
        result = model(**params)
    return model, result


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"degenerate vector while calibrating {label}")
    return vector / norm


def calibrate_retarget_skeleton(
    hand_shapes: np.ndarray, mano_model_path: Path, *, is_rhand: bool = False,
) -> RetargetSkeletonCalibration:
    """Calibrate fixed canonical geometry from a demonstration's hand shape."""
    shapes = np.asarray(hand_shapes, dtype=np.float32)
    if shapes.ndim != 2 or shapes.shape[1] != 10 or not len(shapes):
        raise ValueError(f"expected hand_shapes [T,10], got {shapes.shape}")
    betas = np.median(shapes, axis=0).astype(np.float32)
    zeros3 = np.zeros((1, 3), dtype=np.float32)
    _, neutral = _forward_mano(
        mano_model_path, np.zeros((1, 45), dtype=np.float32), betas[None],
        zeros3, zeros3, is_rhand=is_rhand,
    )
    joints = neutral.joints[0].detach().cpu().numpy()
    vertices = neutral.vertices[0].detach().cpu().numpy()
    palm_normal = _normalize(
        np.cross(joints[1] - joints[0], joints[7] - joints[0]), "palm normal"
    )
    fingers = {}
    tips = vertex_ids["mano"]
    for finger in FOUR_FINGERS:
        mcp, pip, dip = MANO_CHAINS[finger]
        # The end-to-end direction suppresses the joint regressor's zig-zag.
        direction = _normalize(joints[dip] - joints[mcp], f"{finger} direction")
        # The geometric cross product is positive for flexion in MANO's left-
        # hand convention. Right MANO uses the opposite rotation-vector sign,
        # so make the calibrated axis itself handedness-aware before projection.
        flexion_axis = _normalize(
            np.cross(palm_normal, direction), f"{finger} flexion axis"
        )
        if is_rhand:
            flexion_axis = -flexion_axis
        lengths = np.asarray([
            np.linalg.norm(joints[pip] - joints[mcp]),
            np.linalg.norm(joints[dip] - joints[pip]),
            np.linalg.norm(vertices[tips[finger]] - joints[dip]),
        ], dtype=np.float32)
        if np.any(lengths < 1e-6):
            raise ValueError(f"degenerate neutral bone length for {finger}: {lengths}")
        fingers[finger] = FingerCalibration(
            direction.astype(np.float32), flexion_axis.astype(np.float32), lengths,
        )
    thumb_tip_length = float(
        np.linalg.norm(vertices[tips["thumb"]] - joints[MANO_CHAINS["thumb"][-1]])
    )
    return RetargetSkeletonCalibration(betas, fingers, thumb_tip_length)


def _axis_angle_matrices(rotvecs: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(rotvecs.reshape(-1, 3), dtype=torch.float32)
    with torch.inference_mode():
        matrices = batch_rodrigues(tensor)
    return matrices.reshape(*rotvecs.shape[:-1], 3, 3).cpu().numpy()


def _projected_flexion_matrices(rotvecs, axis):
    # MANO local frames share wrist axes at neutral. Projection gives one stable
    # anatomical DOF; clamping removes non-anatomical PIP/DIP hyper-extension.
    angles = np.maximum(np.einsum("ti,i->t", rotvecs, axis), 0.0)
    return _axis_angle_matrices(angles[:, None] * axis[None]), angles


def _build_raw_landmarks(joints, vertices):
    output = np.empty((len(joints), 21, 3), dtype=np.float32)
    output[:, 0] = joints[:, 0]
    tips = vertex_ids["mano"]
    for finger_index, finger in enumerate(MEDIAPIPE_FINGERS):
        dst = 1 + 4 * finger_index
        output[:, dst:dst + 3] = joints[:, MANO_CHAINS[finger]]
        output[:, dst + 3] = vertices[:, tips[finger]]
    return output


def _build_retarget_landmarks(
    hand_poses, joints, raw, calibration, global_orient=None,
):
    output = raw.copy()
    root_rot = (None if global_orient is None
                else _axis_angle_matrices(global_orient))
    # Thumb v1 retains MANO CMC/MCP/IP and only replaces the surface fingertip.
    thumb_direction = output[:, 3] - output[:, 2]
    thumb_direction /= np.linalg.norm(thumb_direction, axis=1, keepdims=True).clip(min=1e-8)
    output[:, 4] = output[:, 3] + thumb_direction * calibration.thumb_tip_length

    for finger in FOUR_FINGERS:
        mcp_joint, pip_joint, dip_joint = MANO_CHAINS[finger]
        dst = 1 + 4 * MEDIAPIPE_FINGERS.index(finger)
        geometry = calibration.fingers[finger]
        mcp_rot = _axis_angle_matrices(
            hand_poses[:, (mcp_joint - 1) * 3:mcp_joint * 3]
        )
        if root_rot is not None:
            mcp_rot = np.matmul(root_rot, mcp_rot)
        pip_rot, _ = _projected_flexion_matrices(
            hand_poses[:, (pip_joint - 1) * 3:pip_joint * 3], geometry.flexion_axis,
        )
        dip_rot, _ = _projected_flexion_matrices(
            hand_poses[:, (dip_joint - 1) * 3:dip_joint * 3], geometry.flexion_axis,
        )
        pip_global = np.matmul(mcp_rot, pip_rot)
        dip_global = np.matmul(pip_global, dip_rot)
        proximal = np.einsum("tij,j->ti", mcp_rot, geometry.direction)
        middle = np.einsum("tij,j->ti", pip_global, geometry.direction)
        distal = np.einsum("tij,j->ti", dip_global, geometry.direction)
        output[:, dst] = joints[:, mcp_joint]
        output[:, dst + 1] = output[:, dst] + proximal * geometry.lengths[0]
        output[:, dst + 2] = output[:, dst + 1] + middle * geometry.lengths[1]
        output[:, dst + 3] = output[:, dst + 2] + distal * geometry.lengths[2]
    return output.astype(np.float32, copy=False)


def mano_to_mediapipe_keypoints(
    hand_poses: np.ndarray, hand_shapes: np.ndarray, mano_model_path: Path, *,
    batch_size: int = 512, global_orient: np.ndarray | None = None,
    transl: np.ndarray | None = None, is_rhand: bool = False,
) -> np.ndarray:
    """Decode MANO into an IK-friendly MediaPipe-ordered ``[T,21,3]`` skeleton."""
    model_path = Path(mano_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"MANO model not found: {model_path}")
    poses, shapes = map(lambda value: np.asarray(value, dtype=np.float32),
                        (hand_poses, hand_shapes))
    if poses.ndim != 2 or poses.shape[1] != 45:
        raise ValueError(f"expected hand_poses [T,45], got {poses.shape}")
    if shapes.shape != (len(poses), 10):
        raise ValueError(f"expected hand_shapes [T,10], got {shapes.shape}")
    if not len(poses):
        raise ValueError("cannot decode an empty trajectory")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(poses).all() or not np.isfinite(shapes).all():
        raise ValueError("MANO pose/shape contains NaN or infinite values")
    count = len(poses)
    orient = (np.zeros((count, 3), np.float32) if global_orient is None
              else np.asarray(global_orient, np.float32))
    translation = (np.zeros((count, 3), np.float32) if transl is None
                   else np.asarray(transl, np.float32))
    if orient.shape != (count, 3) or translation.shape != (count, 3):
        raise ValueError("global_orient and transl must both have shape [T,3]")

    calibration = calibrate_retarget_skeleton(
        shapes, model_path, is_rhand=is_rhand
    )
    fixed_shapes = np.broadcast_to(calibration.betas, shapes.shape)
    output = np.empty((count, 21, 3), dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        _, result = _forward_mano(
            model_path, poses[start:stop], fixed_shapes[start:stop],
            orient[start:stop], translation[start:stop], is_rhand=is_rhand,
        )
        joints = result.joints.detach().cpu().numpy()
        vertices = result.vertices.detach().cpu().numpy()
        raw = _build_raw_landmarks(joints, vertices)
        output[start:stop] = _build_retarget_landmarks(
            poses[start:stop], joints, raw, calibration, orient[start:stop],
        )
    if not np.isfinite(output).all():
        raise ValueError("MANO decoding produced NaN or infinite landmarks")
    return output


def mano_to_wuji_keypoints(
    hand_poses: np.ndarray, hand_shapes: np.ndarray, mano_model_path: Path, *,
    batch_size: int = 512, global_orient: np.ndarray | None = None,
    transl: np.ndarray | None = None, is_rhand: bool = False,
    return_position_tips: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decode MANO as native 21-point geometry for the Wuji adaptive solver.

    Unlike :func:`mano_to_mediapipe_keypoints`, this deliberately does *not*
    reconstruct four fingers from projected flexion or synthesize virtual tips.
    It retains the fixed-median MANO shape policy, but uses the posed MANO joint
    regressor and the official fixed MANO tip vertices directly.  When requested,
    a separate distal-axis position target is returned without changing the
    21-point input used for directions and pinch.  The resulting
    topology is MediaPipe-compatible (wrist, then thumb..pinky) only as an API
    convention; it has not received MediaPipe/video normalization or correction.
    """
    model_path = Path(mano_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"MANO model not found: {model_path}")
    poses, shapes = map(
        lambda value: np.asarray(value, dtype=np.float32), (hand_poses, hand_shapes)
    )
    if poses.ndim != 2 or poses.shape[1] != 45:
        raise ValueError(f"expected hand_poses [T,45], got {poses.shape}")
    if shapes.shape != (len(poses), 10):
        raise ValueError(f"expected hand_shapes [T,10], got {shapes.shape}")
    if not len(poses):
        raise ValueError("cannot decode an empty trajectory")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(poses).all() or not np.isfinite(shapes).all():
        raise ValueError("MANO pose/shape contains NaN or infinite values")

    count = len(poses)
    orient = (np.zeros((count, 3), np.float32) if global_orient is None
              else np.asarray(global_orient, np.float32))
    translation = (np.zeros((count, 3), np.float32) if transl is None
                   else np.asarray(transl, np.float32))
    if orient.shape != (count, 3) or translation.shape != (count, 3):
        raise ValueError("global_orient and transl must both have shape [T,3]")

    # DexterHand shape estimates may fluctuate frame-to-frame.  Retargeting
    # geometry must not, so use the trajectory median exactly as the legacy
    # decoder does.
    fixed_betas = np.broadcast_to(np.median(shapes, axis=0).astype(np.float32), shapes.shape)
    output = np.empty((count, 21, 3), dtype=np.float32)
    position_tips = np.empty((count, 5, 3), dtype=np.float32) if return_position_tips else None
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        model, result = _forward_mano(
            model_path, poses[start:stop], fixed_betas[start:stop],
            orient[start:stop], translation[start:stop], is_rhand=is_rhand,
        )
        joints = result.joints.detach().cpu().numpy()
        vertices = result.vertices.detach().cpu().numpy()
        output[start:stop] = _build_raw_landmarks(joints, vertices)
        if position_tips is not None:
            for local_index in range(stop - start):
                position_tips[start + local_index] = virtual_tip_points(
                    vertices[local_index], np.asarray(model.faces),
                    output[start + local_index], TIP_AXIS_RATIO,
                )[0]
    if not np.isfinite(output).all():
        raise ValueError("MANO decoding produced NaN or infinite landmarks")
    if position_tips is not None:
        return output, position_tips
    return output


def mano_neutral_wuji_keypoints(
    hand_shapes: np.ndarray, mano_model_path: Path, *, is_rhand: bool = False,
    return_position_tips: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed-beta neutral MANO topology and its resolved beta vector.

    This is calibration geometry only.  It is intentionally separate from a
    posed trajectory so morphology ratios never depend on a grasp frame.
    """
    shapes = np.asarray(hand_shapes, dtype=np.float32)
    if shapes.ndim != 2 or shapes.shape[1] != 10 or not len(shapes):
        raise ValueError(f"expected hand_shapes [T,10], got {shapes.shape}")
    if not np.isfinite(shapes).all():
        raise ValueError("MANO shape contains NaN or infinite values")
    betas = np.median(shapes, axis=0).astype(np.float32)
    zeros = np.zeros((1, 3), dtype=np.float32)
    model, result = _forward_mano(
        Path(mano_model_path), np.zeros((1, 45), dtype=np.float32), betas[None],
        zeros, zeros, is_rhand=is_rhand,
    )
    joints = result.joints.detach().cpu().numpy()
    vertices = result.vertices.detach().cpu().numpy()
    landmarks = _build_raw_landmarks(joints, vertices)[0]
    if return_position_tips:
        position_tips = virtual_tip_points(vertices[0], np.asarray(model.faces), landmarks)[0]
        return landmarks, betas, position_tips.astype(np.float32)
    return landmarks, betas


def decode_mano_frame_for_visualization(
    hand_pose, hand_shape, mano_model_path, *, is_rhand: bool = False,
):
    """Return mesh, landmarks, and native ``[V,16]`` MANO LBS weights for one frame."""
    pose = np.asarray(hand_pose, np.float32).reshape(1, 45)
    shape = np.asarray(hand_shape, np.float32).reshape(1, 10)
    zeros = np.zeros((1, 3), np.float32)
    model, result = _forward_mano(
        mano_model_path, pose, shape, zeros, zeros, is_rhand=is_rhand
    )
    joints = result.joints.detach().cpu().numpy()
    vertices = result.vertices.detach().cpu().numpy()
    raw = _build_raw_landmarks(joints, vertices)
    calibration = calibrate_retarget_skeleton(
        shape, mano_model_path, is_rhand=is_rhand
    )
    retarget = _build_retarget_landmarks(pose, joints, raw, calibration)
    weights = model.lbs_weights.detach().cpu().numpy()
    return vertices[0], np.asarray(model.faces), raw[0], retarget[0], weights
