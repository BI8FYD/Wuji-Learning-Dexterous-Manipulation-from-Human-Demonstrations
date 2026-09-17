"""MANO fingertip position targets on the distal skeletal axes."""

from __future__ import annotations

import numpy as np

TIP_AXIS_RATIO = 0.90
PRECEDING_INDICES = (2, 6, 10, 14, 18)
LAST_JOINT_INDICES = (3, 7, 11, 15, 19)


def first_ray_mesh_exit(origin: np.ndarray, direction: np.ndarray,
                        vertices: np.ndarray, faces: np.ndarray) -> float:
    """Nearest positive, two-sided ray/triangle hit, measured in metres."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if origin.shape != (3,) or direction.shape != (3,):
        raise ValueError("ray origin and direction must be [3]")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh vertices/faces must be [V,3]/[F,3]")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-10:
        raise ValueError("distal bone-axis direction is degenerate")
    ray = direction / norm
    tri = vertices[faces]
    edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    pvec = np.cross(np.broadcast_to(ray, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(det) > 1e-12
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = origin - tri[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = qvec @ ray * inv_det
    distance = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    hits = valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (distance > 1e-6)
    if not np.any(hits):
        raise ValueError("distal bone-axis ray does not exit the MANO mesh")
    return float(np.min(distance[hits]))


def virtual_tip_points(vertices: np.ndarray, faces: np.ndarray,
                       landmarks: np.ndarray, ratio: float = TIP_AXIS_RATIO
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return five internal points, surface hits, and joint-to-surface distances."""
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("tip-axis ratio must be strictly between 0 and 1")
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.shape != (21, 3) or not np.isfinite(landmarks).all():
        raise ValueError("MANO landmarks must be finite [21,3]")
    points, surface, lengths = [], [], []
    for preceding, last in zip(PRECEDING_INDICES, LAST_JOINT_INDICES):
        origin = landmarks[last]
        direction = landmarks[last] - landmarks[preceding]
        direction /= np.linalg.norm(direction).clip(min=1e-12)
        distance = first_ray_mesh_exit(origin, direction, vertices, faces)
        points.append(origin + ratio * distance * direction)
        surface.append(origin + distance * direction)
        lengths.append(distance)
    return np.asarray(points), np.asarray(surface), np.asarray(lengths)
