"""Strict name-based mapping between Wuji trajectory and simulator joints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class JointNameMapping:
    """An explicit mapping from a named source vector to a named destination."""

    source_names: tuple[str, ...]
    destination_names: tuple[str, ...]
    destination_indices: np.ndarray

    @classmethod
    def build(
        cls, source_names: Sequence[str], destination_names: Sequence[str]
    ) -> "JointNameMapping":
        source = tuple(str(name) for name in source_names)
        destination = tuple(str(name) for name in destination_names)
        if len(set(source)) != len(source):
            raise ValueError("source joint names contain duplicates")
        if len(set(destination)) != len(destination):
            raise ValueError("destination joint names contain duplicates")
        destination_lookup = {name: index for index, name in enumerate(destination)}
        missing = [name for name in source if name not in destination_lookup]
        if missing:
            raise ValueError(
                f"source joints absent from destination: {missing}; "
                f"destination joints={list(destination)}"
            )
        return cls(
            source_names=source,
            destination_names=destination,
            destination_indices=np.asarray(
                [destination_lookup[name] for name in source], dtype=np.int64
            ),
        )

    def scatter(self, source_values: np.ndarray, destination_values: np.ndarray) -> None:
        """Write a source vector into destination slots by joint name."""
        source_values = np.asarray(source_values)
        if source_values.shape != (len(self.source_names),):
            raise ValueError(
                f"source values must have shape {(len(self.source_names),)}, "
                f"got {source_values.shape}"
            )
        destination_values[self.destination_indices] = source_values
