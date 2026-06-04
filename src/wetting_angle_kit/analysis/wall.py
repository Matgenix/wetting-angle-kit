"""Wall-plane detectors used by trajectory analyzers.

A :class:`WallDetector` returns the z-coordinate of the wall plane
that a :class:`SurfaceFitter` intersects to compute the contact angle.
Three strategies are supported:

- ``min_plus_offset``: take the lowest interface point and shift up by
  a configurable offset. Cheap and self-contained but picks up thermal
  noise from the liquid–vapor interface bottom.
- ``explicit``: use a fixed user-supplied z value. Best when the wall
  plane is known a priori from the simulation setup.
- ``from_atoms``: derive z from a pool of wall atom positions (e.g.
  mean z of the topmost layer). Most physical but requires the
  analyzer to be told which atoms form the wall.

Users construct detectors through classmethod factories on the base
class::

    WallDetector.min_plus_offset(offset=2.0)
    WallDetector.explicit(z_wall=15.0)
    WallDetector.from_atoms(wall_atom_indices=indices)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

#: Interface point set produced by an :class:`InterfaceExtractor`.
#:
#: - In slicing mode, a list of ``(N_i, 2)`` arrays in the per-slice
#:   ``(x, z)`` plane.
#: - In whole mode, a single ``(N, 3)`` array in the internal
#:   ``(x, y, z)`` frame.
InterfaceData: TypeAlias = list[np.ndarray] | np.ndarray


@dataclass(frozen=True)
class WallContext:
    """Per-batch data passed to :meth:`WallDetector.detect`.

    Wrapping the inputs in a single object keeps the detector method
    signature forward-compatible: new detectors can read new fields
    without changing the protocol.

    Attributes
    ----------
    interface_data : list[ndarray] or ndarray
        Interface point set produced by the :class:`InterfaceExtractor`;
        format depends on the extractor kind (per-slice 2D points or a
        3D shell).
    wall_coords : ndarray, optional
        Pooled ``(N, 3)`` positions of wall atoms in the internal
        coordinate frame, if the analyzer was constructed with
        ``wall_atom_indices``. Required by ``from_atoms`` detectors and
        unused by the others.
    """

    interface_data: InterfaceData
    wall_coords: np.ndarray | None = None


class WallDetector(ABC):
    """Abstract base for wall-plane detection strategies.

    Construct concrete detectors with one of the classmethod factories
    :meth:`min_plus_offset`, :meth:`explicit`, or :meth:`from_atoms`.
    Direct subclassing is supported for custom strategies but the
    factories cover all built-in cases.
    """

    @abstractmethod
    def detect(self, ctx: WallContext) -> float:
        """Return the wall-plane z-coordinate for one batch.

        Parameters
        ----------
        ctx : WallContext
            Per-batch data; see :class:`WallContext`.

        Returns
        -------
        float
            Wall-plane z in the internal coordinate frame (Å).
        """

    @classmethod
    def min_plus_offset(cls, offset: float = 2.0) -> "WallDetector":
        """Take the lowest interface point and shift up by ``offset``.

        Parameters
        ----------
        offset : float, default 2.0
            Vertical shift (Å) added to the minimum z to skip the
            wall-adjacent density spike. The default of 2.0 Å matches
            the slicing analyzer's historical behaviour for water on
            silica-like surfaces; tune for other systems.
        """
        return _MinPlusOffsetDetector(offset=offset)

    @classmethod
    def explicit(cls, z_wall: float) -> "WallDetector":
        """Use a fixed wall z-coordinate.

        Parameters
        ----------
        z_wall : float
            Wall-plane z in the internal coordinate frame (Å).
        """
        return _ExplicitDetector(z_wall=z_wall)

    @classmethod
    def from_atoms(
        cls,
        wall_atom_indices: np.ndarray,
        method: Literal["max_z", "mean_top_layer"] = "mean_top_layer",
        top_layer_tolerance: float = 1.0,
    ) -> "WallDetector":
        """Derive wall z from a set of wall atom positions.

        The analyzer must be constructed with the matching
        ``wall_atom_indices`` so the wall atoms are gathered and
        supplied through :attr:`WallContext.wall_coords`.

        Parameters
        ----------
        wall_atom_indices : ndarray
            Indices of the atoms that form the wall.
        method : {"max_z", "mean_top_layer"}, default "mean_top_layer"
            How to reduce wall atom z values to a single plane.
            ``"max_z"`` uses the highest wall atom z; cheap but noisy.
            ``"mean_top_layer"`` averages over all atoms within
            ``top_layer_tolerance`` Å of the maximum, smoothing thermal
            motion.
        top_layer_tolerance : float, default 1.0
            Vertical window (Å) defining the "top layer" for
            ``method="mean_top_layer"``. Ignored for ``"max_z"``.
        """
        return _FromAtomsDetector(
            wall_atom_indices=np.asarray(wall_atom_indices),
            method=method,
            top_layer_tolerance=top_layer_tolerance,
        )


@dataclass(frozen=True)
class _MinPlusOffsetDetector(WallDetector):
    """Concrete detector for :meth:`WallDetector.min_plus_offset`."""

    offset: float

    def detect(self, ctx: WallContext) -> float:
        data = ctx.interface_data
        if isinstance(data, list):
            z_mins = [float(np.min(s[:, 1])) for s in data if s.size > 0]
            if not z_mins:
                raise ValueError(
                    "min_plus_offset: interface_data has no non-empty slices."
                )
            z_min = min(z_mins)
        else:
            if data.size == 0:
                raise ValueError("min_plus_offset: interface_data is empty.")
            z_min = float(np.min(data[:, 2]))
        return z_min + self.offset


@dataclass(frozen=True)
class _ExplicitDetector(WallDetector):
    """Concrete detector for :meth:`WallDetector.explicit`."""

    z_wall: float

    def detect(self, ctx: WallContext) -> float:
        return self.z_wall


# eq=False avoids the auto-generated __eq__ tripping on the numpy field;
# equality between detectors isn't a use case we need.
@dataclass(frozen=True, eq=False)
class _FromAtomsDetector(WallDetector):
    """Concrete detector for :meth:`WallDetector.from_atoms`."""

    wall_atom_indices: np.ndarray
    method: Literal["max_z", "mean_top_layer"]
    top_layer_tolerance: float

    def detect(self, ctx: WallContext) -> float:
        if ctx.wall_coords is None:
            raise ValueError(
                "from_atoms wall detection requires wall_coords in the "
                "context; construct the analyzer with wall_atom_indices "
                "so the wall atoms are loaded each batch."
            )
        z = ctx.wall_coords[:, 2]
        if z.size == 0:
            raise ValueError("from_atoms wall detection received empty wall_coords.")
        if self.method == "max_z":
            return float(np.max(z))
        z_max = float(np.max(z))
        top = z[z >= z_max - self.top_layer_tolerance]
        return float(np.mean(top))
