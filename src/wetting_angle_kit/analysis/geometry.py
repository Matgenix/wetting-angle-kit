"""Droplet symmetry and the internal axis convention.

Every analyzer in :mod:`wetting_angle_kit.analysis` operates on a
:class:`DropletGeometry` instance. The class normalises the three
supported cases (``spherical``, ``cylinder_x``, ``cylinder_y``) and
exposes a single helper, :meth:`to_internal_coords`, that downstream
code can use to assume the cylinder axis is always ``y``.

User-facing APIs accept either a :class:`DropletGeometry` instance or
the bare string name; :meth:`DropletGeometry.coerce` is the canonical
entry point that performs the conversion.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

#: Public type alias for the three accepted droplet geometry names.
DropletGeometryName = Literal["spherical", "cylinder_x", "cylinder_y"]


@dataclass(frozen=True)
class DropletGeometry:
    """Droplet symmetry descriptor with axis-layout helpers.

    Three cases are supported:

    * ``spherical``: the droplet is a 3D cap with no preferred horizontal
      axis. Rays sweep over the upper hemisphere ``(theta, phi)``.
    * ``cylinder_y``: the droplet is a ridge whose translational symmetry
      axis is ``y``. In-plane analysis happens in ``(x, z)`` and slices
      are taken at successive ``y`` positions. No internal axis swap.
    * ``cylinder_x``: the droplet is a ridge whose translational symmetry
      axis is ``x``. A ``[1, 0, 2]`` swap is applied at the analyzer
      boundary so every downstream routine can assume the cylinder axis
      is ``y`` internally. The swap is self-inverse, so the same helper
      maps internal coordinates back to user coordinates.
    """

    _VALID_NAMES: ClassVar[tuple[DropletGeometryName, ...]] = (
        "spherical",
        "cylinder_x",
        "cylinder_y",
    )

    name: DropletGeometryName

    def __post_init__(self) -> None:
        if self.name not in self._VALID_NAMES:
            raise ValueError(
                f"droplet_geometry must be one of {self._VALID_NAMES}; "
                f"got {self.name!r}."
            )

    @classmethod
    def coerce(
        cls, value: "DropletGeometry | DropletGeometryName | str"
    ) -> "DropletGeometry":
        """Return a :class:`DropletGeometry` for either an instance or a name.

        Parameters
        ----------
        value : DropletGeometry or str
            Either an existing instance (returned unchanged) or one of the
            bare name strings ``"spherical"``, ``"cylinder_x"``,
            ``"cylinder_y"``.

        Returns
        -------
        DropletGeometry
        """
        if isinstance(value, cls):
            return value
        return cls(name=value)  # type: ignore[arg-type]

    @property
    def is_spherical(self) -> bool:
        return self.name == "spherical"

    @property
    def is_cylinder(self) -> bool:
        return self.name in ("cylinder_x", "cylinder_y")

    @property
    def cylinder_axis(self) -> Literal["x", "y"] | None:
        """User-frame axis along which the cylinder extends, or ``None``."""
        if self.name == "cylinder_x":
            return "x"
        if self.name == "cylinder_y":
            return "y"
        return None

    def to_internal_coords(self, coords: np.ndarray) -> np.ndarray:
        """Map coordinates from the user frame to the internal frame.

        For ``cylinder_x`` this applies the ``[1, 0, 2]`` swap so the
        cylinder axis ends up on the ``y`` column. Spherical and
        ``cylinder_y`` are returned unchanged. Accepts any array whose
        last axis has length 3 (a single point ``(3,)`` or a batch
        ``(..., 3)``).
        """
        if self.name == "cylinder_x":
            return coords[..., [1, 0, 2]]
        return coords

    def to_user_coords(self, coords: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_internal_coords`.

        The ``[1, 0, 2]`` swap is self-inverse, so this delegates back
        to :meth:`to_internal_coords`.
        """
        return self.to_internal_coords(coords)
