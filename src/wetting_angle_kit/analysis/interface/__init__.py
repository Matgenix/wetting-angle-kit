"""The interface-finding subsystem.

The submodule owns everything related to recovering the liquid–vapor
interface from atom positions. An :class:`InterfaceExtractor` composes
two orthogonal strategy objects:

* a :class:`SpaceSampling` (built via :meth:`SpaceSampling.rays` or
  :meth:`SpaceSampling.grid`) that decides *where* density is
  evaluated;
* a :class:`DensityEstimator` (built via
  :meth:`DensityEstimator.gaussian` or :meth:`DensityEstimator.binning`)
  that decides *how* it is computed.

Both choices are independent — any sampling can be paired with any
density estimator::

    extractor = InterfaceExtractor(
        sampling=SpaceSampling.rays(
            delta_azimuthal=20.0, delta_polar=8.0,
        ),
        density=DensityEstimator.gaussian(density_sigma=3.0),
    )

The pairing between the chosen extractor and the analyzer's
:class:`SurfaceFitter` is validated at :class:`TrajectoryAnalyzer`
construction via :meth:`InterfaceExtractor.validate_compatibility`,
which forwards to :meth:`SpaceSampling.validate_compatibility`.
"""

from wetting_angle_kit.analysis.interface.base import (
    InterfaceExtractor,
    SpaceSampling,
)

__all__ = ["InterfaceExtractor", "SpaceSampling"]
