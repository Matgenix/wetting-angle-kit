"""Density-estimator strategies for the coupled-fit analyzers.

The coupled-fit analyzers
(:class:`CoupledFit2DAnalyzer`, :class:`CoupledFit3DAnalyzer`) build a
cell-centred density field on a fixed grid and fit a hyperbolic-tangent
model to it jointly. The estimator strategy controls how the per-cell
density is computed from the pooled atom positions:

- :meth:`DensityEstimator.binning` — top-hat histogram with
  geometry-aware volume normalisation. The historical method
  (`legacy: BinningBatchFitter.binning`).
- :meth:`DensityEstimator.gaussian` — 3D Gaussian KDE evaluated at the
  cell centres. Same density estimator the ``rays_gaussian`` and
  ``grid_gaussian`` extractors use, so the joint tanh fit sees a
  smooth density field with no per-cell Poisson noise.

The input to the estimator is pooled atom positions in a standard
*droplet-centred internal frame*: ``(x, y)`` are recentered on the
batch-averaged droplet COM (with PBC unwrapping) and ``z`` stays in
the lab frame. That convention lets the estimator pick the right
projection for each ``DropletGeometry`` without needing a separate
pre-projection step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from wetting_angle_kit.analysis._density import GaussianDensityField
from wetting_angle_kit.analysis.geometry import DropletGeometry


@dataclass(frozen=True)
class DensityEstimator(ABC):
    """Strategy interface for the coupled-fit per-cell density.

    Concrete instances come from one of the classmethod factories
    :meth:`binning` or :meth:`gaussian`; the abstract ``evaluate_*``
    methods are dispatched by the analyzer worker after pooling
    droplet-centred atom positions across the batch.
    """

    #: Human-readable kind tag (used in tqdm labels). Set by each
    #: concrete subclass.
    kind: ClassVar[str]

    @abstractmethod
    def evaluate_2d(
        self,
        *,
        atoms_pooled: np.ndarray,
        n_frames: int,
        droplet_geometry: DropletGeometry,
        xi_edges: np.ndarray,
        zi_edges: np.ndarray,
        box_dimension: float | None,
    ) -> np.ndarray:
        """2D density on the ``(xi_cc, zi_cc)`` grid.

        Returns a ``(n_xi, n_zi)`` array in atoms/Å³, averaged across
        the ``n_frames`` pooled into the batch.

        Parameters
        ----------
        atoms_pooled : ndarray, shape (N, 3)
            Pooled atom positions in the droplet-centred internal
            frame (``x``/``y`` recentered on COM; ``z`` in lab frame).
        n_frames : int
            Number of frames pooled. Used to divide the cumulative
            density by frame count.
        droplet_geometry : DropletGeometry
            ``spherical`` projects atoms to radial ``xi = hypot(x, y)``;
            ``cylinder_*`` uses ``xi = |x|`` and folds across the
            cylinder axis.
        xi_edges, zi_edges : ndarray
            1D cell-edge arrays.
        box_dimension : float, optional
            Cylinder axis length, only needed for the binning
            estimator's geometry-aware ``dV`` factor on
            ``cylinder_*`` droplets.
        """

    @abstractmethod
    def evaluate_3d(
        self,
        *,
        atoms_pooled: np.ndarray,
        n_frames: int,
        droplet_geometry: DropletGeometry,
        xi_edges: np.ndarray,
        yi_edges: np.ndarray,
        zi_edges: np.ndarray,
    ) -> np.ndarray:
        """3D density on the ``(xi_cc, yi_cc, zi_cc)`` grid.

        Returns a ``(n_xi, n_yi, n_zi)`` array in atoms/Å³, averaged
        across the ``n_frames`` pooled into the batch.

        Only ``spherical`` is currently exercised — the 3D coupled-fit
        analyzer rejects cylinder droplets at construction.
        """

    # ------------------------------------------------------------------
    # Factories.
    # ------------------------------------------------------------------

    @classmethod
    def binning(cls) -> DensityEstimator:
        """Top-hat histogram density with geometry-aware ``dV``.

        Atoms in each cell contribute ``1 / dV / n_frames`` to the
        density. ``dV`` is the cell's annular volume for
        ``spherical``, ``2 · box_dimension · dxi · dzi`` for
        ``cylinder_*`` (the binning's |x| folding combined with the
        box-length integral along the cylinder axis), and the plain
        ``dxi · dyi · dzi`` in 3D.

        This is the legacy estimator; the coupled-fit analyzers
        default to it for backwards-compatible numerics.
        """
        return _BinningDensityEstimator()

    @classmethod
    def gaussian(
        cls,
        *,
        density_sigma: float = 3.0,
        cutoff_sigma: float = 5.0,
    ) -> DensityEstimator:
        """3D Gaussian KDE evaluated at the cell centres.

        Same density estimator as ``rays_gaussian`` and
        ``grid_gaussian``. For ``spherical`` droplets the per-cell
        evaluation point is ``(xi_cc, 0, zi_cc)``: by axisymmetry the
        single annular point is a representative for the whole
        annulus, so no annular integration is needed.

        Parameters
        ----------
        density_sigma : float, default 3.0
            Gaussian kernel width (Å).
        cutoff_sigma : float, default 5.0
            Per-atom kernel truncation in units of ``density_sigma``.
            Larger values are slower but more accurate in the
            kernel's tails.
        """
        return _GaussianDensityEstimator(
            density_sigma=density_sigma, cutoff_sigma=cutoff_sigma
        )


@dataclass(frozen=True)
class _BinningDensityEstimator(DensityEstimator):
    """Concrete estimator for :meth:`DensityEstimator.binning`."""

    kind: ClassVar[str] = "binning"

    def evaluate_2d(
        self,
        *,
        atoms_pooled: np.ndarray,
        n_frames: int,
        droplet_geometry: DropletGeometry,
        xi_edges: np.ndarray,
        zi_edges: np.ndarray,
        box_dimension: float | None,
    ) -> np.ndarray:
        if droplet_geometry.is_spherical:
            xi_vals = np.hypot(atoms_pooled[:, 0], atoms_pooled[:, 1])
        else:
            xi_vals = np.abs(atoms_pooled[:, 0])
        zi_vals = atoms_pooled[:, 2]
        counts, _, _ = np.histogram2d(xi_vals, zi_vals, bins=(xi_edges, zi_edges))
        dxi = float(xi_edges[1] - xi_edges[0])
        dzi = float(zi_edges[1] - zi_edges[0])
        xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        if droplet_geometry.is_cylinder:
            assert box_dimension is not None
            dV = 2.0 * box_dimension * dxi * dzi
            rho_cc = counts / dV
        else:
            dV_per_row = 2.0 * np.pi * xi_cc * dxi * dzi
            rho_cc = counts / dV_per_row[:, np.newaxis]
        if n_frames > 0:
            rho_cc = rho_cc / n_frames
        return rho_cc

    def evaluate_3d(
        self,
        *,
        atoms_pooled: np.ndarray,
        n_frames: int,
        droplet_geometry: DropletGeometry,
        xi_edges: np.ndarray,
        yi_edges: np.ndarray,
        zi_edges: np.ndarray,
    ) -> np.ndarray:
        counts, _ = np.histogramdd(atoms_pooled, bins=(xi_edges, yi_edges, zi_edges))
        dxi = float(xi_edges[1] - xi_edges[0])
        dyi = float(yi_edges[1] - yi_edges[0])
        dzi = float(zi_edges[1] - zi_edges[0])
        rho = counts / (dxi * dyi * dzi)
        if n_frames > 0:
            rho = rho / n_frames
        return rho


@dataclass(frozen=True)
class _GaussianDensityEstimator(DensityEstimator):
    """Concrete estimator for :meth:`DensityEstimator.gaussian`."""

    kind: ClassVar[str] = "gaussian"

    density_sigma: float
    cutoff_sigma: float

    def _build_field(self, atoms_pooled: np.ndarray) -> GaussianDensityField:
        return GaussianDensityField(
            atom_coords=atoms_pooled,
            density_sigma=self.density_sigma,
            cutoff_sigma=self.cutoff_sigma,
        )

    def evaluate_2d(
        self,
        *,
        atoms_pooled: np.ndarray,
        n_frames: int,
        droplet_geometry: DropletGeometry,
        xi_edges: np.ndarray,
        zi_edges: np.ndarray,
        box_dimension: float | None,
    ) -> np.ndarray:
        field = self._build_field(atoms_pooled)
        xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        zi_cc = 0.5 * (zi_edges[:-1] + zi_edges[1:])
        xi_mesh, zi_mesh = np.meshgrid(xi_cc, zi_cc, indexing="ij")
        # Evaluation plane: y=0 for both geometries.
        #   - spherical: by axisymmetry the (xi, 0, zi) point is a
        #     representative for the whole annulus at radius xi.
        #   - cylinder:  atoms are droplet-centred in y, so y=0 is the
        #     cylinder midpoint. Translational invariance along y means
        #     any y cross-section gives the same density.
        positions = np.column_stack(
            [xi_mesh.ravel(), np.zeros(xi_mesh.size), zi_mesh.ravel()]
        )
        rho_flat = field.evaluate(positions)
        rho_cc = rho_flat.reshape(xi_mesh.shape)
        if n_frames > 0:
            rho_cc = rho_cc / n_frames
        return rho_cc

    def evaluate_3d(
        self,
        *,
        atoms_pooled: np.ndarray,
        n_frames: int,
        droplet_geometry: DropletGeometry,
        xi_edges: np.ndarray,
        yi_edges: np.ndarray,
        zi_edges: np.ndarray,
    ) -> np.ndarray:
        field = self._build_field(atoms_pooled)
        xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        yi_cc = 0.5 * (yi_edges[:-1] + yi_edges[1:])
        zi_cc = 0.5 * (zi_edges[:-1] + zi_edges[1:])
        xi_mesh, yi_mesh, zi_mesh = np.meshgrid(xi_cc, yi_cc, zi_cc, indexing="ij")
        positions = np.column_stack([xi_mesh.ravel(), yi_mesh.ravel(), zi_mesh.ravel()])
        rho_flat = field.evaluate(positions)
        rho = rho_flat.reshape(xi_mesh.shape)
        if n_frames > 0:
            rho = rho / n_frames
        return rho
