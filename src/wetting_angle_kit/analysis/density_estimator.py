"""Density-estimator strategies used across the analysis package.

A :class:`DensityEstimator` answers "how do I compute a density from
a set of atom positions?". The same strategy is consumed in four
distinct evaluation patterns:

- **Pointwise 3D** — used by :class:`InterfaceExtractor` with
  :meth:`SpaceSampling.rays`, via :meth:`build_field`. Returns a
  :class:`DensityFieldProtocol` whose ``.evaluate(positions)`` gives
  the density at arbitrary 3D query points (the ray sample
  positions).
- **Grid + slicing plane** — used by :class:`InterfaceExtractor`
  with :meth:`SpaceSampling.grid` in slicing mode, via
  :meth:`evaluate_on_slice`. Returns a 2D density on a slice-plane
  cell grid.
- **Grid + 3D volume** — used by :class:`InterfaceExtractor` with
  :meth:`SpaceSampling.grid` in whole mode, via
  :meth:`evaluate_on_3d_grid`. Returns a 3D density on a Cartesian
  cell grid.
- **Radial-projected / 3D** for the coupled-fit analyzers — used by
  :class:`CoupledFit2DAnalyzer` (:meth:`evaluate_2d`) and
  :class:`CoupledFit3DAnalyzer` (:meth:`evaluate_3d`). The 2D path
  exploits the spherical droplet's axisymmetry by folding atoms onto
  ``(xi = hypot(x, y), zi)`` cells and dividing by the annular
  volume; the cylinder path folds with ``xi = |x|`` and divides by
  the cylinder-length factor. The 3D path is a plain Cartesian grid.

Two concrete strategies are exposed via the classmethod factories:

- :meth:`DensityEstimator.binning` — top-hat histogram. Cheap and
  exact, intrinsically noisy at low per-cell counts.
- :meth:`DensityEstimator.gaussian` — 3D Gaussian KDE. Smooth, no
  per-cell Poisson noise, slightly more expensive.

Both factories return frozen dataclass instances that carry their
own parameters (``bin_width`` for binning; ``density_sigma`` and
``cutoff_sigma`` for the Gaussian); the consumers only need to know
about the abstract :class:`DensityEstimator` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from wetting_angle_kit.analysis._density import (
    DensityFieldProtocol,
    GaussianDensityField,
    HistogramDensityField,
)
from wetting_angle_kit.analysis.geometry import DropletGeometry


@dataclass(frozen=True)
class DensityEstimator(ABC):
    """Strategy interface for density estimation.

    Concrete instances come from one of the classmethod factories
    :meth:`binning` or :meth:`gaussian`; the abstract methods are
    dispatched by the analyzer / extractor that consumes them.
    """

    #: kind tag (used in tqdm labels).
    kind: ClassVar[str]

    # ------------------------------------------------------------------
    # Pointwise interface (SpaceSampling.rays).
    # ------------------------------------------------------------------

    @abstractmethod
    def build_field(self, atoms: np.ndarray) -> DensityFieldProtocol:
        """Pointwise 3D density evaluator on the given atom set.

        Returns an object exposing ``evaluate(positions)`` for
        arbitrary ``(N, 3)`` query points. Used by
        :meth:`SpaceSampling.rays` to sample density along each ray.

        The binning estimator requires ``bin_width`` to have been set
        on the factory call; calling :meth:`build_field` without one
        raises :class:`ValueError`.
        """

    # ------------------------------------------------------------------
    # Grid interface (SpaceSampling.grid).
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate_on_slice(
        self,
        atoms: np.ndarray,
        slice_center: np.ndarray,
        in_plane_axis: np.ndarray,
        s_centers: np.ndarray,
        z_centers: np.ndarray,
        slab_thickness: float,
    ) -> np.ndarray:
        """2D density on the cell centres of a slice plane.

        Returns a ``(len(s_centers), len(z_centers))`` array. The
        slice plane is defined by ``slice_center`` (a 3D point on
        the plane) and ``in_plane_axis`` (a horizontal unit vector
        defining the radial coordinate ``s``).

        For the Gaussian estimator, the KDE is evaluated at each
        cell-centre 3D point on the plane. For the binning estimator,
        atoms inside the slab ``|perp| ≤ slab_thickness / 2`` are
        histogrammed in ``(s, z)``; each cell's density is
        ``counts / (ds · dz · slab_thickness)``.
        """

    @abstractmethod
    def evaluate_on_3d_grid(
        self,
        atoms: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        z_centers: np.ndarray,
        *,
        x_offset: float,
        y_offset: float,
    ) -> np.ndarray:
        """3D density on the cell centres of a Cartesian grid.

        Returns a ``(len(x_centers), len(y_centers), len(z_centers))``
        array. The grid is laterally droplet-centred (``x_offset``,
        ``y_offset`` shift the cell coordinates back to the lab
        frame for evaluation against the lab-frame atoms).
        """

    # ------------------------------------------------------------------
    # Coupled-fit interface (radial / 3D box density with dV).
    # ------------------------------------------------------------------

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
        """Coupled-fit 2D: radial-projected ``(xi, zi)`` density.

        Returns a ``(n_xi, n_zi)`` array in atoms/Å³, averaged across
        the ``n_frames`` pooled into the batch. For spherical, atoms
        fold onto ``xi = hypot(x, y)`` with annular ``dV``; for
        cylinder, atoms fold onto ``xi = |x|`` with cylinder-length
        ``dV``.
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
        """Coupled-fit 3D: Cartesian ``(xi, yi, zi)`` density.

        Returns a ``(n_xi, n_yi, n_zi)`` array in atoms/Å³, averaged
        across the ``n_frames`` pooled into the batch.

        Only ``spherical`` is currently exercised — the 3D coupled-fit
        analyzer rejects cylinder droplets at construction.
        """

    # ------------------------------------------------------------------
    # Factories.
    # ------------------------------------------------------------------

    @classmethod
    def binning(cls, *, bin_width: float | None = None) -> DensityEstimator:
        """Top-hat histogram density estimator.

        Parameters
        ----------
        bin_width : float, optional
            Side length (Å) of the 3D top-hat kernel used by
            :meth:`build_field` for pointwise evaluation
            (:meth:`SpaceSampling.rays`). Ignored by :meth:`evaluate_on_slice`,
            :meth:`evaluate_on_3d_grid`, :meth:`evaluate_2d`, and
            :meth:`evaluate_3d` — those consumers derive their cell
            sizes from the grid spec they're given. Required only
            when the estimator is consumed pointwise.
        """
        return _BinningDensityEstimator(bin_width=bin_width)

    @classmethod
    def gaussian(
        cls,
        *,
        density_sigma: float = 3.0,
        cutoff_sigma: float = 5.0,
    ) -> DensityEstimator:
        """3D Gaussian KDE density estimator.

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


# ---------------------------------------------------------------------------
# Concrete implementations.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BinningDensityEstimator(DensityEstimator):
    """Concrete estimator for :meth:`DensityEstimator.binning`."""

    kind: ClassVar[str] = "binning"

    #: 3D top-hat kernel side length for pointwise evaluation. Required
    #: only by :meth:`build_field` (:meth:`SpaceSampling.rays`); ``None`` is
    #: fine when this estimator is consumed by grid or coupled-fit.
    bin_width: float | None

    def build_field(self, atoms: np.ndarray) -> DensityFieldProtocol:
        if self.bin_width is None:
            raise ValueError(
                "DensityEstimator.binning() needs bin_width=... for "
                "pointwise evaluation (SpaceSampling.rays). Either pass "
                "bin_width when building the estimator, or use it with the "
                "grid / coupled-fit consumers that derive the cell size "
                "from their grid spec."
            )
        return HistogramDensityField(atom_coords=atoms, bin_width=self.bin_width)

    def evaluate_on_slice(
        self,
        atoms: np.ndarray,
        slice_center: np.ndarray,
        in_plane_axis: np.ndarray,
        s_centers: np.ndarray,
        z_centers: np.ndarray,
        slab_thickness: float,
    ) -> np.ndarray:
        # Slab cut: atoms within ±slab_thickness/2 of the slice plane
        # along the perp direction. Then histogram their (s, z)
        # projection on the slice plane.
        perp_axis = np.array([-in_plane_axis[1], in_plane_axis[0], 0.0])
        rel = atoms - slice_center[None, :]
        s_coord = rel @ in_plane_axis
        perp_coord = rel @ perp_axis
        mask = np.abs(perp_coord) <= 0.5 * slab_thickness
        s_edges = _edges_from_centers(s_centers)
        z_edges = _edges_from_centers(z_centers)
        counts, _, _ = np.histogram2d(
            s_coord[mask], atoms[mask, 2], bins=(s_edges, z_edges)
        )
        ds = float(s_centers[1] - s_centers[0]) if len(s_centers) > 1 else 1.0
        dz = float(z_centers[1] - z_centers[0]) if len(z_centers) > 1 else 1.0
        return counts / (ds * dz * slab_thickness)

    def evaluate_on_3d_grid(
        self,
        atoms: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        z_centers: np.ndarray,
        *,
        x_offset: float,
        y_offset: float,
    ) -> np.ndarray:
        # The grid is droplet-centred (cells defined relative to COM).
        # Shift atoms to the same frame, then histogram into cells.
        atoms_centered = atoms - np.array([x_offset, y_offset, 0.0])
        x_edges = _edges_from_centers(x_centers)
        y_edges = _edges_from_centers(y_centers)
        z_edges = _edges_from_centers(z_centers)
        counts, _ = np.histogramdd(atoms_centered, bins=(x_edges, y_edges, z_edges))
        dx = float(x_centers[1] - x_centers[0]) if len(x_centers) > 1 else 1.0
        dy = float(y_centers[1] - y_centers[0]) if len(y_centers) > 1 else 1.0
        dz = float(z_centers[1] - z_centers[0]) if len(z_centers) > 1 else 1.0
        return counts / (dx * dy * dz)

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

    def build_field(self, atoms: np.ndarray) -> DensityFieldProtocol:
        return GaussianDensityField(
            atom_coords=atoms,
            density_sigma=self.density_sigma,
            cutoff_sigma=self.cutoff_sigma,
        )

    def evaluate_on_slice(
        self,
        atoms: np.ndarray,
        slice_center: np.ndarray,
        in_plane_axis: np.ndarray,
        s_centers: np.ndarray,
        z_centers: np.ndarray,
        slab_thickness: float,  # noqa: ARG002 — unused; KDE is pointwise
    ) -> np.ndarray:
        # Evaluate the 3D KDE at each cell-centre point on the slice
        # plane. The slab thickness is meaningless for a pointwise
        # estimator (kept in the signature for interface symmetry
        # with the binning variant).
        field = self.build_field(atoms)
        s_mesh, z_mesh = np.meshgrid(s_centers, z_centers, indexing="ij")
        positions = np.column_stack(
            [
                slice_center[0] + s_mesh.ravel() * in_plane_axis[0],
                slice_center[1] + s_mesh.ravel() * in_plane_axis[1],
                z_mesh.ravel(),
            ]
        )
        return field.evaluate(positions).reshape(s_mesh.shape)

    def evaluate_on_3d_grid(
        self,
        atoms: np.ndarray,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        z_centers: np.ndarray,
        *,
        x_offset: float,
        y_offset: float,
    ) -> np.ndarray:
        field = self.build_field(atoms)
        x_mesh, y_mesh, z_mesh = np.meshgrid(
            x_centers, y_centers, z_centers, indexing="ij"
        )
        positions = np.column_stack(
            [
                (x_mesh + x_offset).ravel(),
                (y_mesh + y_offset).ravel(),
                z_mesh.ravel(),
            ]
        )
        return field.evaluate(positions).reshape(x_mesh.shape)

    def evaluate_2d(
        self,
        *,
        atoms_pooled: np.ndarray,
        n_frames: int,
        droplet_geometry: DropletGeometry,
        xi_edges: np.ndarray,
        zi_edges: np.ndarray,
        box_dimension: float | None,  # noqa: ARG002 — unused; KDE is pointwise
    ) -> np.ndarray:
        field = self.build_field(atoms_pooled)
        xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        zi_cc = 0.5 * (zi_edges[:-1] + zi_edges[1:])
        xi_mesh, zi_mesh = np.meshgrid(xi_cc, zi_cc, indexing="ij")
        # Evaluation plane: y=0 for both geometries. Spherical: by
        # axisymmetry, (xi, 0, zi) is representative of the whole
        # annulus. Cylinder: atoms are droplet-centred in y, so y=0
        # is the cylinder midpoint; translational invariance.
        positions = np.column_stack(
            [xi_mesh.ravel(), np.zeros(xi_mesh.size), zi_mesh.ravel()]
        )
        rho_cc = field.evaluate(positions).reshape(xi_mesh.shape)
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
        field = self.build_field(atoms_pooled)
        xi_cc = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        yi_cc = 0.5 * (yi_edges[:-1] + yi_edges[1:])
        zi_cc = 0.5 * (zi_edges[:-1] + zi_edges[1:])
        xi_mesh, yi_mesh, zi_mesh = np.meshgrid(xi_cc, yi_cc, zi_cc, indexing="ij")
        positions = np.column_stack([xi_mesh.ravel(), yi_mesh.ravel(), zi_mesh.ravel()])
        rho = field.evaluate(positions).reshape(xi_mesh.shape)
        if n_frames > 0:
            rho = rho / n_frames
        return rho


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Recover cell edges from cell centres assuming uniform spacing."""
    if len(centers) < 2:
        return np.array([float(centers[0]) - 0.5, float(centers[0]) + 0.5])
    step = float(centers[1] - centers[0])
    return np.concatenate([centers - 0.5 * step, [centers[-1] + 0.5 * step]])
