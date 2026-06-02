"""Trajectory-level binning contact-angle analyzer."""

from typing import Any

from wetting_angle_kit.analysis.analyzer import BaseTrajectoryAnalyzer
from wetting_angle_kit.analysis.binning.angle_fitting import (
    BinningBatchFitter,
)
from wetting_angle_kit.analysis.binning.results import BinningResults


class BinningTrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """BaseTrajectoryAnalyzer implementation using the density-binning method."""

    def __init__(
        self,
        parser: Any,
        atom_indices: Any,
        droplet_geometry: str = "spherical",
        binning_params: dict[str, Any] | None = None,
        precentered: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parser : BaseParser
            Trajectory parser providing coordinates and box dimensions.
        atom_indices : Any
            Indices (or IDs) of liquid atoms to include in the density field.
        droplet_geometry : str, default "spherical"
            One of ``"spherical"``, ``"cylinder_x"``, ``"cylinder_y"``.
        binning_params : dict, optional
            Grid definition with keys ``xi_0``, ``xi_f``, ``nbins_xi``,
            ``zi_0``, ``zi_f``, ``nbins_zi``. A heuristic default is used if None.
        precentered : bool, default False
            Skip per-frame circular-mean PBC recentering. Setting this on a
            trajectory that does NOT satisfy the precondition will produce
            wrong results.
        """
        self.parser = parser
        self._analyzer = BinningBatchFitter(
            parser=parser,
            atom_indices=atom_indices,
            droplet_geometry=droplet_geometry,
            binning_params=binning_params,
            precentered=precentered,
        )

    def analyze(
        self,
        frame_range: list[int] | None = None,
        split_factor: int | None = None,
        **kwargs: Any,
    ) -> BinningResults:
        """Run the binning analysis.

        Parameters
        ----------
        frame_range : list[int], optional
            Frame indices to process. If None, all frames are used.
        split_factor : int, optional
            If given, split ``frame_range`` into sub-batches of this size and
            compute one angle per batch; if None, all frames form a single batch.
        **kwargs
            Reserved for future use.

        Returns
        -------
        BinningResults
            Per-batch contact angles, density fields and isoline data.
        """
        if frame_range is None:
            frame_range = list(range(self.parser.frame_count()))
        if split_factor is None:
            batch = self._analyzer.process_batch(frame_range)
            return BinningResults(
                batches=[batch],
                method_metadata={"frames_per_angle": len(frame_range)},
            )
        batches = []
        for batch_idx, start in enumerate(range(0, len(frame_range), split_factor)):
            end = min(start + split_factor, len(frame_range))
            batches.append(
                self._analyzer.process_batch(
                    frame_range[start:end],
                    batch_index=batch_idx + 1,
                )
            )
        return BinningResults(
            batches=batches,
            method_metadata={"frames_per_trajectory": split_factor},
        )
