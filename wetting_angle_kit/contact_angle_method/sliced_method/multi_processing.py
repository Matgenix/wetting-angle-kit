import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from wetting_angle_kit.parser import BaseParser

multiprocessing.set_start_method("spawn", force=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ContactAngleSlicedParallel:
    """Batch-parallel contact angle analyzer for sliced method.

    Frames are grouped into batches to mitigate parser pickling issues and to
    amortize object construction cost. Each batch is processed in a separate
    process using ``ProcessPoolExecutor``.

    Parameters
    ----------
    filename : str
        Path to trajectory file.
    output_repo : str
        Directory to write per-frame results.
    droplet_geometry : str, default "spherical"
        Geometric model identifier (e.g. "cylinder_x", "cylinder_y", "spherical").
    atom_indices : ndarray, optional
        Indices of liquid particles (subset). Empty array selects none.
    delta_gamma : float, optional
        Additional gamma constraint / filtering distance if used by sliced method.
    delta_cylinder : float, optional
        Y (or X) half-width of selection cylinder in cylindrical modes.
    """

    def __init__(
        self,
        filename: str,
        output_repo: str,
        droplet_geometry: str = "spherical",
        atom_indices: Optional[np.ndarray] = None,
        delta_gamma: float = None,
        delta_cylinder: float = None,
    ):
        self.filename = filename
        self.output_repo = output_repo
        self.delta_gamma = delta_gamma
        self.delta_cylinder = delta_cylinder
        self.droplet_geometry = droplet_geometry
        self.atom_indices = atom_indices if atom_indices is not None else np.array([])
        os.makedirs(self.output_repo, exist_ok=True)

    def process_frames_parallel(
        self,
        frames_to_process: List[int],
        num_batches: int = 4,
        max_workers: Optional[int] = None,
    ) -> Dict[int, float]:
        """Process many frames in parallel batches.

        Parameters
        ----------
        frames_to_process : list[int]
            Frame numbers to analyze.
        num_batches : int, default 4
            Number of batches to partition frames into.
        max_workers : int, optional
            Maximum number of worker processes. Defaults to ``num_batches``.

        Returns
        -------
        dict[int, float]
            Mapping frame number -> mean contact angle (failed frames excluded).
        """
        if max_workers is None:
            max_workers = num_batches
        batches = self._create_batches(frames_to_process, num_batches)
        logger.info(
            f"Processing {len(frames_to_process)} frames in {len(batches)} batches "
            f"with {max_workers} workers"
        )
        results: Dict[int, float] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch_worker, batch_frames): batch_frames
                for batch_frames in batches
            }
            completed_batches = 0
            all_alfas = {}
            all_surfaces = {}
            all_popts = {}
            for future in as_completed(future_to_batch):
                batch_frames = future_to_batch[future]
                try:
                    batch_results = future.result()
                    for frame_num, mean_alpha, alfas, surfaces, popts in batch_results:
                        if mean_alpha is not None:
                            results[frame_num] = mean_alpha
                            all_alfas[frame_num] = alfas
                            all_surfaces[frame_num] = surfaces
                            all_popts[frame_num] = popts
                    completed_batches += 1
                    logger.info(
                        f"Completed batch {completed_batches}/{len(batches)} "
                        f"({len(batch_results)} frames)"
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        f"Error in batch for frames {batch_frames}: {e}",
                        exc_info=True,
                    )
        sorted_frames = sorted(all_alfas.keys())

        # Create a list of (frame_num, alfas), (frame_num, surface), (frame_num, popts)

        alfas_with_frames = [(f, all_alfas[f]) for f in sorted_frames]
        np.save(
            f"{self.output_repo}/all_alfas.npy",
            np.array(alfas_with_frames, dtype=object),
        )

        surfaces_with_frames = [(f, all_surfaces[f]) for f in sorted_frames]
        np.save(
            f"{self.output_repo}/all_surfaces.npy",
            np.array(surfaces_with_frames, dtype=object),
        )

        popts_with_frames = [(f, all_popts[f]) for f in sorted_frames]
        np.save(
            f"{self.output_repo}/all_popts.npy",
            np.array(popts_with_frames, dtype=object),
        )
        logger.info(
            f"Successfully processed {len(results)}/{len(frames_to_process)} frames"
        )

        return results

    def _create_batches(self, frames: List[int], num_batches: int) -> List[List[int]]:
        """Return frame batches of near-equal size."""
        if num_batches >= len(frames):
            return [[frame] for frame in frames]
        batch_size = math.ceil(len(frames) / num_batches)
        return [frames[i : i + batch_size] for i in range(0, len(frames), batch_size)]

    def _process_batch_worker(
        self, batch_frames: List[int]
    ) -> List[Tuple[int, Optional[float]]]:
        """Worker routine executed in child process for a batch."""
        try:
            from wetting_angle_kit.io_utils import detect_parser_type
            from wetting_angle_kit.parser.base_parser import BaseParser
            from wetting_angle_kit.parser.parser_ase import AseParser
            from wetting_angle_kit.parser.parser_dump import DumpParser
            from wetting_angle_kit.parser.parser_xyz import XYZParser
        except ImportError as e:  # pragma: no cover
            logger.error(f"Failed to import required classes: {e}")
            return [(frame, None) for frame in batch_frames]
        try:
            parser_type = detect_parser_type(self.filename)
            logger.info(f"Detected parser type: {parser_type}")
            parser_class: Type[BaseParser]
            if parser_type == "dump":
                parser_class = DumpParser
            elif parser_type == "ase":
                parser_class = AseParser
            elif parser_type == "xyz":
                parser_class = XYZParser
            else:
                raise ValueError(f"Unsupported parser type: {parser_type}")
            parser = parser_class(filepath=self.filename)
        except Exception as e:  # pragma: no cover
            logger.error(f"Error initializing parser: {e}")
            return [(frame, None) for frame in batch_frames]
        batch_results: List[Tuple[int, Optional[float]]] = []
        for frame_num in batch_frames:
            try:
                result = self._process_single_frame_with_parsers(
                    frame_num, self.atom_indices, parser
                )
                batch_results.append(result)
            except Exception as e:  # pragma: no cover
                logger.error(f"Error processing frame {frame_num}: {e}")
                batch_results.append((frame_num, None, [], [], []))
        return batch_results

    def _process_single_frame_with_parsers(
        self, frame_num: int, atom_indices: np.ndarray, parser: BaseParser
    ) -> Tuple[int, Optional[float]]:
        """Process a single frame and compute mean contact angle.

        Returns
        -------
        tuple[int, float|None]
            Frame number and mean angle; ``None`` if processing failed.
        """
        try:
            from .angle_fitting_sliced import (
                ContactAngleSliced,
            )

        except ImportError as e:  # pragma: no cover
            logger.error(f"Missing sliced predictor dependency: {e}")
            return frame_num, None, [], [], []
        logger.info(f"START processing frame {frame_num}")
        try:
            liquid_positions = parser.parse(
                frame_index=frame_num,
                indices=atom_indices,
            )
            max_dist = int(
                np.max(
                    np.array(
                        [
                            parser.box_size_y(frame_index=frame_num),
                            parser.box_size_x(frame_index=frame_num),
                        ]
                    )
                )
                / 2
            )
            logger.info(
                f"Frame {frame_num}: Parsed {len(liquid_positions)} liquid "
                f"particles with max_dist {max_dist}"
            )
            if self.droplet_geometry == "cylinder_x":
                liquid_positions = liquid_positions[:, [1, 0, 2]]
            if self.droplet_geometry == "cylinder_x":
                box_dimensions = parser.box_size_x(frame_index=frame_num)
            elif self.droplet_geometry == "cylinder_y":
                box_dimensions = parser.box_size_y(frame_index=frame_num)
            else:
                box_dimensions = None
            mean_liquid_position = np.mean(liquid_positions, axis=0)
            predictor = ContactAngleSliced(
                liquid_coordinates=liquid_positions,
                max_dist=max_dist,
                liquid_geom_center=mean_liquid_position,
                droplet_geometry=self.droplet_geometry,
                delta_gamma=self.delta_gamma,
                width_cylinder=box_dimensions,
                delta_cylinder=self.delta_cylinder,
            )
            list_alfas, list_surfaces, list_popt = predictor.predict_contact_angle()
            if len(list_alfas) == 0:
                logger.warning(f"Frame {frame_num}: No angles computed (empty list).")
                mean_alpha = None
            else:
                mean_alpha = float(np.mean(list_alfas))
            if mean_alpha is not None:
                logger.info(f"Frame {frame_num} - mean angle: {mean_alpha:.2f}°")
            return frame_num, mean_alpha, list_alfas, list_surfaces, list_popt
        except Exception as e:  # pragma: no cover
            logger.error(f"Error processing frame {frame_num}: {e}")
            return frame_num, None, [], [], []
