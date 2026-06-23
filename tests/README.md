# Test suite for wetting-angle-kit

## Layout

```
tests/
├── conftest.py                     Shared constants + the trajectory_path() helper
├── test_io_utils.py                Unit tests for wetting_angle_kit.io_utils
├── test_parser/                    Per-format parsers, water finders, and factory
│   ├── test_parser_dump.py         LAMMPS dump parser (OVITO backend)
│   ├── test_parser_xyz.py          Extended-XYZ parser
│   ├── test_parser_ase.py          ASE-backed parser
│   ├── test_water_finders.py       Water-oxygen identification across formats
│   └── test_parser_factory.py      get_water_finder() dispatch by extension
├── test_analysis/                  Strategy units + end-to-end analyzer runs
│   ├── test_geometry.py            DropletGeometry (spherical / cylinder_x / cylinder_y)
│   ├── test_temporal.py            TemporalAggregator batching
│   ├── test_default_grid_params.py Auto-derived grid_params / binning_params defaults
│   ├── test_density_estimator.py   DensityEstimator: binning vs gaussian
│   ├── test_fitter_error_paths.py  SurfaceFitter error/validation paths
│   ├── test_slicing_fitter.py      SurfaceFitter.slicing() on synthetic shapes
│   ├── test_whole_fitter.py        SurfaceFitter.whole() + bootstrap std
│   ├── test_rays_with_gaussian.py  rays extractor, Gaussian density
│   ├── test_rays_with_binning.py   rays extractor, binning density (+ parity vs Gaussian)
│   ├── test_grid_slicing.py        grid extractor, slicing mode (marching squares)
│   ├── test_grid_whole.py          grid extractor, whole mode (marching cubes)
│   ├── test_slicing_method.py      TrajectoryAnalyzer slicing end-to-end (LAMMPS fixture)
│   ├── test_slicing_edge_cases.py  Pipeline validation / NaN guards / degenerate input
│   ├── test_trajectory_analyzer_integration.py  TrajectoryAnalyzer across strategy combos
│   ├── test_cylinder_coverage.py   Cylinder droplet × every extractor combination
│   ├── test_coupled_fit_2d.py      CoupledFit2DAnalyzer end-to-end
│   ├── test_coupled_fit_3d.py      CoupledFit3DAnalyzer end-to-end
│   ├── test_wall_detector_from_atoms_e2e.py  WallDetector.from_atoms through the pipeline
│   └── test_parallel_path.py       multiprocessing.Pool batch path
├── test_visualization/             Plotter smoke tests + helper unit tests
│   ├── test_angle_evolution_helpers.py
│   ├── test_angle_evolution_plotter.py
│   ├── test_density_contour_plotter.py
│   └── test_droplet_slice_plot.py
└── trajectories/                   Fixture trajectories used by the integration tests
```

## Running the tests

```bash
pytest                                          # full suite
pytest -m "not slow"                            # skip the slow integration tests
pytest -m integration                           # only end-to-end runs on fixtures
pytest --cov=wetting_angle_kit --cov-report=term-missing   # with coverage
```

The default options live in `pyproject.toml` (`[tool.pytest.ini_options]`):
verbose output, the ten slowest durations, and `--strict-markers` /
`--strict-config`.

## Markers

`integration`
: Reads a fixture trajectory from `tests/trajectories/` and exercises an
  analyzer end-to-end. Most of these also need an optional backend
  (see below).

`slow`
: Takes more than ~1 s — typically the per-frame slicing runs that fit a
  circle in every slice. Deselect with `-m "not slow"`.

`unit`
: The default class of pure-Python tests (helpers, geometry, fitters,
  factories). The marker itself is optional and rarely applied.

## Optional dependencies

Several `test_parser` and `test_analysis` modules call
`pytest.importorskip("ovito" / "ase" / "skimage")` at import time, so the
suite runs cleanly without the optional backends — those modules are
skipped rather than failing:

- **OVITO** — LAMMPS dump parsing (`ovito` extra).
- **ASE** — `.traj` / ASE-readable trajectories (`ase` extra).
- **scikit-image** — whole-mode grid extraction via marching cubes
  (`grid3d` extra).

Install everything for a full local run with `pip install -e .[dev,all]`
(plus the conda OVITO package; see `CONTRIBUTING.md`).

## Fixture trajectories

| File | Format | Contents |
| --- | --- | --- |
| `traj_spherical_drop_4k.lammpstrj` | LAMMPS dump | Spherical water droplet (~4000 molecules) on a wall |
| `traj_10_3_330w_nve_4k_reajust.lammpstrj` | LAMMPS dump | Smaller water-on-wall NVE run |
| `slice_10_mace_mlips_cylindrical_2_5.traj` | ASE `.traj` | Cylindrical droplet from a MACE MLIP run |
| `slice_10_mace_mlips_cylindrical_2_5.xyz` | extended XYZ | Same cylinder data, used for the XYZ parser |

The integration tests recover contact angles in a physically reasonable
band (~90–110° on the water/graphene-like fixtures) and assert on that
band as a regression check. If you change a numerical default or add a
method, expect to revisit those tolerances.

## Adding tests

- Put a unit test next to the existing unit tests for the same module.
- Mark trajectory-backed tests with `@pytest.mark.integration` (and
  `@pytest.mark.slow` if they take more than ~1 s), and guard any optional
  backend with `pytest.importorskip(...)`.
- Resolve fixture paths with `conftest.trajectory_path("foo.lammpstrj")`
  or `os.path.join(os.path.dirname(__file__), "../trajectories/...")`.
- Use the `tmp_path` built-in for any output directories so each test is
  hermetic.
