# Contributing to wetting-angle-kit

Thanks for your interest in contributing. This document covers the basics
for getting a local development environment running, the conventions the
project follows, and how to report problems.

## Reporting issues and asking for help

- **Bugs and feature requests:** open an issue at
  [github.com/Matgenix/wetting-angle-kit/issues](https://github.com/Matgenix/wetting-angle-kit/issues).
  Please include a minimal reproducer (trajectory snippet or code) when
  possible, the Python version, and the package version (the git commit
  if you installed from source).
- **Questions / usage support:** also via the issue tracker — tag the
  issue with `question`.

## Development setup

The project requires Python 3.10 or higher and uses a `src/` layout. We
recommend using conda to manage the environment, as OVITO is distributed
through its own conda channel and is awkward to install otherwise.

```bash
git clone https://github.com/Matgenix/wetting-angle-kit.git
cd wetting-angle-kit
conda create -n wetting-angle-kit python=3.11
conda activate wetting-angle-kit
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
pip install -e .[dev,all,doc]
pre-commit install
```

`[all]` pulls in both trajectory backends (`ase`, `ovito`) and the
visualization helpers. If you do not need the OVITO backend (i.e. you
are not working with LAMMPS dump files), skip the `conda install ovito`
line and install with `pip install -e .[dev,ase,doc]` instead.

## Running tests

```bash
pytest                       # full suite
pytest -m "not slow"         # skip slow tests
pytest -m unit               # unit tests only
pytest -m integration        # end-to-end on fixture trajectories
pytest --cov=wetting_angle_kit
```

Fixture trajectories live under `tests/trajectories/`. Integration tests
read these files and assert on contact-angle values within reasonable
bands; if you add a new method or change a numerical default, expect to
update those tolerances.

## Code style and quality

The project uses pre-commit to enforce a consistent style. After
`pre-commit install`, every commit runs:

- `ruff` (lint + format)
- `mypy` in strict mode on `src/wetting_angle_kit/`
- `codespell`
- RST and YAML sanity checks

Run all hooks manually with:

```bash
pre-commit run --all-files
```

Other conventions:

- Public functions and classes get NumPy-style docstrings with
  Parameters / Returns / Raises sections.
- Type hints are required (`disallow_untyped_defs = true`).
- Prefer raising `ValueError` (or a more specific exception) over
  `assert` for runtime invariants; asserts are removed under `python -O`.

## Adding a new parser

The parser ABC is in
[src/wetting_angle_kit/parsers/base.py](src/wetting_angle_kit/parsers/base.py).
Implement `parse()`, `frame_count()`, and (if applicable) `box_size`. The
factory in
[src/wetting_angle_kit/parsers/factory.py](src/wetting_angle_kit/parsers/factory.py)
dispatches by file extension — add your extension there. Mirror the existing
parsers' handling of orthogonal cells and periodic boundary conditions.

## Adding a new contact-angle method

Subclass `BaseTrajectoryAnalyzer`
([src/wetting_angle_kit/analysis/analyzer.py](src/wetting_angle_kit/analysis/analyzer.py))
and add an integration test in `tests/test_analysis/` that
exercises the method on one of the fixture trajectories.

## Pull requests

1. Branch off `main`.
2. Add tests for any new behaviour.
3. Run `pre-commit run --all-files` and `pytest` locally before pushing.
4. Open a pull request describing the change and linking any related
   issue.
