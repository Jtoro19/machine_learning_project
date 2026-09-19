# Project Packaging Specification

## Purpose

Make the shared `src/` code importable from any consumer (notebooks, tests, future skills and the dashboard) without manual kernel registration or path hacks, and without hardcoded absolute paths anywhere in the codebase. This capability defines the installable package boundary that every other capability in this change is built on top of.

## Requirements

### Requirement: Installable src-layout package

The system MUST expose the shared code as an installable Python package at `src/nids/`, and `pyproject.toml` MUST declare a build backend so the package installs automatically when the project's dependency manager syncs the environment.

Because the importable package name (`nids`) does not match the normalized project name (`machine_learning_project`), automatic package auto-detection by the build backend is not reliable. The build configuration MUST therefore declare the package location explicitly rather than relying on name-matching auto-detection.

#### Scenario: Fresh clone becomes importable after environment sync

- GIVEN a fresh clone of the repository with no prior build artifacts
- WHEN the environment is synced (`uv sync`)
- THEN `nids` installs into the virtual environment as an editable package
- AND running `python -c "import nids"` inside that environment succeeds with no error

#### Scenario: Explicit package location prevents silent auto-detection failure

- GIVEN the importable package directory name (`nids`) differs from the normalized project name (`machine_learning_project`)
- WHEN `pyproject.toml` build configuration is inspected
- THEN it MUST contain an explicit wheel packages declaration naming `src/nids`
- AND the package MUST NOT rely solely on build-backend name-matching auto-detection

### Requirement: Notebook kernel resolves the package with zero manual registration

Any notebook launched through the project's documented launch command MUST be able to `import nids` without a separate kernel installation step. The system MUST document the one launch command that guarantees this.

#### Scenario: Package resolves when launched through the documented command

- GIVEN the environment has been synced and the package is installed editable
- WHEN Jupyter is launched through the documented command from the repository root
- THEN the running kernel's interpreter MUST be the project's own virtual environment
- AND the first cell of any notebook can execute `import nids` successfully

#### Scenario: A kernel from a different interpreter does not resolve the package

- GIVEN a Jupyter kernel was registered against a Python interpreter other than the project's virtual environment
- WHEN a notebook running on that kernel attempts `import nids`
- THEN the import MUST fail
- AND this constraint MUST be documented so it is not rediscovered as a bug

### Requirement: Repository-root resolution with no hardcoded absolute paths

No module under `src/nids/` MAY contain a hardcoded absolute filesystem path. The package MUST expose a path-resolution surface that derives the repository root from the installed package's own location, and derives the raw-data and results directories from that root.

#### Scenario: Repository root resolves independent of the calling process working directory

- GIVEN the package is installed into the active environment
- WHEN the repository-root resolution function is called from a process whose current working directory is not the repository root (for example, from within `notebooks/`)
- THEN it MUST still resolve to the actual repository root
- AND the raw-data directory and a given results directory MUST resolve as paths under that root

#### Scenario: No absolute path literals exist in the package source

- GIVEN the full `src/nids/` source tree
- WHEN it is inspected for filesystem path literals
- THEN no module MAY contain a hardcoded absolute path
- AND every path used by the package MUST be derived at runtime from the repository-root resolution surface
