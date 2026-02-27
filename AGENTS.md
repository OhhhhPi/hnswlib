# Repository Guidelines

## Project Structure & Module Organization
- Core library code lives in `hnswlib/` (header-only C++ implementation).
- Python bindings are in `python_bindings/` (`bindings.cpp`, package init, Python-only helpers).
- Tests are split by language: `tests/python/` for binding behavior and `tests/cpp/` for native correctness and stress checks.
- Runnable examples are in `examples/python/` and `examples/cpp/`.
- Benchmark and experiment pipelines live in `benchmark/` and `scripts/`; large local artifacts are typically under `data/`, `results/`, and `logs/`.

## Build, Test, and Development Commands
- `python -m pip install .`: build and install the Python extension locally.
- `make test`: run Python binding tests (`unittest` discovery for `bindings_test*.py`).
- `python -m unittest discover -v --start-directory examples/python --pattern "example*.py"`: verify Python examples still run.
- `mkdir -p build && cd build && cmake .. && make -j`: build C++ examples and test binaries.
- `cd tests/cpp && python update_gen_data.py`: generate/update C++ test data before update-related tests.
- `cd build && ./test_updates && ./epsilon_search_test`: run representative C++ tests (CI runs the full list in `.github/workflows/build.yml`).

## Coding Style & Naming Conventions
- Follow existing C++11 style in `hnswlib/*.h`: 4-space indentation, `CamelCase` types, `snake_case`/`camelCase` functions as already used, and trailing `_` for member fields.
- Keep Python code PEP 8-friendly: 4-space indentation, `snake_case` functions/files.
- No formatter/linter is enforced in-repo; keep diffs minimal and consistent with surrounding code.

## Testing Guidelines
- Framework: Python uses `unittest`; C++ uses compiled test executables.
- Test naming: Python files follow `bindings_test*.py`; C++ files follow `*_test.cpp`.
- Add or update tests for any behavior/API change, especially serialization, filtering, multithreading, and replace-deleted paths.

## Commit & Pull Request Guidelines
- Target branch for PRs is `develop`.
- Commit messages in history are short, imperative summaries; optional prefixes like `feat(scope): ...` are acceptable.
- PRs should include: what changed, why, test commands run, and any performance/recall impact when relevant.
- Link related issues and call out compatibility risks (index format, API changes, threading behavior).

## Configuration & Portability Notes
- `HNSWLIB_NO_NATIVE=1` disables `-march=native` for more portable Python extension builds.
- For cross-platform parity, mirror CI steps in `.github/workflows/build.yml` before submitting.
