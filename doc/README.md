# Building NCCL Documentation

## Quick Start: Wrapper Script (Recommended)

The easiest method, from the NCCL root directory:

```bash
doc/build_docs.sh
```

This automatically:
- Creates/updates a Python virtual environment at `doc/.venv`
- Installs the recommended pinned sphinx dependencies from `doc/requirements.txt`
- Builds HTML documentation using Make
- Packages HTML documentation into a .zip file.

The `doc/.venv` environment is persistent, but can be removed manually.  It is
automatically updated in case requirements.txt changes.

Output: `build/doc/html/index.html` and `build/pkg/doc/nccl-doc_*.zip`

## Manual Build: Makefile

If you have sufficient versions of the sphinx packages already installed, you
may not need the virtual environment, and can just run `make pkg.doc.build`.  If
you wish to install and use recommended python package versions, follow these
steps:

```bash
# 1. Create and activate virtual environment
python3 -m venv doc/.venv
source doc/.venv/bin/activate

# 2. Install dependencies
pip install -r doc/requirements.txt

# 3. Build documentation
make pkg.doc.build

# 4. Deactivate venv
deactivate
```

## Alternative: CMake

CMake build system also supports building the docs:

```bash
mkdir -p cmake_build && cd cmake_build
cmake .. -DBUILD_DOC_PACKAGE=ON
cmake --build . --target doc_build
```

## Additional Targets

For other Sphinx output formats (using Makefile or wrapper script):

```bash
doc/build_docs.sh linkcheck  # Check external links
doc/build_docs.sh latex      # Build LaTeX/PDF
doc/build_docs.sh man        # Build man pages
```

Run `make -C doc help` to see all available targets.