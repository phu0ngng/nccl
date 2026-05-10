#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

"""
Generate Cython bindings for NCCL using cybind.

Drives one or more cybind targets:
  - nccl_core: libnccl.so bindings (output: nccl/bindings/)
  - nccl_ep:   libnccl_ep.so bindings (output: nccl/ep/bindings/)

Each target supplies its own config, template tree, and module path.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# cybind repository configuration
CYBIND_COMMIT = "b3d44118f6af338357db8664f9f97ad538b9eead"
CYBIND_SSH_URL = "ssh://git@gitlab-master.nvidia.com:12051/leof/cybind.git"

# Script directory for resolving default paths
SCRIPT_DIR = Path(__file__).resolve().parent
NCCL4PY_DIR = SCRIPT_DIR.parent
REPO_ROOT = NCCL4PY_DIR.parent.parent

# Global logger - will be configured in main()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Target:
    """Per-binding-target configuration for cybind."""

    name: str  # cybind library name (matches top-level key in YAML)
    assets_subdir: Path  # build_assets/<this> contains cybind/{*.yaml,templates/}
    config_filename: str  # YAML filename inside assets_subdir/cybind/
    module_parts: tuple[
        str, ...
    ]  # dotted module path, e.g. ("nccl4py", "ep", "bindings", "nccl_ep")
    output_dir: Path  # destination for generated bindings inside the package
    header_files: tuple[Path, ...]  # extra headers to stage into the cybind input dir
    nccl_header_required: bool  # whether to also stage build/include/nccl.h
    entry_headers: tuple[
        str, ...
    ]  # filenames cybind parses from input dir (must match YAML `headers:`)


TARGETS: dict[str, Target] = {
    "nccl_core": Target(
        name="nccl",
        assets_subdir=SCRIPT_DIR / "cybind",
        config_filename="nccl.cybind.yaml",
        module_parts=("nccl4py", "bindings", "nccl"),
        output_dir=NCCL4PY_DIR / "nccl" / "bindings",
        header_files=(),
        nccl_header_required=True,
        entry_headers=("nccl_device_expanded.h",),
    ),
    "nccl_ep": Target(
        name="nccl_ep",
        assets_subdir=SCRIPT_DIR / "nccl_ep" / "cybind",
        config_filename="nccl_ep.cybind.yaml",
        module_parts=("nccl4py", "ep", "bindings", "nccl_ep"),
        output_dir=NCCL4PY_DIR / "nccl" / "ep" / "bindings",
        header_files=(
            REPO_ROOT / "contrib" / "nccl_ep" / "include" / "nccl_ep.h",
            REPO_ROOT / "contrib" / "nccl_ep" / "include" / "ep_enums.h",
        ),
        nccl_header_required=True,
        entry_headers=("nccl_ep.h",),
    ),
}


def check_command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def clone_cybind(cybind_dir: Path) -> Path:
    if not check_command_exists("git"):
        raise RuntimeError("git command not found. Please install git.")

    try:
        subprocess.run(
            ["git", "clone", CYBIND_SSH_URL, str(cybind_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "switch", "--detach", CYBIND_COMMIT],
            cwd=cybind_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Cloned -> {cybind_dir}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone cybind: {e.stderr}") from e

    return cybind_dir


def stage_input_headers(target: Target, nccl_include_dir: Path, staged_dir: Path) -> Path:
    """Stage every header cybind needs into a single deterministic input dir.

    Two reasons this exists rather than pointing --input-dir at one of the
    source trees directly:
      1. cybind's CLI only accepts one --input-dir, but nccl_ep.h lives in
         contrib/nccl_ep/include while nccl.h lives in build/include.
      2. We want #include <nccl.h> to resolve to the byte-for-byte copy we
         control. CPATH-style alternatives are searched after every -I path
         (fake_libc_include, cybind/include, CUDA include, the input dir),
         so a stray nccl.h on any of those would silently win. Staging puts
         the intended nccl.h on the -I search path itself.

    Returns:
        Path to the staged include directory.
    """
    staged_dir.mkdir(parents=True, exist_ok=True)
    for header in target.header_files:
        if not header.exists():
            raise FileNotFoundError(f"Required header not found: {header}")
        shutil.copy2(header, staged_dir / header.name)
        logger.info(f"Staged {header.name}")

    if target.nccl_header_required:
        nccl_h = nccl_include_dir / "nccl.h"
        if not nccl_h.exists():
            raise FileNotFoundError(
                f"nccl.h not found at {nccl_h}. Build NCCL first or pass --nccl-include-dir."
            )
        # Only copy nccl.h here for nccl_ep (which needs it via #include <nccl.h>);
        # for nccl_core, the nccl_include_dir IS the input dir, no staging needed.
        if target.name != "nccl":
            shutil.copy2(nccl_h, staged_dir / nccl_h.name)
            logger.info(f"Staged {nccl_h.name}")

    return staged_dir


def prepare_cybind_assets(cybind_dir: Path, target: Target) -> None:
    """
    Copy NCCL configuration and templates to cybind directory.

    Args:
        - cybind_dir (Path): Path to cybind repository.
        - target (Target): Target whose YAML config and templates to stage.

    Raises:
        - ``FileNotFoundError``: If required files don't exist.
    """
    config_src = target.assets_subdir / target.config_filename
    config_dst = cybind_dir / "cybind" / "assets" / "configs" / target.config_filename

    if not config_src.exists():
        raise FileNotFoundError(f"Config file not found: {config_src}")

    logger.info(f"Copying config: {config_src} -> {config_dst}")
    shutil.copy2(config_src, config_dst)

    templates_src = target.assets_subdir / "templates"
    if not templates_src.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_src}")

    # cybind looks up templates by the module's dotted path, dropping the leaf:
    #   templates/<module_parts[:-1]>/<libname>.{pxd,pyx}
    templates_dst = cybind_dir.joinpath("cybind", "assets", "templates", *target.module_parts[:-1])
    templates_dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying templates: {templates_src} -> {templates_dst}")
    shutil.copytree(templates_src, templates_dst, dirs_exist_ok=True)


def run_cybind(cybind_dir: Path, target: Target, input_dir: Path, output_dir: Path) -> None:
    """
    Run cybind to generate bindings for the given target.

    Args:
        - cybind_dir (Path): Path to cybind repository.
        - target (Target): Target whose library name drives cybind's --generate flag.
        - input_dir (Path): Directory containing the headers for cybind to parse.
        - output_dir (Path): Output directory for generated bindings.

    Raises:
        - ``RuntimeError``: If uv is not available or cybind execution fails.

    Note:
        - Requires CUDA_PATH environment variable to be set for CUDA header resolution.
        - Uses uv to create an isolated virtual environment and install cybind dependencies.
    """
    if not check_command_exists("uv"):
        raise RuntimeError("uv command not found. Please install uv (https://docs.astral.sh/uv/)")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "--isolated",
        "--with",
        ".",
        "-m",
        "cybind",
        "--generate",
        target.name,
        "--input-dir",
        str(input_dir),
        "--output",
        str(output_dir),
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Working directory: {cybind_dir}")
    logger.info(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'not set')}")

    try:
        logger.info("Running cybind...")
        result = subprocess.run(cmd, cwd=cybind_dir, check=True, capture_output=True, text=True)
        logger.info("cybind execution stdout:")
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")
        if result.stderr:
            logger.info("cybind execution stderr:")
            for line in result.stderr.splitlines():
                logger.info(f"  {line}")
        logger.info("Successfully generated bindings")
    except subprocess.CalledProcessError as e:
        logger.error(f"cybind failed with exit code {e.returncode}")
        if e.stdout:
            logger.error("=== stdout ===")
            for line in e.stdout.splitlines():
                logger.error(line)
        if e.stderr:
            logger.error("=== stderr ===")
            for line in e.stderr.splitlines():
                logger.error(line)
        raise RuntimeError("cybind execution failed") from e


def copy_bindings(cybind_output_dir: Path, target: Target) -> None:
    """
    Copy generated bindings to the target's destination directory in nccl4py.

    Args:
        - cybind_output_dir (Path): Directory containing generated bindings.
        - target (Target): Target describing the source module path and final destination.

    Raises:
        - ``FileNotFoundError``: If generated files don't exist.
    """
    # cybind emits to <out>/<module_parts[:-1]>/
    generated_bindings = cybind_output_dir.joinpath(*target.module_parts[:-1])
    if not generated_bindings.exists():
        raise FileNotFoundError(f"Generated bindings not found: {generated_bindings}")

    package_bindings_dir = target.output_dir
    if package_bindings_dir.exists():
        logger.info(f"Removing target directory: {package_bindings_dir}")
        shutil.rmtree(package_bindings_dir)

    logger.info(f"Copying generated bindings: {generated_bindings} -> {package_bindings_dir}")
    shutil.copytree(generated_bindings, package_bindings_dir)

    # Copy static template files cybind doesn't process (__init__.py, utils.*).
    templates_dir = target.assets_subdir / "templates"

    logger.info("Copying static files from templates...")

    bindings_init_src = templates_dir / "__init__.py"
    if bindings_init_src.exists():
        logger.info(f"  {bindings_init_src.name}")
        shutil.copy2(bindings_init_src, package_bindings_dir / "__init__.py")

    internal_dir = package_bindings_dir / "_internal"
    internal_init_src = templates_dir / "_internal" / "__init__.py"
    if internal_init_src.exists():
        logger.info(f"  _internal/{internal_init_src.name}")
        shutil.copy2(internal_init_src, internal_dir / "__init__.py")

    for utils_file in (templates_dir / "_internal").glob("utils.*"):
        logger.info(f"  _internal/{utils_file.name}")
        shutil.copy2(utils_file, internal_dir / utils_file.name)

    logger.info("Successfully copied all bindings")


def generate_target(target: Target, nccl_include_dir: Path) -> None:
    """Run the full generate pipeline for a single target."""
    logger.warning(f">>> Generating bindings for target: {target.name}")

    logger.info(f"Assets directory: {target.assets_subdir}")
    logger.info(f"Target bindings directory: {target.output_dir}")
    logger.info("")

    with tempfile.TemporaryDirectory(prefix=f"nccl4py_cybind_{target.name}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logger.info(f">>> Using temporary directory: {temp_dir}")
        logger.info("")

        # Step 1: Clone cybind
        logger.info(">>> Cloning cybind...")
        cybind_dir = temp_dir / "cybind"
        clone_cybind(cybind_dir)
        logger.info("")

        # Step 2: Prepare assets
        logger.info(">>> Preparing cybind assets...")
        prepare_cybind_assets(cybind_dir, target)
        logger.info("")

        # Stage input headers. For nccl_core we still pass nccl_include_dir directly
        # (it already has nccl.h plus its expanded variant); for nccl_ep we stage
        # nccl_ep.h + ep_enums.h + nccl.h into a single dir.
        if target.header_files:
            input_dir = stage_input_headers(target, nccl_include_dir, temp_dir / "input")
        else:
            input_dir = nccl_include_dir

        # Pre-flight: cybind parses each entry header from input_dir. Failing
        # here gives a clearer message than a cpp "No such file or directory"
        # buried inside cybind's subprocess output.
        for hdr in target.entry_headers:
            if not (input_dir / hdr).exists():
                hint = (
                    " Build NCCL first (e.g. `make -j src.build`) to generate it."
                    if hdr.endswith("_expanded.h")
                    else " Re-check --nccl-include-dir or the contrib/ source tree."
                )
                raise FileNotFoundError(
                    f"Required cybind input header '{hdr}' not found in {input_dir}.{hint}"
                )

        # Step 3: Run cybind
        logger.info(">>> Running cybind...")
        cybind_output_dir = temp_dir / "cybind_output"
        run_cybind(cybind_dir, target, input_dir, cybind_output_dir)
        logger.info("")

        # Step 4: Copy bindings to package
        logger.info(">>> Copying bindings to package...")
        copy_bindings(cybind_output_dir, target)
        logger.info("")

    logger.warning("=" * 60)
    logger.warning(f"SUCCESS: Cython bindings for {target.name} generated successfully!")
    logger.warning(f"Bindings location: {target.output_dir}")
    logger.warning("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Cython bindings for NCCL using cybind")
    parser.add_argument(
        "--target",
        choices=[*TARGETS.keys(), "all"],
        default="all",
        help="Which binding to generate (default: all).",
    )
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=None,
        help="Path to CUDA installation (default: $CUDA_PATH or $CUDA_HOME)",
    )
    parser.add_argument(
        "--nccl-include-dir",
        type=Path,
        default=REPO_ROOT / "build" / "include",
        help="Directory containing nccl.h (default: <repo>/build/include).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    try:
        # CUDA_PATH (cybind uses it to find cuda.h)
        if args.cuda_home:
            if not args.cuda_home.exists():
                logger.error(f"CUDA home not found at {args.cuda_home}")
                return 1
            cuda_path = str(args.cuda_home)
        else:
            cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
            if not cuda_path:
                logger.error("Provide --cuda-home or set CUDA_PATH or CUDA_HOME")
                return 1
            if not Path(cuda_path).is_dir():
                logger.error(f"CUDA path is not a directory: {cuda_path}")
                return 1

        os.environ["CUDA_PATH"] = cuda_path
        logger.info(f"CUDA: {cuda_path}")
        logger.info(f"NCCL header: {args.nccl_include_dir / 'nccl.h'}")
        logger.info("")

        selected = list(TARGETS.values()) if args.target == "all" else [TARGETS[args.target]]
        for target in selected:
            generate_target(target, args.nccl_include_dir)

        return 0
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
