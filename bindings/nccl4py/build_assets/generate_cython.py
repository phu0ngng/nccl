#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

"""
Generate Cython bindings for NCCL using cybind.

This script clones cybind, configures it with NCCL-specific templates,
runs the code generator, and copies the generated bindings to the nccl4py
package directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# cybind repository configuration
CYBIND_COMMIT = "813db740e5ee440d0f21b4ee80546e26410f9de6"
CYBIND_SSH_URL = "ssh://git@gitlab-master.nvidia.com:12051/xiakunl/cybind.git"

# Global logger - will be configured in main()
logger = logging.getLogger(__name__)


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(command) is not None


def clone_cybind(cybind_dir: Path) -> Path:
    """
    Clone cybind repository to specified directory.

    Args:
        - cybind_dir (Path): Directory for cloned cybind repository.

    Returns:
        ``Path``: Path to cloned cybind directory.

    Raises:
        - ``RuntimeError``: If git is not available or clone fails.
    """
    if not check_command_exists("git"):
        raise RuntimeError("git command not found. Please install git.")

    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", CYBIND_SSH_URL, str(cybind_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Checkout specific commit
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


def prepare_cybind_assets(
    cybind_dir: Path,
    nccl4py_assets_dir: Path,
) -> None:
    """
    Copy NCCL configuration and templates to cybind directory.

    Args:
        - cybind_dir (Path): Path to cybind repository.
        - nccl4py_assets_dir (Path): Path to nccl4py build_assets directory.

    Raises:
        - ``FileNotFoundError``: If required files don't exist.
    """
    # Copy config_nccl.py
    config_src = nccl4py_assets_dir / "cybind" / "config_nccl.py"
    config_dst = cybind_dir / "cybind" / "assets" / "configs" / "config_nccl.py"

    if not config_src.exists():
        raise FileNotFoundError(f"Config file not found: {config_src}")

    logger.info(f"Copying config: {config_src} -> {config_dst}")
    shutil.copy2(config_src, config_dst)

    # Copy templates
    templates_src = nccl4py_assets_dir / "cybind" / "templates"
    templates_dst = cybind_dir / "cybind" / "assets" / "templates" / "nccl4py" / "bindings"

    if not templates_src.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_src}")

    templates_dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying templates: {templates_src} -> {templates_dst}")
    shutil.copytree(templates_src, templates_dst, dirs_exist_ok=True)


def run_cybind(
    cybind_dir: Path,
    nccl_include_dir: Path,
    output_dir: Path,
) -> None:
    """
    Run cybind to generate NCCL bindings.

    Args:
        - cybind_dir (Path): Path to cybind repository.
        - nccl_include_dir (Path): Path to directory containing nccl.h.
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
        "nccl",
        "--input-dir",
        str(nccl_include_dir),
        "--output",
        str(output_dir),
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Working directory: {cybind_dir}")
    logger.info(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'not set')}")

    try:
        logger.info("Running cybind...")
        result = subprocess.run(
            cmd,
            cwd=cybind_dir,
            check=True,
            capture_output=True,
            text=True,
        )
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


def copy_bindings(
    cybind_output_dir: Path,
    nccl4py_assets_dir: Path,
    nccl4py_bindings_dir: Path,
) -> None:
    """
    Copy generated bindings to nccl4py package directory.

    Args:
        - cybind_output_dir (Path): Directory containing generated bindings.
        - nccl4py_assets_dir (Path): Path to nccl4py build_assets directory.
        - nccl4py_bindings_dir (Path): Target directory in nccl4py package.

    Raises:
        - ``FileNotFoundError``: If generated files don't exist.
    """
    # Generated bindings are in cybind_output_dir/nccl4py/bindings/
    generated_bindings = cybind_output_dir / "nccl4py" / "bindings"

    if not generated_bindings.exists():
        raise FileNotFoundError(f"Generated bindings not found: {generated_bindings}")

    # Clean target directory
    if nccl4py_bindings_dir.exists():
        logger.info(f"Removing target directory: {nccl4py_bindings_dir}")
        shutil.rmtree(nccl4py_bindings_dir)

    # Copy generated bindings from cybind
    logger.info(f"Copying generated bindings: {generated_bindings} -> {nccl4py_bindings_dir}")
    shutil.copytree(generated_bindings, nccl4py_bindings_dir)

    # Copy static files that cybind doesn't process
    templates_dir = nccl4py_assets_dir / "cybind" / "templates"

    # Copy __init__.py files
    logger.info("Copying static files from templates...")

    bindings_init_src = templates_dir / "__init__.py"
    bindings_init_dst = nccl4py_bindings_dir / "__init__.py"
    if bindings_init_src.exists():
        logger.info(f"  {bindings_init_src.name}")
        shutil.copy2(bindings_init_src, bindings_init_dst)

    internal_init_src = templates_dir / "_internal" / "__init__.py"
    internal_init_dst = nccl4py_bindings_dir / "_internal" / "__init__.py"
    if internal_init_src.exists():
        logger.info(f"  _internal/{internal_init_src.name}")
        shutil.copy2(internal_init_src, internal_init_dst)

    # Copy utils.* files
    utils_src_dir = templates_dir / "_internal"
    utils_files = list(utils_src_dir.glob("utils.*"))
    for utils_file in utils_files:
        utils_dst = nccl4py_bindings_dir / "_internal" / utils_file.name
        logger.info(f"  _internal/{utils_file.name}")
        shutil.copy2(utils_file, utils_dst)

    logger.info("Successfully copied all bindings")


def main() -> int:
    """
    Main entry point for generate_cython script.

    Returns:
        ``int``: Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="Generate Cython bindings for NCCL using cybind")
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=None,
        help="Path to CUDA installation (default: $CUDA_PATH or $CUDA_HOME environment variable)",
    )
    parser.add_argument(
        "--nccl-include-dir",
        type=Path,
        default=SCRIPT_DIR.parent.parent.parent / "build" / "include",
        help="Path to directory containing nccl.h (default: ../../build/include relative to script)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=SCRIPT_DIR.parent / "nccl" / "bindings",
        help="Output directory for generated bindings (default: ../nccl/bindings relative to script)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    try:
        # Use command-line args or defaults (already set to script-relative paths)
        nccl_include_dir = args.nccl_include_dir

        nccl_header = nccl_include_dir / "nccl.h"
        if not nccl_header.exists():
            logger.error(f"nccl.h not found at {nccl_header}")
            logger.error("Please build NCCL first or specify --nccl-include-dir")
            return 1

        logger.info(f"NCCL header: {nccl_header}")

        # Determine CUDA path and set CUDA_PATH environment variable.
        # CUDA_PATH is required by cybind to locate CUDA headers when parsing nccl.h
        # Checked in this order: --cuda-home, $CUDA_PATH, $CUDA_HOME.
        cuda_path = None
        if args.cuda_home:
            if not args.cuda_home.exists():
                logger.error(f"CUDA home not found at {args.cuda_home}")
                return 1
            cuda_path = str(args.cuda_home)
        else:
            cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
            if not cuda_path:
                logger.error(
                    "Provide --cuda-home or set CUDA_PATH or CUDA_HOME environment variable"
                )
                return 1
            if not Path(cuda_path).is_dir():
                logger.error(
                    f"CUDA_PATH/CUDA_HOME points to non-existent directory or not a directory: {cuda_path}"
                )
                return 1

        # Set CUDA_PATH in environment for cybind (needed to resolve CUDA types in nccl.h)
        os.environ["CUDA_PATH"] = cuda_path
        logger.info(f"CUDA: {cuda_path}")
        logger.info("")

        # Paths
        nccl4py_assets_dir = SCRIPT_DIR
        nccl4py_bindings_dir = args.output_dir

        logger.info(f"Assets directory: {nccl4py_assets_dir}")
        logger.info(f"Target bindings directory: {nccl4py_bindings_dir}")
        logger.info("")

        # Create temporary directory for all intermediate files
        with tempfile.TemporaryDirectory(prefix="nccl4py_cybind_") as temp_dir_str:
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
            prepare_cybind_assets(cybind_dir, nccl4py_assets_dir)
            logger.info("")

            # Step 3: Run cybind
            logger.info(">>> Running cybind...")
            cybind_output_dir = temp_dir / "cybind_output"
            run_cybind(cybind_dir, nccl_include_dir, cybind_output_dir)
            logger.info("")

            # Step 4: Copy bindings to package
            logger.info(">>> Copying bindings to package...")
            copy_bindings(cybind_output_dir, nccl4py_assets_dir, nccl4py_bindings_dir)
            logger.info("")

        logger.warning("=" * 60)
        logger.warning("SUCCESS: Cython bindings generated successfully!")
        logger.warning(f"Bindings location: {nccl4py_bindings_dir}")
        logger.warning("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
