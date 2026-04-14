#!/bin/bash
#
# Entrypoint for pipeline_ctl.
# Sets up a Python venv, installs dependencies, and runs the tool.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
PIPELINE_CTL="${SCRIPT_DIR}/pipeline_ctl.py"

if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "Error: requirements.txt not found at ${REQUIREMENTS}"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

if ! python3 -c "import requests; from simple_term_menu import TerminalMenu" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r "$REQUIREMENTS"
fi

exec python3 "$PIPELINE_CTL" "$@"
