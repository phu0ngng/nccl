#!/bin/bash
#
# Updates the end year in SPDX copyright headers across the repository.
#
# Usage:
#   ./update_copyright_year.sh [new_year]
#
# If no year is provided, the current year is used.
#
# Examples:
#   ./update_copyright_year.sh        # uses current year
#   ./update_copyright_year.sh 2027   # explicitly set to 2027
#

set -euo pipefail

NEW_YEAR="${1:-$(date +%Y)}"

# Validate year
if ! [[ "$NEW_YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "Error: Argument must be a 4-digit year"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Updating copyright end year -> $NEW_YEAR"
echo "Repository: $REPO_ROOT"
echo ""

# Directories to exclude
EXCLUDE_DIRS="doca-gpunetio|nvtx3|\.git|build|__pycache__|node_modules"

# Find files with any NVIDIA copyright that has a year range (YYYY-YYYY)
# where the end year is NOT already the new year
COUNT=0
while IFS= read -r file; do
    # Skip excluded directories
    if echo "$file" | grep -qE "($EXCLUDE_DIRS)"; then
        continue
    fi

    sed -i'' -e "s/Copyright (c) \([0-9]\{4\}\)-[0-9]\{4\} NVIDIA CORPORATION/Copyright (c) \1-${NEW_YEAR} NVIDIA CORPORATION/g" "$file"
    echo "  Updated: $file"
    COUNT=$((COUNT + 1))
done < <(grep -rl "Copyright (c) [0-9]\{4\}-[0-9]\{4\} NVIDIA CORPORATION" "$REPO_ROOT" \
    --include='*.cc' --include='*.c' --include='*.cu' --include='*.h' \
    --include='*.hpp' --include='*.cpp' --include='*.cuh' \
    --include='*.py' --include='*.pyx' --include='*.pxd' \
    --include='*.sh' --include='*.mk' \
    --include='Makefile' --include='*.h.in' --include='*.sh.in' \
    2>/dev/null | grep -vE "($EXCLUDE_DIRS)" || true)

echo ""
echo "Done. Updated $COUNT files."
