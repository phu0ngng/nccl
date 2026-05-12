#!/bin/bash
#
# Report GitLab CI pipeline status to a GitHub commit as a status check.
#
# Usage: github-status.sh <state>
#   state: pending, success, failure, error
#
# Required environment variables:
#   GITHUB_SHA      - The commit SHA to report status on
#   GITHUB_REPO     - The GitHub repo in "owner/repo" format (e.g. NVIDIA/nccl)
#   GITHUB_TOKEN    - GitHub personal access token with repo:status scope
#
# Optional environment variables:
#   CI_PIPELINE_URL - GitLab pipeline URL (auto-provided by GitLab CI)

set -euo pipefail

STATE="${1:?Usage: github-status.sh <pending|success|failure|error>}"

if [[ -z "${GITHUB_SHA:-}" ]]; then
    echo "GITHUB_SHA not set, skipping GitHub status update"
    exit 0
fi

if [[ -z "${GITHUB_REPO:-}" ]]; then
    echo "ERROR: GITHUB_REPO is required (e.g. NVIDIA/nccl)"
    exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "ERROR: GITHUB_TOKEN is required to report status to GitHub"
    exit 1
fi

case "$STATE" in
    pending) DESCRIPTION="Internal CI pipeline is running..." ;;
    success) DESCRIPTION="Internal CI pipeline succeeded" ;;
    failure) DESCRIPTION="Internal CI pipeline failed" ;;
    error)   DESCRIPTION="Internal CI pipeline encountered an error" ;;
    *)
        echo "ERROR: Invalid state: $STATE (expected pending|success|failure|error)"
        exit 1
        ;;
esac

echo "Setting GitHub commit status to '${STATE}' on ${GITHUB_REPO}@${GITHUB_SHA:0:7}..."
curl -sf -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/repos/${GITHUB_REPO}/statuses/${GITHUB_SHA}" \
    -d @- <<EOF
{
    "state": "${STATE}",
    "target_url": "${CI_PIPELINE_URL:-}",
    "description": "${DESCRIPTION}",
    "context": "Internal CI"
}
EOF

echo "GitHub status '${STATE}' set on ${GITHUB_REPO}@${GITHUB_SHA:0:7}"
