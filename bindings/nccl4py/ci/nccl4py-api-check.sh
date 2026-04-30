#!/bin/bash
#
# nccl4py-api-check.sh
#
# Detects changes to NCCL public API headers in an MR and posts a comment
# on the merge request notifying nccl4py engineers.
#
# Required CI variables (PARENT_* are forwarded from the parent pipeline trigger
# because this job runs in a child pipeline and predefined CI_MERGE_REQUEST_*
# variables are not auto-populated when CI_PIPELINE_SOURCE is "parent_pipeline"):
#   PARENT_CI_MERGE_REQUEST_IID            - MR internal ID
#   PARENT_CI_MERGE_REQUEST_DIFF_BASE_SHA  - Merge base commit SHA
#   CI_API_V4_URL                          - GitLab API base URL
#   CI_PROJECT_ID                          - Project ID
#   CI_JOB_TOKEN                           - Ephemeral job token for API auth
#
# When the script is run outside a child pipeline (e.g. from the test suite),
# CI_MERGE_REQUEST_IID and CI_MERGE_REQUEST_DIFF_BASE_SHA are accepted as
# fallbacks.
#
# Optional CI variables:
#   NCCL4PY_REVIEWERS - Space-separated GitLab @mentions (e.g. "@leofang @user2")
#   DRY_RUN           - Set to "1" to print comment to stdout instead of posting
#

set -euo pipefail

MARKER="<!-- nccl4py-api-check -->"

MONITORED_PATHS=(
  "src/nccl.h.in"
  "src/include/nccl_common.h"
  "src/include/nccl_device.h"
  "src/include/nccl_device/"
)

GITLAB_API="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}"
MR_IID="${PARENT_CI_MERGE_REQUEST_IID:-${CI_MERGE_REQUEST_IID:-}}"
MERGE_BASE="${PARENT_CI_MERGE_REQUEST_DIFF_BASE_SHA:-${CI_MERGE_REQUEST_DIFF_BASE_SHA:-}}"

# ---------------------------------------------------------------------------
# Guard: ensure we're in an MR pipeline
# ---------------------------------------------------------------------------
if [[ -z "$MR_IID" ]]; then
  echo "Not an MR pipeline (PARENT_CI_MERGE_REQUEST_IID/CI_MERGE_REQUEST_IID is unset), skipping."
  exit 0
fi

if [[ -z "$MERGE_BASE" ]]; then
  echo "PARENT_CI_MERGE_REQUEST_DIFF_BASE_SHA/CI_MERGE_REQUEST_DIFF_BASE_SHA is unset, skipping."
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: Detect changed public header files
# ---------------------------------------------------------------------------
echo "Checking for public API header changes against merge base ${MERGE_BASE}..."

CHANGED_FILES=$(git diff --name-only "${MERGE_BASE}...HEAD" -- "${MONITORED_PATHS[@]}" || true)

if [[ -z "$CHANGED_FILES" ]]; then
  echo "No public API header changes detected."
  exit 0
fi

echo "Changed public headers:"
echo "$CHANGED_FILES"

# ---------------------------------------------------------------------------
# Step 2: Analyze the diff for specific API changes
# ---------------------------------------------------------------------------
FULL_DIFF=$(git diff "${MERGE_BASE}...HEAD" -- ${CHANGED_FILES})

# New function declarations (added lines with ncclResult_t ncclXxx or similar patterns)
NEW_FUNCS=$(echo "$FULL_DIFF" | grep -E '^\+.*(ncclResult_t|const char\*|void)[[:space:]]+nccl[A-Z]' | grep -vF '+++' | grep -vF 'pnccl' | sed -E 's/^\+.*[[:space:]](nccl[A-Za-z0-9_]+)[[:space:]]*\(.*/\1/' | sort -u || true)

# Removed function declarations
REMOVED_FUNCS=$(echo "$FULL_DIFF" | grep -E '^\-.*(ncclResult_t|const char\*|void)[[:space:]]+nccl[A-Z]' | grep -vF -- '---' | grep -vF 'pnccl' | sed -E 's/^\-.*[[:space:]](nccl[A-Za-z0-9_]+)[[:space:]]*\(.*/\1/' | sort -u || true)

# Struct-related changes (lines near typedef struct)
STRUCT_CHANGES=$(echo "$FULL_DIFF" | grep -E '^\+.*typedef struct|^\+.*\} nccl[A-Za-z_]+_t;' | grep -vF '+++' | sed -E 's/.*\} (nccl[A-Za-z_]+_t).*/\1/' | sort -u || true)

# Enum value changes
ENUM_CHANGES=$(echo "$FULL_DIFF" | grep -E '^\+[[:space:]]*nccl[A-Z][A-Za-z0-9_]*[[:space:]]*=' | grep -vF '+++' | sed -E 's/^\+[[:space:]]*(nccl[A-Za-z0-9_]+)[[:space:]]*=.*/\1/' | sort -u || true)

# Macro changes
MACRO_CHANGES=$(echo "$FULL_DIFF" | grep -E '^\+#define NCCL_' | grep -vF '+++' | sed -E 's/^\+#define (NCCL_[A-Za-z0-9_]+).*/\1/' | sort -u || true)

# ---------------------------------------------------------------------------
# Step 3: Compose the comment body
# ---------------------------------------------------------------------------

# Build changed files list
FILES_LIST=""
while IFS= read -r f; do
  [[ -n "$f" ]] && FILES_LIST="${FILES_LIST}\n- \`${f}\`"
done <<< "$CHANGED_FILES"

# Build summary sections
SUMMARY=""

if [[ -n "$NEW_FUNCS" ]]; then
  SUMMARY="${SUMMARY}\n\n**New function declarations:**"
  COUNT=0
  while IFS= read -r fn; do
    if [[ -n "$fn" ]] && (( COUNT < 20 )); then
      SUMMARY="${SUMMARY}\n- \`${fn}\`"
      COUNT=$((COUNT + 1))
    fi
  done <<< "$NEW_FUNCS"
  TOTAL=$(echo "$NEW_FUNCS" | grep -c '.' || true)
  if (( TOTAL > 20 )); then
    SUMMARY="${SUMMARY}\n- ... and $((TOTAL - 20)) more"
  fi
fi

if [[ -n "$REMOVED_FUNCS" ]]; then
  SUMMARY="${SUMMARY}\n\n**Removed function declarations:**"
  COUNT=0
  while IFS= read -r fn; do
    if [[ -n "$fn" ]] && (( COUNT < 20 )); then
      SUMMARY="${SUMMARY}\n- \`${fn}\`"
      COUNT=$((COUNT + 1))
    fi
  done <<< "$REMOVED_FUNCS"
  TOTAL=$(echo "$REMOVED_FUNCS" | grep -c '.' || true)
  if (( TOTAL > 20 )); then
    SUMMARY="${SUMMARY}\n- ... and $((TOTAL - 20)) more"
  fi
fi

if [[ -n "$STRUCT_CHANGES" ]]; then
  SUMMARY="${SUMMARY}\n\n**Modified/new structures:**"
  while IFS= read -r s; do
    [[ -n "$s" ]] && SUMMARY="${SUMMARY}\n- \`${s}\`"
  done <<< "$STRUCT_CHANGES"
fi

if [[ -n "$ENUM_CHANGES" ]]; then
  SUMMARY="${SUMMARY}\n\n**New/modified enum values:**"
  COUNT=0
  while IFS= read -r e; do
    if [[ -n "$e" ]] && (( COUNT < 20 )); then
      SUMMARY="${SUMMARY}\n- \`${e}\`"
      COUNT=$((COUNT + 1))
    fi
  done <<< "$ENUM_CHANGES"
  TOTAL=$(echo "$ENUM_CHANGES" | grep -c '.' || true)
  if (( TOTAL > 20 )); then
    SUMMARY="${SUMMARY}\n- ... and $((TOTAL - 20)) more"
  fi
fi

if [[ -n "$MACRO_CHANGES" ]]; then
  SUMMARY="${SUMMARY}\n\n**Changed macros:**"
  while IFS= read -r m; do
    [[ -n "$m" ]] && SUMMARY="${SUMMARY}\n- \`${m}\`"
  done <<< "$MACRO_CHANGES"
fi

# Fallback if regex didn't capture specifics but files did change
if [[ -z "$SUMMARY" ]]; then
  SUMMARY="\n\nDetailed change categorization not available -- please review the changed files manually."
fi

# Reviewer mentions
REVIEWERS="${NCCL4PY_REVIEWERS:-}"
REVIEWER_LINE=""
if [[ -n "$REVIEWERS" ]]; then
  REVIEWER_LINE="\n\n**cc:** ${REVIEWERS}"
else
  echo "WARNING: NCCL4PY_REVIEWERS is not set, posting comment without @mentions."
fi

# Pipeline link
PIPELINE_LINK="${CI_PIPELINE_URL:-}"

# Assemble full comment
COMMENT_BODY="${MARKER}
## :warning: NCCL Public API Change Detected

This MR modifies NCCL public headers that may require **nccl4py** binding updates.$(echo -e "$REVIEWER_LINE")

### Changed Files
$(echo -e "$FILES_LIST")

### Summary of Changes
$(echo -e "$SUMMARY")

### Required Actions
| Change Type | Action |
|---|---|
| New APIs | Update \`config_nccl.py\` function list + regenerate Cython |
| Structure changes | Regenerate Cython code |
| New public structures | May need to add to \`types\` dict in \`config_nccl.py\` |

---
*Auto-generated by \`nccl4py-api-check\` CI job.${PIPELINE_LINK:+ [Pipeline](${PIPELINE_LINK})}*"

# ---------------------------------------------------------------------------
# Step 4: Dry-run or post the comment
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "=== DRY RUN: Comment that would be posted ==="
  echo "$COMMENT_BODY"
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 5: Check for existing comment (idempotency)
# ---------------------------------------------------------------------------
echo "Checking for existing nccl4py-api-check comment on MR !${MR_IID}..."

EXISTING_NOTE_ID=""
PAGE=1
while true; do
  NOTES_RESPONSE=$(curl --silent --fail --header "JOB-TOKEN: $CI_JOB_TOKEN" \
    "${GITLAB_API}/merge_requests/${MR_IID}/notes?per_page=100&page=${PAGE}" 2>&1) || {
    echo "WARNING: Failed to fetch MR notes (page ${PAGE}). Will attempt to post a new comment."
    break
  }

  # Check if response contains our marker
  NOTE_ID=$(echo "$NOTES_RESPONSE" | python3 -c "
import sys, json
try:
    notes = json.load(sys.stdin)
    for note in notes:
        if '${MARKER}' in note.get('body', ''):
            print(note['id'])
            break
except Exception:
    pass
" 2>/dev/null || true)

  if [[ -n "$NOTE_ID" ]]; then
    EXISTING_NOTE_ID="$NOTE_ID"
    echo "Found existing comment (note ID: ${EXISTING_NOTE_ID})."
    break
  fi

  # Check if there are more pages
  COUNT=$(echo "$NOTES_RESPONSE" | python3 -c "
import sys, json
try:
    print(len(json.load(sys.stdin)))
except Exception:
    print(0)
" 2>/dev/null || echo "0")

  if (( COUNT < 100 )); then
    break
  fi
  ((PAGE++))
done

# ---------------------------------------------------------------------------
# Step 6: Post or update the comment
# ---------------------------------------------------------------------------
JSON_PAYLOAD=$(jq -n --arg body "$COMMENT_BODY" '{body: $body}')

if [[ -n "$EXISTING_NOTE_ID" ]]; then
  echo "Updating existing comment (note ID: ${EXISTING_NOTE_ID})..."
  HTTP_CODE=$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --request PUT \
    --header "JOB-TOKEN: $CI_JOB_TOKEN" \
    --header "Content-Type: application/json" \
    --data "$JSON_PAYLOAD" \
    "${GITLAB_API}/merge_requests/${MR_IID}/notes/${EXISTING_NOTE_ID}") || true

  if [[ "$HTTP_CODE" =~ ^2 ]]; then
    echo "Successfully updated comment on MR !${MR_IID}."
  else
    echo "WARNING: Failed to update comment (HTTP ${HTTP_CODE}). Printing comment to job log as fallback."
    echo "$COMMENT_BODY"
  fi
else
  echo "Posting new comment on MR !${MR_IID}..."
  HTTP_CODE=$(curl --silent --output /dev/null --write-out "%{http_code}" \
    --request POST \
    --header "JOB-TOKEN: $CI_JOB_TOKEN" \
    --header "Content-Type: application/json" \
    --data "$JSON_PAYLOAD" \
    "${GITLAB_API}/merge_requests/${MR_IID}/notes") || true

  if [[ "$HTTP_CODE" =~ ^2 ]]; then
    echo "Successfully posted comment on MR !${MR_IID}."
  else
    echo "WARNING: Failed to post comment (HTTP ${HTTP_CODE}). Printing comment to job log as fallback."
    echo "$COMMENT_BODY"
  fi
fi
