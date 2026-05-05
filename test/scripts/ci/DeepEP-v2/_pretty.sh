# Pretty-print helpers for DeepEP-v2 CI scripts.
#
# Provides: banner, section_start/section_end (paired), info, pass, fail,
# summary <total> [failed_test_name ...].
#
# In GitLab CI, section_start/end also emit collapsible-section markers so
# each section becomes an expandable group in the web UI.

# ANSI colors on a TTY or in GitLab CI; off when redirected to plain logs.
if [ -t 1 ] || [ -n "${GITLAB_CI:-}" ]; then
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    RED=$'\033[31m'
    GREEN=$'\033[32m'
    CYAN=$'\033[36m'
    RESET=$'\033[0m'
else
    BOLD='' DIM='' RED='' GREEN='' CYAN='' RESET=''
fi

_RULE='======================================================================'
_SECTION_NAME=""
_SECTION_TITLE=""

# `printf --` terminates options — defensive against any format string that
# could start with a dash if colors are empty.

banner() {
    echo
    printf -- "${CYAN}${BOLD}%s${RESET}\n" "$_RULE"
    printf -- "${CYAN}${BOLD}  %s${RESET}\n" "$1"
    printf -- "${CYAN}${BOLD}%s${RESET}\n" "$_RULE"
    echo
}

_slugify() {
    echo "$1" | tr -c 'A-Za-z0-9_' '_' | sed -E 's/_+/_/g; s/_$//; s/^_//'
}

section_start() {
    local title="$1"
    local name; name=$(_slugify "$title")
    _SECTION_NAME="$name"
    _SECTION_TITLE="$title"
    echo
    # Top rule sits OUTSIDE the section so it stays visible when collapsed.
    printf -- "${CYAN}${BOLD}%s${RESET}\n" "$_RULE"
    # Title line carries the section_start marker — this is the toggle.
    [ -n "${GITLAB_CI:-}" ] && printf '\e[0Ksection_start:%s:%s\r\e[0K' "$(date +%s)" "$name"
    printf -- "${CYAN}${BOLD}  ▶ START | %s${RESET}\n" "$title"
    printf -- "${CYAN}${BOLD}%s${RESET}\n" "$_RULE"
    echo
}

section_end() {
    local title="${_SECTION_TITLE:-section}"
    local name="${_SECTION_NAME:-section}"
    echo
    printf -- "${CYAN}${BOLD}%s${RESET}\n" "$_RULE"
    printf -- "${CYAN}${BOLD}  ◀ END   | %s${RESET}\n" "$title"
    printf -- "${CYAN}${BOLD}%s${RESET}\n" "$_RULE"
    [ -n "${GITLAB_CI:-}" ] && printf '\e[0Ksection_end:%s:%s\r\e[0K' "$(date +%s)" "$name"
    _SECTION_NAME=""
    _SECTION_TITLE=""
}

info() { printf -- "${DIM}  %s${RESET}\n" "$1"; }
pass() { printf -- "${GREEN}${BOLD}[PASS]${RESET} %s\n" "$1"; }
fail() { printf -- "${RED}${BOLD}[FAIL]${RESET} %s\n" "$1"; }

# summary <total> [failed_test_name ...]
summary() {
    local total=$1
    shift
    local fails=$#
    echo
    if [ "$fails" -eq 0 ]; then
        printf -- "${GREEN}${BOLD}%s${RESET}\n" "$_RULE"
        printf -- "${GREEN}${BOLD}  ALL %d TESTS PASSED${RESET}\n" "$total"
        printf -- "${GREEN}${BOLD}%s${RESET}\n" "$_RULE"
    else
        printf -- "${RED}${BOLD}%s${RESET}\n" "$_RULE"
        printf -- "${RED}${BOLD}  %d / %d TESTS FAILED${RESET}\n" "$fails" "$total"
        local t
        for t in "$@"; do
            printf -- "${RED}    - %s${RESET}\n" "$t"
        done
        printf -- "${RED}${BOLD}%s${RESET}\n" "$_RULE"
    fi
    echo
}
