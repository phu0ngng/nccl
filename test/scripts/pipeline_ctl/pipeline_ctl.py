#!/usr/bin/env python3

import json
import os
import subprocess
import sys

try:
    import requests
    from simple_term_menu import TerminalMenu
except ImportError:
    print(
        "Missing dependencies. Install them with:\n"
        "  pip install simple-term-menu requests\n"
        "Or use the requirements.txt in this directory:\n"
        "  pip install -r test/scripts/requirements.txt"
    )
    sys.exit(1)


GITLAB_URL = "https://gitlab-master.nvidia.com/api/v4/projects"
GITLAB_PROJECT_ID = "34033"

TESTS = [
    ("Unit Tests", "RUN_UNIT_TESTS"),
    ("API Tests", "RUN_API_TESTS"),
    ("Perf Single", "RUN_PERF_TESTS_SINGLE"),
    ("Perf Multi", "RUN_PERF_TESTS_MULTIPLE"),
    ("Perf Regression", "RUN_PERF_REGRESSION"),
    ("Resiliency", "RUN_RESILIENCY_TESTS"),
    ("DLFW", "RUN_DLFW_TESTS"),
    ("Profiler", "RUN_PROFILER_TESTS"),
    ("Inspector", "RUN_INSPECTOR_TESTS"),
    ("NCCL EP", "RUN_NCCL_EP_CI"),
    ("NCCL4PY", "RUN_NCCL4PY_CI"),
    ("DeepEP", "RUN_DEEP_EP_CI"),
]

CLUSTERS = [
    ("IPP6", "RUN_ON_IPP6"),
    ("EOS", "RUN_ON_EOS"),
    ("Pre-Nyx", "RUN_ON_PRE_NYX"),
    ("Pre-Tyche", "RUN_ON_PRE_TYCHE"),
    ("Theia", "RUN_ON_THEIA"),
    ("AWS", "RUN_ON_AWS"),
    ("DGX Spark", "RUN_ON_DGXSPARK"),
    ("Draco OCI", "RUN_ON_DRACO_OCI"),
    ("BIA", "RUN_ON_BIA"),
    ("Polyphe", "RUN_ON_POLYPHE"),
    ("Windows", "RUN_ON_WINDOWS"),
    ("P100 (GC)", "RUN_ON_P100"),
    ("P40 (GC)", "RUN_ON_P40"),
    ("GP100 (GC)", "RUN_ON_GP100"),
]

PRESET_MODES = ["Custom", "Pre-submit", "Post-submit", "Nightly", "Weekly"]
PRESET_VALUES = ["custom", "pre-submit", "post-submit", "nightly", "weekly"]

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"


def get_current_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "main"


def prompt_branch() -> str | None:
    default = get_current_branch()
    print(f"\n{BOLD}Branch{RESET} (Enter for {CYAN}{default}{RESET}): ", end="", flush=True)
    value = input().strip()
    return value if value else default


def prompt_mode() -> str | None:
    print(f"\n{BOLD}Pipeline Mode{RESET}  (↑/↓ navigate, Enter select, q quit)")
    menu = TerminalMenu(
        PRESET_MODES,
        title="",
        menu_cursor_style=("fg_cyan", "bold"),
    )
    idx = menu.show()
    if idx is None:
        return None
    return PRESET_VALUES[idx]


def prompt_multi(title: str, items: list[tuple[str, str]]) -> list[str] | None:
    """Multi-select menu. Returns list of selected CI variable names, or None on quit."""
    ALL_LABEL = "── All ──"
    labels = [ALL_LABEL] + [label for label, _ in items]

    print(f"\n{BOLD}{title}{RESET}  (Space toggle, Enter confirm, q quit)")
    menu = TerminalMenu(
        labels,
        title="",
        multi_select=True,
        show_multi_select_hint=False,
        multi_select_select_on_accept=False,
        multi_select_empty_ok=True,
        menu_cursor_style=("fg_cyan", "bold"),
    )
    indices = menu.show()
    if indices is None:
        return None
    if isinstance(indices, int):
        indices = (indices,)

    if 0 in indices:
        return [var for _, var in items]

    return [items[i - 1][1] for i in indices if i >= 1]


def build_variables(mode: str, selected_tests: list[str], selected_clusters: list[str]) -> list[dict[str, str]]:
    variables = []
    if mode != "custom":
        variables.append({"key": "TRIGGER_PIPELINE", "value": mode})
    else:
        variables.append({"key": "TRIGGER_PIPELINE", "value": "custom"})
        for _, var in TESTS:
            variables.append({"key": var, "value": "1" if var in selected_tests else "0"})
        for _, var in CLUSTERS:
            variables.append({"key": var, "value": "1" if var in selected_clusters else "0"})
    return variables


def print_summary(branch: str, mode: str, variables: list[dict[str, str]]) -> None:
    print(f"\n{'─' * 50}")
    print(f"{BOLD}Pipeline Summary{RESET}")
    print(f"{'─' * 50}")
    print(f"  Branch:  {CYAN}{branch}{RESET}")
    print(f"  Mode:    {CYAN}{mode}{RESET}")
    if mode == "custom":
        enabled_tests = [v["key"] for v in variables if v["value"] == "1" and not v["key"].startswith("RUN_ON_")]
        enabled_clusters = [v["key"] for v in variables if v["value"] == "1" and v["key"].startswith("RUN_ON_")]
        print(f"  Tests:   {', '.join(enabled_tests) if enabled_tests else '(none)'}")
        print(f"  Clusters:{' ' + ', '.join(enabled_clusters) if enabled_clusters else ' (none)'}")
    else:
        print(f"  Config:  TRIGGER_PIPELINE={mode}")
    print(f"{'─' * 50}")


def prompt_confirm() -> str | None:
    actions = ["Launch pipeline", "Dry run (show payload)", "Back to start", "Quit"]
    menu = TerminalMenu(
        actions,
        title="",
        menu_cursor_style=("fg_cyan", "bold"),
    )
    idx = menu.show()
    if idx is None or idx == 3:
        return None
    return ["launch", "dry-run", "restart"][idx]


def launch_pipeline(token: str, payload: dict) -> None:
    url = f"{GITLAB_URL}/{GITLAB_PROJECT_ID}/pipeline"
    try:
        resp = requests.post(
            url,
            headers={
                "PRIVATE-TOKEN": token,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        data = resp.json()
        if resp.ok:
            web_url = data.get("web_url", "")
            pid = data.get("id", "?")
            link = f"\033]8;;{web_url}\033\\{web_url}\033]8;;\033\\" if web_url else ""
            print(f"\n{GREEN}Pipeline #{pid} launched!{RESET}")
            print(f"  {link}")
        else:
            msg = data.get("message", {})
            print(f"\n{RED}GitLab API error ({resp.status_code}):{RESET}")
            print(json.dumps(msg, indent=2))
    except requests.RequestException as exc:
        print(f"\n{RED}Request failed:{RESET} {exc}")


def main() -> None:
    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}  NCCL Pipeline Control{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

    while True:
        branch = prompt_branch()
        if branch is None:
            return

        mode = prompt_mode()
        if mode is None:
            return

        selected_tests: list[str] = []
        selected_clusters: list[str] = []

        if mode == "custom":
            result = prompt_multi("Select Tests", TESTS)
            if result is None:
                return
            selected_tests = result

            result = prompt_multi("Select Clusters", CLUSTERS)
            if result is None:
                return
            selected_clusters = result

            if not selected_tests and not selected_clusters:
                print(f"\n{YELLOW}Warning: no tests or clusters selected.{RESET}")

        variables = build_variables(mode, selected_tests, selected_clusters)
        payload = {"ref": branch, "variables": variables}

        print_summary(branch, mode, variables)
        action = prompt_confirm()

        if action is None:
            return
        elif action == "restart":
            continue
        elif action == "dry-run":
            print(f"\n{BOLD}Dry run payload:{RESET}")
            print(json.dumps(payload, indent=2))
            return
        elif action == "launch":
            token = os.environ.get("GITLAB_API_TOKEN", "")
            if not token:
                print(f"\n{RED}Error:{RESET} Set the GITLAB_API_TOKEN environment variable")
                sys.exit(1)
            launch_pipeline(token, payload)
            return


if __name__ == "__main__":
    main()
