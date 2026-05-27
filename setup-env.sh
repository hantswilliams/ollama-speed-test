#!/usr/bin/env bash
# -------------------------------------------------------------------
# setup-env.sh — Bidirectional sync between local .env and 1Password
#
# Modes:
#   ./setup-env.sh             # default: diff + offer to push local-only keys
#   ./setup-env.sh --pull      # force overwrite .env from 1Password
#   ./setup-env.sh --push      # push local-only keys to 1Password + .env.template
#   ./setup-env.sh --status    # diff only, no changes, no prompts
#   ./setup-env.sh --dry-run … # combine with any mode: show what would happen
#
# Prerequisites:
#   - 1Password CLI installed: https://developer.1password.com/docs/cli
#   - Signed in: op signin
#
# What this script does NOT do:
#   - Never silently overwrites a local .env that has extra keys
#   - Never removes keys from 1Password (only adds or updates)
#   - Never pushes empty-value local keys (skips placeholders like
#     `KEY=your-key-here` that haven't been filled in yet)
# -------------------------------------------------------------------

set -euo pipefail

VAULT="Developer Projects"
ITEM="ollama-llama-speed-test-env"
ENV_FILE=".env"
ENV_TEMPLATE=".env.example"

MODE="default"
DRY_RUN="no"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pull)    MODE="pull";   shift ;;
        --push)    MODE="push";   shift ;;
        --status)  MODE="status"; shift ;;
        --dry-run) DRY_RUN="yes"; shift ;;
        -h|--help)
            head -25 "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--pull|--push|--status] [--dry-run]" >&2
            exit 2
            ;;
    esac
done

# ----- Preconditions -----
if ! command -v op &> /dev/null; then
    echo "Error: 1Password CLI (op) is not installed." >&2
    echo "Install it: https://developer.1password.com/docs/cli" >&2
    exit 1
fi
if ! op account list &> /dev/null; then
    echo "Error: Not signed in to 1Password. Run: op signin" >&2
    exit 1
fi
if ! op item get "$ITEM" --vault "$VAULT" &> /dev/null; then
    cat >&2 <<EOF
Error: Item '$ITEM' not found in vault '$VAULT'.

To create it manually, run:
  op item create --category='Secure Note' \\
    --vault='$VAULT' \\
    --title='$ITEM' \\
    'HF_TOKEN=your-hf-token-here'

Or add it via the 1Password app, then re-run this script.
EOF
    exit 1
fi

# ----- The Python helper does the heavy lifting -----
# Embedded so this script stays single-file (no scripts/ dependency)
# and runs against the system python3 (no .venv required for the first
# pass on a fresh machine).
python3 - "$MODE" "$DRY_RUN" "$VAULT" "$ITEM" "$ENV_FILE" "$ENV_TEMPLATE" <<'PYEOF'
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

mode, dry_run_str, vault, item, env_file_arg, env_template_arg = sys.argv[1:7]
DRY_RUN = dry_run_str == "yes"
ENV_FILE = Path(env_file_arg)
ENV_TEMPLATE = Path(env_template_arg)

# Values that look like unset placeholders — we never push these
PLACEHOLDER_RE = re.compile(r"^your-.*-here$", re.IGNORECASE)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def info(msg: str) -> None:
    print(f"  ℹ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def err(msg: str) -> None:
    print(f"  ✗ {msg}", file=sys.stderr)


# ---------- .env parsing ----------
ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE, skipping comments + blanks.
    Handles surrounding single/double quotes; preserves inner content as-is.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = ENV_LINE_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        # Strip surrounding quotes if balanced
        if (len(val) >= 2
                and val[0] == val[-1]
                and val[0] in ("'", '"')):
            val = val[1:-1]
        out[key] = val
    return out


# ---------- 1Password I/O ----------
def fetch_1password_fields() -> dict[str, str]:
    """Return {LABEL: VALUE} for all labeled fields in the 1Password item."""
    proc = subprocess.run(
        ["op", "item", "get", item, "--vault", vault, "--format=json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(proc.stdout)
    skip_labels = {"notesPlain", "password"}
    out: dict[str, str] = {}
    for field in data.get("fields", []):
        label = field.get("label", "")
        value = field.get("value", "")
        if not label or label in skip_labels:
            continue
        out[label] = value or ""
    return out


def push_to_1password(updates: dict[str, str]) -> None:
    """Add / update fields in the 1Password item.

    op item edit can add new text fields with `LABEL[text]=VALUE` or
    update existing ones with just `LABEL=VALUE`. We use the explicit
    `[text]` annotation for new fields and the bare form for updates.
    """
    if not updates:
        return
    existing = fetch_1password_fields()
    args = ["op", "item", "edit", item, "--vault", vault]
    for k, v in updates.items():
        if k in existing:
            args.append(f"{k}={v}")
        else:
            args.append(f"{k}[text]={v}")
    if DRY_RUN:
        info(f"DRY-RUN: would run: op item edit {item} --vault '{vault}' "
             + " ".join(f"{k}=<value>" for k in updates))
        return
    subprocess.run(args, check=True, capture_output=True)


# ---------- .env.template updates ----------
def update_template(new_keys: list[str]) -> bool:
    """Append placeholders for new keys to .env.template. Idempotent.
    Returns True if the file was modified.
    """
    if not new_keys:
        return False
    existing_text = ENV_TEMPLATE.read_text(encoding="utf-8") if ENV_TEMPLATE.exists() else ""
    existing_keys: set[str] = set()
    for line in existing_text.splitlines():
        m = ENV_LINE_RE.match(line.strip().lstrip("#").strip())
        if m:
            existing_keys.add(m.group(1))
    to_add = [k for k in new_keys if k not in existing_keys]
    if not to_add:
        return False
    if DRY_RUN:
        info(f"DRY-RUN: would append {len(to_add)} placeholder(s) to {ENV_TEMPLATE}:")
        for k in to_add:
            info(f"           {k}={placeholder_for(k)}")
        return False
    today = date.today().isoformat()
    parts: list[str] = []
    if existing_text and not existing_text.endswith("\n"):
        parts.append("")
    parts.append("")
    parts.append(f"# Auto-added by setup-env.sh --push on {today}")
    for k in to_add:
        parts.append(f"{k}={placeholder_for(k)}")
    parts.append("")  # trailing newline
    with ENV_TEMPLATE.open("a", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return True


def placeholder_for(key: str) -> str:
    """Generate a `your-NAME-here` style placeholder for a key."""
    fragment = key.lower().replace("_", "-")
    return f"your-{fragment}-here"


# ---------- pull (overwrite .env from 1Password) ----------
def write_env_from_remote(remote: dict[str, str]) -> None:
    """Write .env with the canonical 1Password-sourced content."""
    if DRY_RUN:
        info(f"DRY-RUN: would write {ENV_FILE} with {len(remote)} key(s):")
        for k in sorted(remote):
            info(f"           {k}=<value>")
        return
    lines = [
        "# Ollama / Llama-server Speed Test — Environment Variables",
        f"# Synced from 1Password ('{item}' in '{vault}') by setup-env.sh",
        f"# Last pull: {date.today().isoformat()}",
        "# Manual edits are OK — run `./setup-env.sh --push` to upstream them.",
        "",
    ]
    for k in sorted(remote):
        lines.append(f"{k}={remote[k]}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- Diff helper ----------
def compute_diff(local: dict[str, str], remote: dict[str, str]) -> dict[str, list[str]]:
    """Categorize keys into in-sync / local-only / remote-only / value-differs."""
    in_sync: list[str] = []
    local_only: list[str] = []
    remote_only: list[str] = []
    differs: list[str] = []
    for k, v in local.items():
        if k not in remote:
            local_only.append(k)
        elif v != remote[k]:
            differs.append(k)
        else:
            in_sync.append(k)
    for k in remote:
        if k not in local:
            remote_only.append(k)
    # Skip placeholder values from local_only (haven't been filled in yet)
    skipped_placeholders = [k for k in local_only if PLACEHOLDER_RE.match(local.get(k, ""))]
    local_only = [k for k in local_only if k not in skipped_placeholders]
    return {
        "in_sync": sorted(in_sync),
        "local_only": sorted(local_only),
        "remote_only": sorted(remote_only),
        "differs": sorted(differs),
        "skipped_placeholders": sorted(skipped_placeholders),
    }


def print_diff(diff: dict[str, list[str]]) -> None:
    print(f"Comparing local {ENV_FILE} vs 1Password item '{item}' in vault '{vault}':")
    print()
    if diff["in_sync"]:
        print("  In sync:")
        for k in diff["in_sync"]:
            print(f"    = {k}")
        print()
    if diff["local_only"]:
        print("  In .env but missing from 1Password (would be uploaded by push):")
        for k in diff["local_only"]:
            print(f"    ↑ {k}")
        print()
    if diff["remote_only"]:
        print("  In 1Password but missing from .env (would be added by pull):")
        for k in diff["remote_only"]:
            print(f"    ↓ {k}")
        print()
    if diff["differs"]:
        print("  Values differ between local and 1Password:")
        for k in diff["differs"]:
            print(f"    ≠ {k}")
        print()
    if diff["skipped_placeholders"]:
        print("  Local keys still set to placeholder values (skipped):")
        for k in diff["skipped_placeholders"]:
            print(f"    ⊘ {k}")
        print()
    if not any([diff["local_only"], diff["remote_only"], diff["differs"]]):
        print("  ✓ Everything in sync.")
        print()


# ---------- prompt helper ----------
def confirm(prompt: str, default_yes: bool = False) -> bool:
    suffix = " [Y/n] " if default_yes else " [y/N] "
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        return False
    if not ans:
        return default_yes
    return ans in ("y", "yes")


# ---------- modes ----------
def do_status() -> int:
    local = parse_env_file(ENV_FILE)
    remote = fetch_1password_fields()
    diff = compute_diff(local, remote)
    print_diff(diff)
    return 0


def do_pull() -> int:
    local = parse_env_file(ENV_FILE)
    remote = fetch_1password_fields()
    diff = compute_diff(local, remote)

    if diff["local_only"]:
        warn(f"{len(diff['local_only'])} local-only key(s) would be LOST by pull:")
        for k in diff["local_only"]:
            print(f"     ↑ {k}")
        if not confirm("Pull anyway (will overwrite .env)?", default_yes=False):
            err("Aborted.")
            return 1
    write_env_from_remote(remote)
    ok(f"Wrote {ENV_FILE} with {len(remote)} key(s) from 1Password" + (" (dry-run)" if DRY_RUN else ""))
    return 0


def do_push() -> int:
    local = parse_env_file(ENV_FILE)
    remote = fetch_1password_fields()
    diff = compute_diff(local, remote)

    to_push = {k: local[k] for k in diff["local_only"] if local[k]}
    if not to_push and not diff["differs"]:
        ok("No new local keys to push. .env and 1Password are in sync.")
        return 0

    if to_push:
        info(f"Will push {len(to_push)} new key(s) to 1Password:")
        for k in sorted(to_push):
            print(f"     ↑ {k}")
    if diff["differs"]:
        info(f"Values differ on {len(diff['differs'])} key(s) — left as-is "
             "(use --pull to override local, or edit 1Password manually):")
        for k in diff["differs"]:
            print(f"     ≠ {k}")

    if to_push:
        push_to_1password(to_push)
        if not DRY_RUN:
            ok(f"Pushed {len(to_push)} key(s) to 1Password")
        if update_template(list(to_push)):
            ok(f"Updated {ENV_TEMPLATE} with placeholder(s)")
        elif not DRY_RUN:
            info(f"{ENV_TEMPLATE} already lists all pushed keys — no changes")
    return 0


def do_default() -> int:
    """Smart default — pull if .env missing; otherwise diff + offer push."""
    if not ENV_FILE.exists():
        info(f"No local {ENV_FILE} found — pulling fresh from 1Password...")
        remote = fetch_1password_fields()
        write_env_from_remote(remote)
        ok(f"Wrote {ENV_FILE} with {len(remote)} key(s)" + (" (dry-run)" if DRY_RUN else ""))
        return 0

    local = parse_env_file(ENV_FILE)
    remote = fetch_1password_fields()
    diff = compute_diff(local, remote)
    print_diff(diff)

    if diff["local_only"]:
        to_push = {k: local[k] for k in diff["local_only"] if local[k]}
        if to_push and confirm(
            f"Push {len(to_push)} new local key(s) to 1Password "
            f"(and update {ENV_TEMPLATE})?",
            default_yes=False,
        ):
            push_to_1password(to_push)
            if not DRY_RUN:
                ok(f"Pushed {len(to_push)} key(s) to 1Password")
            if update_template(list(to_push)):
                ok(f"Updated {ENV_TEMPLATE}")
    return 0


# ---------- dispatch ----------
try:
    if mode == "status":
        sys.exit(do_status())
    elif mode == "pull":
        sys.exit(do_pull())
    elif mode == "push":
        sys.exit(do_push())
    else:
        sys.exit(do_default())
except subprocess.CalledProcessError as exc:
    err(f"1Password command failed: {exc}")
    if exc.stderr:
        print(exc.stderr, file=sys.stderr)
    sys.exit(1)
except KeyboardInterrupt:
    print("\nInterrupted.", file=sys.stderr)
    sys.exit(130)
PYEOF
