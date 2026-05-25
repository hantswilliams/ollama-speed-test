#!/usr/bin/env python3
"""Free up local RAM by tearing down whatever LLM inference state is sitting around.

Standalone — only uses Python stdlib + macOS commands (vm_stat, pgrep, ollama).
Safe to copy to a fresh machine and run; no project deps.

Default behavior:
  1. Kill any llama-server processes (SIGTERM, then SIGKILL if needed)
  2. Evict every model loaded in Ollama (`ollama stop`) — leaves the daemon up
  3. Show free-RAM before/after and report the delta

Opt-in flags for the heavier stuff:
  --include-ollama     also kill the `ollama serve` daemon
  --include-opencode   also kill any opencode TUI processes
  --include-dashboard  also kill the dashboard proxy on :11435
  --purge              run `sudo purge` (prompts for password)
  --dry-run            show what would happen; don't change anything
  --quiet              minimal output (good for scripting)

Usage:
    python3 services/llm_cleanup/cleanup.py
    python3 services/llm_cleanup/cleanup.py --include-ollama --purge
    ./services/llm_cleanup/cleanup.py --dry-run     # if executable bit set
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time


# ---- RAM reporting -------------------------------------------------------


def free_ram_gb() -> float:
    """Best-effort estimate of memory available to new processes on macOS.

    Sum of (free + inactive + speculative + purgeable) pages × page size.
    Returns 0.0 if vm_stat isn't available.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True, timeout=3).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return 0.0
    page_size = 4096
    counts = {"free": 0, "inactive": 0, "speculative": 0, "purgeable": 0}
    for line in out.splitlines():
        m = re.search(r"page size of (\d+) bytes", line)
        if m:
            page_size = int(m.group(1))
            continue
        for key in counts:
            label = f"Pages {key}"
            if line.startswith(label):
                num = re.search(r"(\d+)", line.split(":", 1)[1])
                if num:
                    counts[key] = int(num.group(1))
                break
    total_bytes = sum(counts.values()) * page_size
    return total_bytes / (1024 ** 3)


# ---- ollama helpers ------------------------------------------------------


def ollama_ps() -> list:
    """Returns list of (model_name, size_str) for currently-loaded models."""
    if not shutil.which("ollama"):
        return []
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    lines = out.strip().splitlines()
    if len(lines) < 2:
        return []
    rows = []
    for line in lines[1:]:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= 3:
            rows.append((parts[0], parts[2]))
    return rows


def evict_ollama_model(name: str, quiet: bool = False) -> bool:
    try:
        proc = subprocess.run(
            ["ollama", "stop", name],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        if not quiet:
            print(f"    failed to evict {name}: {exc}")
        return False
    if proc.returncode != 0:
        if not quiet:
            print(f"    failed to evict {name}: {(proc.stderr or proc.stdout).strip()}")
        return False
    return True


# ---- process killing -----------------------------------------------------


def find_pids(pattern: str, exact: bool) -> list:
    if not shutil.which("pgrep"):
        return []
    flag = "-x" if exact else "-f"
    res = subprocess.run(["pgrep", flag, pattern], capture_output=True, text=True)
    my_pid = os.getpid()
    return [int(p) for p in res.stdout.split() if p.strip().isdigit() and int(p) != my_pid]


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_pids(pids: list, label: str, quiet: bool, timeout: float = 8.0) -> int:
    """SIGTERM then escalate to SIGKILL. Returns count actually killed."""
    if not pids:
        return 0
    if not quiet:
        print(f"  stopping {len(pids)} {label} process(es): {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            if not quiet:
                print(f"    no permission to signal PID {pid}")
    deadline = time.time() + timeout
    stragglers = []
    while time.time() < deadline:
        stragglers = [p for p in pids if is_alive(p)]
        if not stragglers:
            return len(pids)
        time.sleep(0.4)
    for pid in stragglers:
        if not quiet:
            print(f"    SIGKILL stubborn PID {pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(pids)


# ---- backend-specific actions -------------------------------------------


def stop_llama_server(quiet: bool, dry_run: bool) -> int:
    pids = find_pids("llama-server", exact=True)
    if not pids:
        if not quiet:
            print("  no llama-server processes running")
        return 0
    if dry_run:
        if not quiet:
            print(f"  [dry-run] would kill llama-server PIDs: {pids}")
        return 0
    return kill_pids(pids, "llama-server", quiet)


def evict_ollama_models(quiet: bool, dry_run: bool) -> int:
    loaded = ollama_ps()
    if not loaded:
        if not quiet:
            print("  no models loaded in Ollama")
        return 0
    if dry_run:
        if not quiet:
            print(f"  [dry-run] would evict: {[name for name, _ in loaded]}")
        return 0
    evicted = 0
    for name, size in loaded:
        if not quiet:
            print(f"  evicting {name} ({size})")
        if evict_ollama_model(name, quiet=quiet):
            evicted += 1
    return evicted


def stop_ollama_daemon(quiet: bool, dry_run: bool) -> int:
    # `ollama serve` runs as `ollama` with `serve` in argv
    pids = find_pids("ollama serve", exact=False)
    if not pids:
        if not quiet:
            print("  no ollama serve daemon running")
        return 0
    if dry_run:
        if not quiet:
            print(f"  [dry-run] would kill ollama serve PIDs: {pids}")
        return 0
    return kill_pids(pids, "ollama serve", quiet)


def stop_opencode(quiet: bool, dry_run: bool) -> int:
    pids = find_pids("opencode", exact=True)
    if not pids:
        if not quiet:
            print("  no opencode processes running")
        return 0
    if dry_run:
        if not quiet:
            print(f"  [dry-run] would kill opencode PIDs: {pids}")
        return 0
    return kill_pids(pids, "opencode", quiet)


def stop_dashboard_proxy(quiet: bool, dry_run: bool) -> int:
    # The proxy is a python process serving on :11435; match by port via lsof
    if not shutil.which("lsof"):
        if not quiet:
            print("  lsof unavailable — can't detect dashboard proxy")
        return 0
    res = subprocess.run(
        ["lsof", "-nP", "-i", ":11435", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True,
    )
    my_pid = os.getpid()
    pids = [int(p) for p in res.stdout.split() if p.strip().isdigit() and int(p) != my_pid]
    if not pids:
        if not quiet:
            print("  no dashboard proxy listening on :11435")
        return 0
    if dry_run:
        if not quiet:
            print(f"  [dry-run] would kill dashboard proxy PIDs: {pids}")
        return 0
    return kill_pids(pids, "dashboard proxy", quiet)


def run_purge(quiet: bool, dry_run: bool) -> None:
    if not shutil.which("sudo") or not shutil.which("purge"):
        if not quiet:
            print("  sudo/purge unavailable — skipping --purge")
        return
    if dry_run:
        if not quiet:
            print("  [dry-run] would run `sudo purge` (file-cache flush, prompts for password)")
        return
    if not quiet:
        print("  running `sudo purge` — you may be prompted for your password")
    try:
        subprocess.run(["sudo", "purge"], check=False)
    except KeyboardInterrupt:
        print("    cancelled")


# ---- top memory consumers ------------------------------------------------


def _derive_short_name(exe_path: str) -> str:
    """Pull a readable name from an executable path.

    Prefers the rightmost .app bundle (catches helpers like 'Code Helper (Renderer)'),
    falls back to the basename.
    """
    bundles = re.findall(r"/([^/]+?)\.app(?:/|$)", exe_path)
    if bundles:
        return bundles[-1]
    return os.path.basename(exe_path) or exe_path


def top_memory_consumers(limit: int, min_mb: float = 200.0) -> list:
    """Returns list of (pid, rss_mb, short_name, exe_path) sorted by RSS desc.

    Uses `ps -axo pid=,rss=,comm=` and reconstructs the full exe path even when
    it contains spaces (common on macOS).
    """
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,rss=,comm="],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    my_pid = os.getpid()
    entries = []
    for line in out.splitlines():
        # First 2 whitespace-delimited tokens are pid + rss; the remainder is
        # the full exe path (may itself contain spaces).
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_kb = int(parts[1])
        except ValueError:
            continue
        if pid == my_pid:
            continue
        rss_mb = rss_kb / 1024
        if rss_mb < min_mb:
            continue
        exe = parts[2]
        entries.append((pid, rss_mb, _derive_short_name(exe), exe))
    entries.sort(key=lambda e: e[1], reverse=True)
    return entries[:limit]


def fmt_mb(mb: float) -> str:
    return f"{mb/1024:.1f} GB" if mb >= 1024 else f"{int(mb)} MB"


def print_top_memory(limit: int = 10) -> None:
    rows = top_memory_consumers(limit=limit)
    if not rows:
        return
    print()
    print(f"Top {len(rows)} memory consumers still running (resident set size):")
    for pid, rss_mb, short, exe in rows:
        short_disp = short if len(short) <= 34 else short[:31] + "..."
        # Trim the exe path for context — keep the suffix since the prefix is repetitive
        if len(exe) > 60:
            exe = "…" + exe[-59:]
        print(f"  {fmt_mb(rss_mb):>8}  PID {pid:<6}  {short_disp:<35}  {exe}")


# ---- main ----------------------------------------------------------------


def fmt_loaded(rows: list) -> str:
    if not rows:
        return "(none)"
    return ", ".join(f"{name} ({size})" for name, size in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tear down local LLM inference state and report freed memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--include-ollama", action="store_true",
                        help="Also kill the `ollama serve` daemon (default just evicts loaded models).")
    parser.add_argument("--include-opencode", action="store_true",
                        help="Also kill any opencode TUI processes.")
    parser.add_argument("--include-dashboard", action="store_true",
                        help="Also kill the dashboard proxy on :11435.")
    parser.add_argument("--purge", action="store_true",
                        help="Run `sudo purge` to flush the macOS file cache (prompts for password).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen; don't change anything.")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output (good for scripting).")
    parser.add_argument("--top", type=int, default=10,
                        help="Show the top N memory consumers still running after cleanup (default 10; pass 0 to skip).")
    args = parser.parse_args()

    quiet = args.quiet
    before_ram = free_ram_gb()
    before_loaded = ollama_ps()

    if not quiet:
        mode = " [DRY RUN]" if args.dry_run else ""
        print(f"=== llm_cleanup{mode} ===")
        print(f"Before: {before_ram:.1f} GB free  |  Ollama loaded: {fmt_loaded(before_loaded)}")
        print()
        print("Actions:")

    stop_llama_server(quiet, args.dry_run)
    evict_ollama_models(quiet, args.dry_run)
    if args.include_ollama:
        stop_ollama_daemon(quiet, args.dry_run)
    if args.include_opencode:
        stop_opencode(quiet, args.dry_run)
    if args.include_dashboard:
        stop_dashboard_proxy(quiet, args.dry_run)
    if args.purge:
        run_purge(quiet, args.dry_run)

    # Brief settle so vm_stat reflects the kills
    if not args.dry_run:
        time.sleep(1.0)
    after_ram = free_ram_gb()
    after_loaded = ollama_ps()
    delta = after_ram - before_ram

    if quiet:
        print(f"freed_gb={delta:.1f} before_gb={before_ram:.1f} after_gb={after_ram:.1f}")
        return

    print()
    print(f"After:  {after_ram:.1f} GB free  |  Ollama loaded: {fmt_loaded(after_loaded)}")
    sign = "+" if delta >= 0 else ""
    print(f"Delta:  {sign}{delta:.1f} GB")

    if args.top > 0:
        print_top_memory(limit=args.top)


if __name__ == "__main__":
    main()
