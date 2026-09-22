"""Read-only Git helpers shared by audit and experiment wrappers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Mapping, Sequence


PathNormalizer = Callable[[str], str]


ALLOWED_READ_ONLY_SUBCOMMANDS = {
    "rev-parse",
    "status",
    "diff",
    "diff-files",
    "diff-index",
    "ls-files",
    "log",
    "show",
    "branch",
    "symbolic-ref",
    "config",
    "cat-file",
}

FORBIDDEN_SUBCOMMANDS = {
    "add",
    "am",
    "apply",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "fetch",
    "merge",
    "mv",
    "pull",
    "push",
    "rebase",
    "reset",
    "restore",
    "rm",
    "stash",
    "switch",
    "tag",
    "update-index",
}

FORBIDDEN_DIFF_OPTIONS = {"--output", "--ext-diff"}


class GitReadOnlyError(ValueError):
    """Raised when a Git command is outside the approved read-only contract."""


def normalize_git_args(command: Sequence[str]) -> list[str]:
    """Return Git arguments without the executable after read-only validation."""

    parts = [str(part) for part in command]
    if parts and Path(parts[0]).name.lower() in {"git", "git.exe"}:
        parts = parts[1:]
    if not parts:
        raise GitReadOnlyError("empty git command")

    subcommand = parts[0]
    if subcommand.startswith("-"):
        raise GitReadOnlyError(f"git option before subcommand is not allowed: {subcommand}")
    if subcommand in FORBIDDEN_SUBCOMMANDS:
        raise GitReadOnlyError(f"git subcommand is forbidden: {subcommand}")
    if subcommand not in ALLOWED_READ_ONLY_SUBCOMMANDS:
        raise GitReadOnlyError(f"git subcommand is not in the read-only allowlist: {subcommand}")
    if subcommand == "branch" and parts[1:] != ["--show-current"]:
        raise GitReadOnlyError("only 'git branch --show-current' is allowed")
    if subcommand == "config" and (len(parts) < 3 or parts[1] != "--get"):
        raise GitReadOnlyError("only 'git config --get <key>' is allowed")
    if subcommand in {"diff", "diff-files", "diff-index"} and any(
        arg == "--ext-diff" or arg.startswith("--output") for arg in parts[1:]
    ):
        raise GitReadOnlyError("git diff output/external-diff options are not allowed")
    return parts


def run_git_readonly(
    command: Sequence[str],
    cwd: str | Path,
    *,
    check: bool = False,
    text: bool = True,
    timeout: int | float | None = None,
    encoding: str | None = None,
    errors: str | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run an approved read-only Git command with an explicit working directory."""

    args = normalize_git_args(command)
    kwargs: dict[str, object] = {
        "cwd": Path(cwd),
        "check": check,
        "text": text,
        "timeout": timeout,
        "shell": False,
    }
    if text and encoding is not None:
        kwargs["encoding"] = encoding
    if text and errors is not None:
        kwargs["errors"] = errors
    if stdout is None and stderr is None:
        kwargs["capture_output"] = True
    else:
        kwargs["stdout"] = subprocess.PIPE if stdout is None else stdout
        kwargs["stderr"] = subprocess.PIPE if stderr is None else stderr
    if env is not None:
        kwargs["env"] = dict(env)
    return subprocess.run(["git", *args], **kwargs)


def git_stdout_checked(
    command: Sequence[str],
    cwd: str | Path,
    *,
    strip: bool = True,
    timeout: int | float | None = None,
    stderr_to_devnull: bool = False,
) -> str:
    """Return stdout for a checked read-only Git command."""

    completed = run_git_readonly(
        command,
        cwd,
        check=True,
        text=True,
        timeout=timeout,
        stderr=subprocess.DEVNULL if stderr_to_devnull else None,
    )
    return completed.stdout.strip() if strip else completed.stdout


def git_stdout_no_check(
    command: Sequence[str],
    cwd: str | Path,
    *,
    strip: bool = True,
    timeout: int | float | None = None,
    encoding: str | None = None,
    errors: str | None = None,
    empty_on_failure: bool = False,
    stdout_or_stderr: bool = False,
) -> str:
    """Return stdout for a non-checked read-only Git command."""

    completed = run_git_readonly(
        command,
        cwd,
        check=False,
        text=True,
        timeout=timeout,
        encoding=encoding,
        errors=errors,
    )
    if empty_on_failure and completed.returncode != 0:
        return ""
    output = (completed.stdout or completed.stderr) if stdout_or_stderr else completed.stdout
    return output.strip() if strip else output


def git_lines_unavailable_on_error(
    command: Sequence[str],
    cwd: str | Path,
    *,
    normalize_paths: bool = False,
    timeout: int | float | None = 30,
) -> list[str]:
    """Preserve legacy helpers that returned a one-row diagnostic on Git failure."""

    command_list = [str(part) for part in command]
    try:
        completed = run_git_readonly(command_list, cwd, check=False, text=True, timeout=timeout)
    except Exception as exc:
        return [f"git command unavailable: {exc}"]
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        return [stderr or f"{' '.join(command_list)} exited {completed.returncode}"]
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if normalize_paths:
        return [line.replace("\\", "/") for line in lines]
    return lines


def git_lines_checked(command: Sequence[str], cwd: str | Path) -> list[str]:
    """Return non-empty stdout lines or raise the legacy RuntimeError message."""

    args = normalize_git_args(command)
    completed = run_git_readonly(args, cwd, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return [line for line in completed.stdout.splitlines() if line]


def git_status_lines(status_text: str) -> list[str]:
    """Return non-empty porcelain status lines while preserving order."""

    return [line for line in status_text.splitlines() if line.strip()]


def git_status_path(line: str, *, strip_outer_quotes: bool = False) -> str:
    """Return the path segment from a porcelain v1 status line."""

    path = line[3:].strip().replace("\\", "/") if len(line) >= 3 else ""
    return path.strip('"') if strip_outer_quotes else path


def git_status_counts(lines: Iterable[str]) -> dict[str, int]:
    """Count common porcelain v1 status categories deterministically."""

    counts: Counter[str] = Counter()
    for line in lines:
        code = line[:2]
        if code == "??":
            counts["untracked"] += 1
        elif "M" in code:
            counts["modified"] += 1
        elif "A" in code:
            counts["added"] += 1
        elif "D" in code:
            counts["deleted"] += 1
        else:
            counts["other"] += 1
    return {key: counts.get(key, 0) for key in ["modified", "added", "deleted", "untracked", "other"]}


def git_status_entries_z(cwd: str | Path) -> list[tuple[str, str]]:
    """Return ``git status --porcelain=v1 -z`` entries as ``(status, path)`` tuples."""

    completed = run_git_readonly(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd,
        check=False,
        text=False,
    )
    if completed.returncode != 0:
        return []
    raw = completed.stdout.decode("utf-8", errors="replace")
    parts = [part for part in raw.split("\0") if part]
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(parts):
        entry = parts[index]
        status = entry[:2]
        path = entry[3:].replace("\\", "/")
        entries.append((status, path))
        index += 2 if status.startswith("R") or status.startswith("C") else 1
    return entries


def git_state_maps_tuple(cwd: str | Path) -> tuple[set[str], dict[str, str]]:
    """Return legacy ``(tracked, status_code_by_path)`` state maps."""

    tracked_result = run_git_readonly(["ls-files"], cwd, check=False, text=True, timeout=30)
    tracked = set(tracked_result.stdout.splitlines()) if tracked_result.returncode == 0 else set()
    status_result = run_git_readonly(["status", "--porcelain=v1"], cwd, check=False, text=True, timeout=30)
    status: dict[str, str] = {}
    if status_result.returncode == 0:
        for line in status_result.stdout.splitlines():
            if not line:
                continue
            code = line[:2]
            path = git_status_path(line, strip_outer_quotes=True)
            status[path] = code
    return tracked, status


def git_state_path_sets(cwd: str | Path, normalize_path: PathNormalizer) -> dict[str, set[str]]:
    """Return legacy tracked/modified/staged/untracked/ignored path sets."""

    tracked = {normalize_path(path) for path in git_lines_checked(["ls-files"], cwd)}
    modified = {normalize_path(path) for path in git_lines_checked(["diff", "--name-only"], cwd)}
    staged = {normalize_path(path) for path in git_lines_checked(["diff", "--cached", "--name-only"], cwd)}
    untracked = {
        normalize_path(path) for path in git_lines_checked(["ls-files", "--others", "--exclude-standard"], cwd)
    }
    ignored = {
        normalize_path(path)
        for path in git_lines_checked(["ls-files", "--others", "--ignored", "--exclude-standard"], cwd)
    }
    return {
        "tracked": tracked,
        "modified": modified,
        "staged": staged,
        "untracked": untracked,
        "ignored": ignored,
    }


def classify_git_state(
    relative_path: str,
    states: Mapping[str, set[str]],
    normalize_path: PathNormalizer,
) -> str:
    """Classify a path against legacy Git state maps."""

    path = normalize_path(relative_path)
    if path in states.get("staged", set()):
        return "staged"
    if path in states.get("modified", set()):
        return "tracked_modified"
    if path in states.get("tracked", set()):
        return "tracked_clean"
    if path in states.get("untracked", set()):
        return "untracked"
    if path in states.get("ignored", set()):
        return "ignored"
    return "nested_or_unclassified"


def filter_git_path_output(text: str, generated_paths: set[str], *, status_format: bool) -> str:
    """Remove generated paths from legacy Git command text."""

    if not text:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        normalized = line.replace("\\", "/")
        path_part = normalized[3:] if status_format and len(normalized) > 3 else normalized.strip()
        if any(path_part == output or output in normalized for output in generated_paths):
            continue
        kept.append(line)
    return "\n".join(kept)


def audit_resume_git_baseline(final_git_status: str, audit_created_status_paths: set[str]) -> dict[str, object]:
    """Reconstruct the forensic audit resume baseline from final status text."""

    final_lines = git_status_lines(final_git_status)
    audit_created = [line for line in final_lines if git_status_path(line) in audit_created_status_paths]
    reconstructed = [line for line in final_lines if git_status_path(line) not in audit_created_status_paths]
    return {
        "basis": "reconstructed_from_final_status_minus_new_forensic_audit_paths",
        "status_row_count": len(reconstructed),
        "status_counts": git_status_counts(reconstructed),
        "audit_created_status_row_count": len(audit_created),
        "audit_created_paths": sorted(git_status_path(line) for line in audit_created),
        "note": "The first resume-time git check verified branch and HEAD before generation. Repeated deterministic rebuilds overwrite live 'initial before this builder run' counts, so this baseline is reconstructed from final status by removing files first created by the complete audit.",
    }
