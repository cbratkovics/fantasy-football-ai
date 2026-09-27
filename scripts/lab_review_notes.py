#!/usr/bin/env python3
"""Write private review notes for one Decision Lab case to a directory OUTSIDE the repository.

The notes are a fixed Markdown template with blank prompts. Only what was passed on the command
line is filled in (the revision, the exact command, a log file embedded verbatim, a receipt
summary read with ``json.load``); nothing is inferred, and the script never claims that anything
ran. The policy is not replayed here (``make lab-replay`` does that).

    python scripts/lab_review_notes.py --case-id <id> [--out-dir DIR] [--receipt PATH]
        [--command TEXT] [--log PATH] [--revision SHA]

The output directory defaults to ``$HOME/private-career-notes/fantasy-football/``. A directory
inside this repository, or inside any of its ``git worktree list`` paths, is refused with exit
status 2, so the notes can never be tracked, mirrored to the Space, or scanned into the site.
Files are named ``<YYYYMMDD-HHMMSS>-<case-id>.md`` and are never overwritten (a numeric suffix
is added on collision).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = Path.home() / "private-career-notes" / "fantasy-football"
EXIT_REFUSED = 2

SECTIONS = (
    "What I predicted before execution",
    "Implementation inspected and revision",
    "Exact command and observed output",
    "The one assumption I changed",
    "How the explanation and saved decision followed from the evidence",
    "Why the result matters",
    "What the check did not prove",
)
BLANK_PROMPT = "_(fill in)_"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def repo_root(cwd: Path = HERE) -> Path | None:
    top = _git(["rev-parse", "--show-toplevel"], cwd)
    return Path(top).resolve() if top else None


def worktree_paths(cwd: Path = HERE) -> list[Path]:
    """Every path listed by ``git worktree list --porcelain`` (the main tree included)."""
    listing = _git(["worktree", "list", "--porcelain"], cwd) or ""
    paths = []
    for line in listing.splitlines():
        if line.startswith("worktree "):
            paths.append(Path(line[len("worktree ") :]).resolve())
    return paths


def protected_roots(cwd: Path = HERE) -> list[Path]:
    roots = [p for p in [repo_root(cwd), *worktree_paths(cwd)] if p is not None]
    return sorted(set(roots))


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def refusal_reason(out_dir: Path, roots: list[Path]) -> str | None:
    for root in roots:
        if is_inside(out_dir, root):
            return (
                f"refusing to write private notes inside the repository or one of its worktrees "
                f"({root}); pass --out-dir pointing outside it"
            )
    return None


def current_revision(cwd: Path = HERE) -> str:
    return _git(["rev-parse", "HEAD"], cwd) or "(revision unavailable: not a git checkout)"


def receipt_summary(receipt: dict) -> list[str]:
    """Plain-text lines from a receipt document; read-only, no replay."""
    result = receipt.get("result") or {}
    action = receipt.get("action") or {}
    return [
        f"- decision_id: `{receipt.get('decision_id', '(missing)')}`",
        f"- status: {result.get('status', '(missing)')}",
        f"- recommended id: {result.get('recommended_player_id') or '(none)'}",
        f"- action state: {action.get('state', '(missing)')}",
    ]


def _fence(text: str, language: str = "") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{language}\n{text.rstrip()}\n{ticks}"


def render(
    *,
    case_id: str,
    revision: str,
    created_at: dt.datetime,
    command: str | None = None,
    log_text: str | None = None,
    receipt_lines: list[str] | None = None,
) -> str:
    stamp = created_at.strftime("%Y-%m-%d %H:%M:%S")
    body: dict[str, list[str]] = {name: [BLANK_PROMPT] for name in SECTIONS}
    body["Implementation inspected and revision"] = [f"Revision: `{revision}`", "", BLANK_PROMPT]
    evidence: list[str] = []
    if command:
        evidence += ["Command:", "", _fence(command, "sh"), ""]
    if log_text is not None:
        evidence += ["Observed output (embedded verbatim):", "", _fence(log_text), ""]
    evidence.append(BLANK_PROMPT)
    body["Exact command and observed output"] = evidence
    if receipt_lines:
        body["How the explanation and saved decision followed from the evidence"] = [
            "Receipt summary (read from the file; the policy was not replayed here):",
            *receipt_lines,
            "",
            BLANK_PROMPT,
        ]
    lines = [
        f"# Private review notes: {case_id}",
        "",
        f"Written {stamp} (local time). Prompts left as `{BLANK_PROMPT}` were not filled in;",
        "nothing below was inferred from the code or the bundle.",
        "",
    ]
    for name in SECTIONS:
        lines += [f"## {name}", "", *body[name], ""]
    return "\n".join(lines).rstrip() + "\n"


def safe_stem(case_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip("-.")
    return stem or "case"


def unique_path(out_dir: Path, created_at: dt.datetime, case_id: str) -> Path:
    base = f"{created_at.strftime('%Y%m%d-%H%M%S')}-{safe_stem(case_id)}"
    candidate = out_dir / f"{base}.md"
    n = 1
    while candidate.exists():
        n += 1
        candidate = out_dir / f"{base}-{n}.md"
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case-id", required=True, help="Decision Lab case id the notes are about")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for the notes; must be outside the repository (default: %(default)s)",
    )
    parser.add_argument("--receipt", type=Path, help="receipt JSON to summarise (no replay)")
    parser.add_argument("--command", help="the exact command that was run")
    parser.add_argument("--log", type=Path, help="file whose content is embedded verbatim")
    parser.add_argument("--revision", help="commit hash inspected (default: git rev-parse HEAD)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    reason = refusal_reason(out_dir, protected_roots())
    if reason:
        print(reason, file=sys.stderr)
        return EXIT_REFUSED
    log_text = args.log.read_text(encoding="utf-8") if args.log else None
    receipt_lines = None
    if args.receipt:
        receipt_lines = receipt_summary(json.loads(args.receipt.read_text(encoding="utf-8")))
    created_at = dt.datetime.now()
    text = render(
        case_id=args.case_id,
        revision=args.revision or current_revision(),
        created_at=created_at,
        command=args.command,
        log_text=log_text,
        receipt_lines=receipt_lines,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = unique_path(out_dir, created_at, args.case_id)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
