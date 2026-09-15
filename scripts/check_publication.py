#!/usr/bin/env python3
"""Reject high-confidence coaching material from tracked and deployable text.

Diagnostics intentionally identify only the file, line, and stable rule id. They do not print
matched content, which keeps CI logs from republishing material that the check rejects.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    identifier: str
    pattern: re.Pattern[str]


CONTENT_RULES = (
    Rule(
        "coaching.interview",
        re.compile(
            r"\binterview\s+(?:prep(?:aration)?|narrative|script|talking\s+points?)\b", re.I
        ),
    ),
    Rule("coaching.mock-interview", re.compile(r"\bmock\s+interview\b", re.I)),
    Rule("coaching.behavioral-answer", re.compile(r"\bSTAR[- ](?:answer|response)\b", re.I)),
    Rule("coaching.resume", re.compile(r"\br[ée]sum[ée]\s+(?:advice|tip|rewrite|bullet)\b", re.I)),
    Rule("coaching.career", re.compile(r"\bcareer[- ](?:coaching|signal\s+roadmap)\b", re.I)),
    Rule(
        "coaching.role-targeting",
        re.compile(r"\brole[- ]targeting\s+(?:guidance|strategy|plan)\b", re.I),
    ),
    Rule("coaching.timed-pitch", re.compile(r"\b(?:timed\s+narrative|elevator\s+pitch)\b", re.I)),
)

SUSPICIOUS_NAME = re.compile(
    r"(?:^|[-_. ])(?:interview(?:[-_. ]?(?:prep|script|notes))?|"
    r"resume[-_. ]?(?:advice|rewrite)|career[-_. ]?(?:coaching|roadmap))(?:[-_. ]|$)",
    re.I,
)

SKIP_PARTS = {".git", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {
    "",
    ".css",
    ".csv",
    ".html",
    ".ipynb",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mdx",
    ".map",
    ".ndjson",
    ".py",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".atom",
    ".rss",
    ".xml",
    ".yaml",
    ".yml",
}


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def files_under(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and not SKIP_PARTS.intersection(candidate.parts):
                    yield candidate


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def scan(paths: Iterable[Path], root: Path) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen or not is_text_candidate(path):
            continue
        seen.add(path)
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        if SUSPICIOUS_NAME.search(path.name):
            findings.append((relative, 0, "document.suspicious-name"))
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for rule in CONTENT_RULES:
                if rule.pattern.search(line):
                    findings.append((relative, line_number, rule.identifier))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="scan tracked text and optional generated publication paths"
    )
    parser.add_argument("paths", nargs="*", type=Path, help="additional files or directories")
    args = parser.parse_args(argv)
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    candidates = [*tracked_files(root), *files_under(args.paths)]
    findings = scan(candidates, root)
    for path, line, rule in findings:
        print(f"{path}:{line}: {rule}", file=sys.stderr)
    if findings:
        print(f"publication check failed with {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"publication check passed ({len(set(map(Path.resolve, candidates)))} files considered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
