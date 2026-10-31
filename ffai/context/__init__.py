"""Optional, separately versioned roster/game context; never changes policy-v1 evidence."""

from ffai.context.snapshot import build_snapshot, load_snapshot, validate_snapshot

__all__ = ["build_snapshot", "load_snapshot", "validate_snapshot"]
