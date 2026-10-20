
"""Utilities for validating paths used for output files."""

from pathlib import Path


def validate_output_path(
    path: str | Path,
    allowed_root: str | Path,
) -> Path:
    """Ensure an output path stays within its allowed directory."""
    root = Path(allowed_root).resolve()
    candidate = Path(path).resolve()

    if not candidate.is_relative_to(root):
        raise ValueError(
            f"Output path must be inside '{root}': '{candidate}'"
        )

    return candidate
