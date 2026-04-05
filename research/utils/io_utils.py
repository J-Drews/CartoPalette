"""
I/O utilities: config loading, path management, logging setup.
"""

import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml


# Project root (two levels up from research/utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def atomic_write_json(path: str | Path, data: dict | list, *, indent: int | None = None) -> None:
    """Write JSON atomically to avoid truncated files on sync/networked folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=f"{path.suffix}.tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def load_locations(path: str | Path | None = None) -> list[dict]:
    """Load the locations CSV file.

    Returns:
        List of dicts with keys: id, name, lat, lon, climate_zone, land_cover, split
    """
    if path is None:
        path = PROJECT_ROOT / "configs" / "data" / "locations.csv"
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a named logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger