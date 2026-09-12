"""Explicit, preflight-validated whitelist of runnable motion recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
from typing import Any

from .primitives import Primitive, PrimitiveLibrary


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ENTRY_KEYS = frozenset({"file", "enabled", "tags"})


@dataclass(frozen=True)
class MotionEntry:
    name: str
    file: Path
    enabled: bool
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CatalogError:
    name: str
    code: str
    message: str


@dataclass(frozen=True)
class CatalogReport:
    errors: tuple[CatalogError, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class CatalogValidationError(ValueError):
    def __init__(self, report: CatalogReport):
        self.report = report
        message = "; ".join(
            f"{error.name}: {error.code}: {error.message}" for error in report.errors
        )
        super().__init__(message)


class MotionCatalog:
    def __init__(self, path: Path, entries: tuple[MotionEntry, ...]):
        self.path = path.resolve()
        self.recordings_dir = self.path.parent
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> "MotionCatalog":
        path = Path(path).resolve()
        with path.open("rb") as stream:
            document = tomllib.load(stream)

        if set(document) != {"motions"}:
            raise ValueError("catalog must contain only the 'motions' table")
        motions = document["motions"]
        if not isinstance(motions, dict):
            raise ValueError("motions must be a table")

        entries = tuple(
            cls._entry(path, name, values) for name, values in motions.items()
        )
        return cls(path, entries)

    @staticmethod
    def _entry(path: Path, name: str, values: Any) -> MotionEntry:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError(f"invalid motion name: {name!r}")
        if not isinstance(values, dict):
            raise ValueError(f"motion {name!r} must be a table")

        unknown = set(values) - _ENTRY_KEYS
        if unknown:
            raise ValueError(f"motion {name!r} has unknown keys: {sorted(unknown)!r}")
        if "file" not in values or "enabled" not in values:
            raise ValueError(f"motion {name!r} requires file and enabled")
        if not isinstance(values["file"], str) or not values["file"]:
            raise ValueError(f"motion {name!r} file must be a non-empty string")
        if not isinstance(values["enabled"], bool):
            raise ValueError(f"motion {name!r} enabled must be a boolean")

        tags = values.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"motion {name!r} tags must be an array of strings")

        recordings_dir = path.parent.resolve()
        recording = (recordings_dir / values["file"]).resolve()
        if recording.parent != recordings_dir:
            raise ValueError(f"motion {name!r} file is outside catalog directory")

        return MotionEntry(
            name=name,
            file=recording,
            enabled=values["enabled"],
            tags=tuple(tags),
        )

    def names(self) -> list[str]:
        return sorted(entry.name for entry in self.entries if entry.enabled)

    def validate(self) -> CatalogReport:
        errors: list[CatalogError] = []
        for entry in self.entries:
            if not entry.enabled:
                continue
            try:
                Primitive.load(entry.name, recordings_dir=self.recordings_dir)
            except OSError as exc:
                errors.append(CatalogError(entry.name, "invalid_recording", str(exc)))
            except KeyError as exc:
                errors.append(CatalogError(entry.name, "malformed_columns", str(exc)))
            except ValueError as exc:
                errors.append(
                    CatalogError(entry.name, _recording_error_code(exc), str(exc))
                )
        return CatalogReport(tuple(errors))

    def library(self) -> PrimitiveLibrary:
        report = self.validate()
        if not report.ok:
            raise CatalogValidationError(report)
        return PrimitiveLibrary(
            recordings_dir=self.recordings_dir,
            allowed_names=frozenset(self.names()),
        )


def _recording_error_code(exc: ValueError) -> str:
    message = str(exc)
    if message.startswith("non_finite:"):
        return "non_finite"
    if message.startswith("malformed_columns:"):
        return "malformed_columns"
    return "invalid_recording"
