from pathlib import Path

import pytest

from motion.catalog import MotionCatalog
from motion.primitives import PrimitiveLibrary


HEADER = (
    "timestamp,base_yaw.pos,base_pitch.pos,elbow_pitch.pos,"
    "wrist_roll.pos,wrist_pitch.pos\n"
)


def valid_rows() -> str:
    return (
        "0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1.0,1.0,1.0,1.0,1.0,1.0\n"
    )


def write_recording(path: Path, rows: str) -> None:
    path.write_text(HEADER + rows)


def write_catalog(path: Path, motions: dict[str, dict[str, object]]) -> None:
    lines: list[str] = []
    for name, values in motions.items():
        lines.append(f"[motions.{name}]")
        for key, value in values.items():
            rendered = repr(value).lower() if isinstance(value, bool) else repr(value)
            lines.append(f"{key} = {rendered}")
        lines.append("")
    path.write_text("\n".join(lines))


def test_catalog_is_an_explicit_sorted_whitelist(tmp_path):
    write_recording(tmp_path / "wave.csv", valid_rows())
    write_recording(tmp_path / "draft.csv", valid_rows())
    write_catalog(
        tmp_path / "catalog.toml",
        {"wave": {"file": "wave.csv", "enabled": True}},
    )

    catalog = MotionCatalog.load(tmp_path / "catalog.toml")

    assert catalog.names() == ["wave"]


def test_catalog_preflight_rejects_non_finite_recording_before_hardware(tmp_path):
    write_recording(tmp_path / "unsafe.csv", "0.0,nan,0.0,0.0,0.0,0.0\n")
    write_catalog(
        tmp_path / "catalog.toml",
        {"unsafe": {"file": "unsafe.csv", "enabled": True}},
    )

    report = MotionCatalog.load(tmp_path / "catalog.toml").validate()

    assert not report.ok
    assert report.errors[0].code == "non_finite"


def test_catalog_preflight_reports_malformed_columns(tmp_path):
    (tmp_path / "broken.csv").write_text("timestamp,base_yaw.pos\n0.0,0.0\n")
    write_catalog(
        tmp_path / "catalog.toml",
        {"broken": {"file": "broken.csv", "enabled": True}},
    )

    report = MotionCatalog.load(tmp_path / "catalog.toml").validate()

    assert not report.ok
    assert report.errors[0].code == "malformed_columns"


def test_catalog_preflight_reports_inconsistent_columns(tmp_path):
    (tmp_path / "broken.csv").write_text(
        HEADER + "0.0,0.0,0.0,0.0,0.0\n"
    )
    write_catalog(
        tmp_path / "catalog.toml",
        {"broken": {"file": "broken.csv", "enabled": True}},
    )

    report = MotionCatalog.load(tmp_path / "catalog.toml").validate()

    assert not report.ok
    assert report.errors[0].code == "malformed_columns"


def test_catalog_preflight_uses_generic_code_for_missing_recording(tmp_path):
    write_catalog(
        tmp_path / "catalog.toml",
        {"missing": {"file": "missing.csv", "enabled": True}},
    )

    report = MotionCatalog.load(tmp_path / "catalog.toml").validate()

    assert not report.ok
    assert report.errors[0].code == "invalid_recording"


@pytest.mark.parametrize(
    ("motion", "values", "message"),
    [
        ("Bad-name", {"file": "wave.csv", "enabled": True}, "invalid motion name"),
        ("wave", {"file": "wave.csv", "enabled": True, "danger": "yes"}, "unknown keys"),
        ("wave", {"file": "../wave.csv", "enabled": True}, "outside catalog directory"),
    ],
)
def test_catalog_load_rejects_invalid_configuration(tmp_path, motion, values, message):
    write_catalog(tmp_path / "catalog.toml", {motion: values})

    with pytest.raises(ValueError, match=message):
        MotionCatalog.load(tmp_path / "catalog.toml")


def test_catalog_library_rejects_unlisted_recording(tmp_path):
    write_recording(tmp_path / "wave.csv", valid_rows())
    write_recording(tmp_path / "draft.csv", valid_rows())
    write_catalog(
        tmp_path / "catalog.toml",
        {"wave": {"file": "wave.csv", "enabled": True}},
    )

    library = MotionCatalog.load(tmp_path / "catalog.toml").library()

    with pytest.raises(KeyError, match="draft"):
        library.get("draft")


def test_library_rejects_unlisted_name_before_filesystem_access(tmp_path):
    library = PrimitiveLibrary(
        recordings_dir=tmp_path / "missing-recordings",
        allowed_names=frozenset({"wave"}),
    )

    with pytest.raises(KeyError, match="draft"):
        library.get("draft")
