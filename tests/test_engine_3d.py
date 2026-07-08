"""Tests for 3D engine visualization helpers (pure functions, no mesh files)."""

from pathlib import Path
import numpy as np
import pyvista as pv
import pytest
import yaml

from src.viz.engine_3d import (
    _active_engine_model,
    _health_key,
    _load_viz_config,
    _load_sensor_config,
    _mesh_bounds,
    _mesh_to_json,
    _compute_explode_offsets,
    STAGE_ORDER,
)


def test_health_key():
    assert _health_key("compressor") == "CompressorHealth"
    assert _health_key("combustor") == "CombustorHealth"
    assert _health_key("turbine") == "TurbineHealth"
    assert _health_key("casing") == "CasingHealth"


def test_active_engine_model_default(monkeypatch):
    monkeypatch.setattr("src.viz.engine_3d.VIZ_CONFIG", Path("nonexistent_file.yaml"))
    assert _active_engine_model() == "generic_turbine"


def test_active_engine_model_from_config(tmp_path, monkeypatch):
    cfg = tmp_path / "viz_config.yaml"
    cfg.write_text("active_engine_model: kj66", encoding="utf-8")
    monkeypatch.setattr("src.viz.engine_3d.VIZ_CONFIG", cfg)
    assert _active_engine_model() == "kj66"


def test_load_viz_config_missing(monkeypatch):
    monkeypatch.setattr("src.viz.engine_3d.VIZ_CONFIG", Path("nonexistent.yaml"))
    assert _load_viz_config() == {}


def test_load_viz_config_present(tmp_path, monkeypatch):
    cfg = tmp_path / "viz_config.yaml"
    data = {"active_engine_model": "kj66", "health_thresholds": []}
    cfg.write_text(yaml.dump(data), encoding="utf-8")
    monkeypatch.setattr("src.viz.engine_3d.VIZ_CONFIG", cfg)
    assert _load_viz_config() == data


def test_load_sensor_config_missing(monkeypatch):
    monkeypatch.setattr("src.viz.engine_3d.SENSOR_CONFIG", Path("nonexistent.yaml"))
    assert _load_sensor_config() == {}


def test_load_sensor_config_present(tmp_path, monkeypatch):
    cfg = tmp_path / "sensors.yaml"
    data = {"sensors": [{"name": "T3", "pos": [0, 0, 1]}]}
    cfg.write_text(yaml.dump(data), encoding="utf-8")
    monkeypatch.setattr("src.viz.engine_3d.SENSOR_CONFIG", cfg)
    assert _load_sensor_config() == data["sensors"]


def test_mesh_bounds():
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.float32)
    faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])
    mesh = pv.PolyData(pts, faces)
    b = _mesh_bounds(mesh)
    assert b == pytest.approx((0.0, 2.0, 0.0, 3.0, 0.0, 4.0))


def test_mesh_to_json():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([3, 0, 1, 2])
    mesh = pv.PolyData(pts, faces)
    result = _mesh_to_json(mesh)
    assert "vertices" in result
    assert "faces" in result
    assert len(result["vertices"]) == 3
    assert result["faces"] == [[0, 1, 2]]


def test_compute_explode_offsets():
    meshes = {}
    for i, stage in enumerate(STAGE_ORDER):
        x = float(i * 100)
        pts = np.array([[x, 0, 0], [x + 20, 0, 0], [x, 10, 0]], dtype=np.float32)
        faces = np.array([3, 0, 1, 2])
        meshes[stage] = pv.PolyData(pts, faces)
    offsets = _compute_explode_offsets(meshes)
    assert set(offsets.keys()) == set(STAGE_ORDER)
    # casing offset should be zero (casing doesn't explode)
    assert offsets["casing"][0] == 0.0
    # positive-x stages should have positive offsets
    assert offsets["compressor"][0] > 0
    assert offsets["combustor"][0] > 0
    assert offsets["turbine"][0] > 0


def test_compute_explode_offsets_missing_stage():
    pts = np.array([[100, 0, 0], [120, 0, 0], [100, 10, 0]], dtype=np.float32)
    meshes = {"compressor": pv.PolyData(pts, np.array([3, 0, 1, 2]))}
    offsets = _compute_explode_offsets(meshes)
    assert offsets["casing"] == [0, 0, 0]
    assert offsets["combustor"] == [0, 0, 0]
    assert offsets["turbine"] == [0, 0, 0]
    assert offsets["compressor"] != [0.0, 0.0, 0.0]
