import json
from pathlib import Path

import pytest
from box import ConfigBox
from ensure.main import EnsureError

from KidneyDiseaseClassifier.utils.common import (
    read_yaml,
    create_directories,
    save_json,
    load_json,
    get_size,
)


def test_read_yaml_returns_configbox(tmp_path):
    p = tmp_path / "sample.yaml"
    p.write_text("key: value\nnested:\n  a: 1\n")
    result = read_yaml(Path(p))
    assert isinstance(result, ConfigBox)
    assert result.key == "value"
    assert result.nested.a == 1


def test_read_yaml_empty_raises(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ValueError):
        read_yaml(Path(p))


def test_read_yaml_rejects_non_path():
    # @ensure_annotations enforces the Path annotation.
    with pytest.raises(EnsureError):
        read_yaml("not-a-path-object")


def test_create_directories(tmp_path):
    target = tmp_path / "a" / "b"
    create_directories([str(target)], verbose=False)
    assert target.exists()


def test_save_and_load_json(tmp_path):
    p = tmp_path / "data.json"
    save_json(Path(p), {"accuracy": 0.9})
    loaded = load_json(Path(p))
    assert loaded.accuracy == 0.9
    assert json.loads(p.read_text())["accuracy"] == 0.9


def test_get_size(tmp_path):
    p = tmp_path / "file.bin"
    p.write_bytes(b"0" * 2048)
    assert "KB" in get_size(Path(p))
