"""Tests for training history JSON/CSV export."""

import csv
import json

from training.history_export import save_history_csv, save_history_json


def _sample_history():
    return [
        {"total_reward": 3.0, "steps": 4, "won": False, "epsilon": 0.9, "loss": 1.23},
        {"total_reward": 10.0, "steps": 6, "won": True, "epsilon": 0.85, "loss": None},
    ]


def test_save_history_json_adds_episode_numbers(tmp_path):
    path = tmp_path / "history.json"
    save_history_json(_sample_history(), path)

    records = json.loads(path.read_text())

    assert len(records) == 2
    assert records[0]["episode"] == 1
    assert records[1]["episode"] == 2


def test_save_history_json_preserves_all_fields(tmp_path):
    path = tmp_path / "history.json"
    save_history_json(_sample_history(), path)

    records = json.loads(path.read_text())

    assert records[0]["total_reward"] == 3.0
    assert records[0]["steps"] == 4
    assert records[0]["won"] is False
    assert records[0]["epsilon"] == 0.9
    assert records[0]["loss"] == 1.23
    assert records[1]["won"] is True
    assert records[1]["loss"] is None


def test_save_history_csv_writes_header_and_rows(tmp_path):
    path = tmp_path / "history.csv"
    save_history_csv(_sample_history(), path)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert set(rows[0].keys()) == {"episode", "total_reward", "steps", "won", "epsilon", "loss"}
    assert rows[0]["episode"] == "1"
    assert rows[0]["steps"] == "4"
    assert rows[1]["episode"] == "2"
    assert rows[1]["won"] == "True"


def test_save_history_csv_handles_empty_history(tmp_path):
    path = tmp_path / "empty.csv"
    save_history_csv([], path)

    assert path.exists()
    assert path.read_text() == ""


def test_save_history_json_handles_empty_history(tmp_path):
    path = tmp_path / "empty.json"
    save_history_json([], path)

    assert json.loads(path.read_text()) == []
