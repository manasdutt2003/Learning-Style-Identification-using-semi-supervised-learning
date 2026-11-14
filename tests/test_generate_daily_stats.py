import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_daily_stats as gds  # noqa: E402


def test_writes_valid_json_and_returns_same_data(tmp_path):
    out = tmp_path / "nested" / "stats.json"
    stats = gds.generate_stats(path=str(out), seed=1)
    assert json.loads(out.read_text(encoding="utf-8")) == stats


def test_value_ranges_and_choices(tmp_path):
    stats = gds.generate_stats(path=str(tmp_path / "s.json"), seed=2)
    assert 50 <= stats["active_users"] <= 500
    assert 100 <= stats["predictions_made"] <= 1000
    assert 0.85 <= stats["model_accuracy"] <= 0.98
    assert stats["top_learning_style"] in gds.LEARNING_STYLES
    assert stats["best_performing_model"] in gds.MODELS


def test_output_is_flagged_synthetic(tmp_path):
    stats = gds.generate_stats(path=str(tmp_path / "s.json"))
    assert stats["synthetic"] is True


def test_seed_makes_random_fields_reproducible(tmp_path):
    a = gds.generate_stats(path=str(tmp_path / "a.json"), seed=7)
    b = gds.generate_stats(path=str(tmp_path / "b.json"), seed=7)
    for key in ("active_users", "model_accuracy", "top_learning_style", "predictions_made"):
        assert a[key] == b[key]
