from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.extension.f1bx_execution_stage1 import bool_value, per_user_stats
from src.extension.g1_primary_ranking_execution import g1_beyond_metric_values


def test_bool_value_parses_python_and_serialized_booleans() -> None:
    assert bool_value(True) is True
    assert bool_value(False) is False
    assert bool_value("True") is True
    assert bool_value("False") is False
    assert bool_value("true") is True
    assert bool_value("false") is False
    assert bool_value(1) is True
    assert bool_value(0) is False


def test_per_user_stats_does_not_treat_false_string_as_ground_truth() -> None:
    recommendations = pd.DataFrame(
        [
            {
                "dataset": "beauty",
                "method": "MostPop",
                "user_id": "u1",
                "rank": "1",
                "recommended_item_id": "head_item",
                "is_ground_truth": "False",
                "positive_item_id": "target_item",
            },
            {
                "dataset": "beauty",
                "method": "MostPop",
                "user_id": "u1",
                "rank": "19",
                "recommended_item_id": "target_item",
                "is_ground_truth": "True",
                "positive_item_id": "target_item",
            },
        ]
    )
    material = SimpleNamespace(spec=SimpleNamespace(name="beauty"), history_lengths_by_user={})

    stats = per_user_stats(recommendations, material)

    assert stats.loc[0, "rank_one_based"] == 19
    assert stats.loc[0, "hit_at_5"] == 0


def test_g1_serendipity_cell_reconstruction_from_known_hits() -> None:
    items = ["i1", "i2", "i3", "i4", "i5", "h1"]
    material = SimpleNamespace(
        train_counts={item: 1 for item in items},
        total_train_interactions=len(items),
        candidates=pd.DataFrame({"candidate_item_id": items}),
        catalog_items=set(items),
        item_features={item: ["tag"] for item in items},
        history_lengths_by_user={"u1": 1, "u2": 1, "u3": 1},
    )
    stats = pd.DataFrame({"user_id": ["u1", "u2", "u3"]})
    top_lists = [["i1", "i2", "i3", "i4", "i5"] for _ in range(3)]
    ranks = [1, 2, 6]
    hits = [1, 1, 0]
    targets = ["i1", "i2", "missing"]
    histories = [["h1"], ["h1"], ["h1"]]
    mostpop_hit_by_user = {"u1": 0.0, "u2": 1.0, "u3": 0.0}

    values = g1_beyond_metric_values(
        material,
        stats,
        top_lists,
        ranks,
        hits,
        targets,
        histories,
        mostpop_hit_by_user,
    )

    assert values["serendipity_at_5"][0] == 1 / 3
