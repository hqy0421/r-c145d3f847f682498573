"""``read_ranges``: aggregate overlapping ranges, deduped by stable series id.

A run report rolls site data up per park while also publishing a few
sub-ranges as their own entries, so one series is selected by several entries
at once. These tests pin the three properties that make that correct:

* **Stable-id dedupe.** A series reachable through the park *and* through its
  own sub-range is one member of each entry, and the ClickHouse read happens
  once over the union of member ids — not once per entry.
* **An honest denominator.** ``effective_count`` on each aggregate row is the
  number of distinct series that actually *contributed* to it. The summary
  splits ``resolved`` into ``effective`` + ``zero_observation``.
* **Unconfirmed members stay structural.** A missing range root, or an
  instrumented node with no matching series, is retained on
  ``RangeRead.unconfirmed`` and counted in its own summary column. It never
  enters ``zero_observation``: "we could not establish this member" and "this
  member reported nothing" are different facts and must not share a
  denominator.

Skipped if ``TIMEDB_PG_DSN`` / ``TIMEDB_CH_URL`` are not set.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl
import pytest

import energydb as edb
from energydb import Client, RangeRead
from energydb.errors import NodeNotFoundError, SeriesNotFoundError, ValidationError

if not (os.environ.get("TIMEDB_PG_DSN") and os.environ.get("TIMEDB_CH_URL")):
    pytest.skip(
        "TIMEDB_PG_DSN / TIMEDB_CH_URL not set: skipping range-read tests",
        allow_module_level=True,
    )


BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _hours(n: int) -> list[datetime]:
    return [BASE + timedelta(hours=i) for i in range(n)]


def _write(client: Client, rows: list[tuple[str, list[float]]]) -> None:
    records = [
        {
            "path": path,
            "data_type": "actual",
            "name": "power",
            "valid_time": BASE + timedelta(hours=i),
            "value": value,
        }
        for path, values in rows
        for i, value in enumerate(values)
    ]
    client.write(pl.DataFrame(records))


@pytest.fixture
def client():
    """``P/ParkA`` (two reporting sites), ``P/ParkB`` (one), ``P/ParkC`` (silent)."""
    c = Client()
    c.delete()
    c.create()
    c.register_tree(
        edb.Portfolio(
            name="P",
            members=[
                edb.Site(name="ParkA", members=[edb.Sensor(name="S1"), edb.Sensor(name="S2")]),
                edb.Site(name="ParkB", members=[edb.Sensor(name="S3")]),
                edb.Site(name="ParkC", members=[edb.Sensor(name="S4")]),
            ],
        )
    )
    for path in ("P/ParkA/S1", "P/ParkA/S2", "P/ParkB/S3", "P/ParkC/S4"):
        c.get_node(path).register_series(
            name="power",
            canonical_unit="MW",
            data_type="actual",
            timeseries_type="FLAT",
            retention="forever",
        )
    # ParkC/S4 is registered but never reports: the zero-observation member.
    _write(
        c,
        [
            ("P/ParkA/S1", [1.0, 2.0]),
            ("P/ParkA/S2", [10.0, 20.0]),
            ("P/ParkB/S3", [100.0, 200.0]),
        ],
    )
    yield c
    c.delete()
    c.close()


# ---------------------------------------------------------------------------
# Dedupe by stable series id
# ---------------------------------------------------------------------------


def test_overlapping_entries_count_each_series_once(client):
    """The park and its sub-range both select S1; each entry sees it once."""
    result = client.read_ranges(
        {"park": client.get_node("P/ParkA"), "site1": "P/ParkA/S1"},
        agg="mean",
    )

    park = result.data.filter(pl.col("entry") == "park").sort("valid_time")
    site1 = result.data.filter(pl.col("entry") == "site1").sort("valid_time")

    # ParkA averages S1 and S2; the sub-range reports S1 alone. S1 is not
    # double-counted into the park's mean (which would be 5.5 → 7.0).
    assert park["value"].to_list() == [5.5, 11.0]
    assert park["effective_count"].to_list() == [2, 2]
    assert site1["value"].to_list() == [1.0, 2.0]
    assert site1["effective_count"].to_list() == [1, 1]


def test_one_selection_listed_twice_is_still_one_member(client):
    """A site named by both the park path and its own path is one member."""
    result = client.read_ranges({"all": ["P/ParkA", "P/ParkA/S1", "P/ParkA/S2"]}, agg="sum")

    assert result.summary.row(0, named=True)["resolved"] == 2
    assert result.data["value"].to_list() == [11.0, 22.0]
    assert result.data["effective_count"].to_list() == [2, 2]


def test_values_are_read_once_over_the_union_of_series(client, monkeypatch):
    """Overlap must not cost extra ClickHouse work or extra ids."""
    calls: list[list[int]] = []
    real_read = client.td.read

    def counting_read(*, series_ids, **kwargs):
        calls.append(list(series_ids))
        return real_read(series_ids=series_ids, **kwargs)

    monkeypatch.setattr(client.td, "read", counting_read)
    client.read_ranges(
        {
            "park": "P/ParkA",
            "site1": "P/ParkA/S1",
            "site2": "P/ParkA/S2",
            "parkB": "P/ParkB",
        }
    )

    assert len(calls) == 1, "one read for the whole entry set"
    assert len(calls[0]) == len(set(calls[0])) == 3, "three distinct series, S1/S2 appear once"


# ---------------------------------------------------------------------------
# The denominator: effective vs zero-observation vs unconfirmed
# ---------------------------------------------------------------------------


def test_effective_count_is_contributing_series_not_resolved_series(client):
    """ParkC's registered-but-silent series must not inflate any aggregate."""
    result = client.read_ranges({"silent": "P/ParkC", "parkA": "P/ParkA"})

    silent = result.summary.filter(pl.col("entry") == "silent").row(0, named=True)
    assert silent["resolved"] == 1
    assert silent["effective"] == 0
    assert silent["zero_observation"] == 1, "resolved but no rows, counted on its own"
    assert silent["unconfirmed"] == 0
    assert silent["complete"] is True
    # No aggregate row exists for an entry that contributed nothing.
    assert result.data.filter(pl.col("entry") == "silent").is_empty()


def test_resolved_is_split_between_effective_and_zero_observation(client):
    """The two denominators are disjoint and exhaust the resolved members."""
    result = client.read_ranges({"parkA": "P/ParkA", "parkC": "P/ParkC", "parkB": "P/ParkB"})

    for row in result.summary.iter_rows(named=True):
        assert row["resolved"] == row["effective"] + row["zero_observation"]
        assert row["unconfirmed"] == 0
        assert row["complete"] is True

    parks = {row["entry"]: row for row in result.summary.iter_rows(named=True)}
    assert parks["parkA"]["effective"] == 2
    assert parks["parkC"]["effective"] == 0


def test_unconfirmed_member_is_retained_and_outside_the_denominators(client):
    """A missing range root is retained structurally, never as a zero observation."""
    result = client.read_ranges({"ghost": "P/NoSuchPark", "parkC": "P/ParkC"})

    ghost = result.summary.filter(pl.col("entry") == "ghost").row(0, named=True)
    assert ghost["resolved"] == 0
    assert ghost["effective"] == 0
    assert ghost["zero_observation"] == 0, "unconfirmed must not land in the zero bucket"
    assert ghost["unconfirmed"] == 1
    assert ghost["complete"] is False

    gap = result.unconfirmed.filter(pl.col("entry") == "ghost").row(0, named=True)
    assert gap["path"] == "P/NoSuchPark"
    assert gap["reason"] == "range not found"
    assert result.complete is False


def test_instrumented_node_without_the_metric_is_unconfirmed(client):
    """A site that has series, but none of the requested metric, is a gap."""
    result = client.read_ranges({"parkA": "P/ParkA"}, data_type="forecast")

    row = result.summary.row(0, named=True)
    assert row == {
        "entry": "parkA",
        "resolved": 0,
        "effective": 0,
        "zero_observation": 0,
        "unconfirmed": 2,
        "complete": False,
    }
    reasons = set(result.unconfirmed["reason"].to_list())
    assert reasons == {"no matching series"}
    assert set(result.unconfirmed["path"].to_list()) == {"P/ParkA/S1", "P/ParkA/S2"}
    assert set(result.unconfirmed["data_type"].to_list()) == {"forecast"}


def test_structural_node_is_not_a_member(client):
    """A park with no series at all was never expected to report, so it is neither
    an observation nor a gap."""
    result = client.read_ranges({"park": "P"})

    row = result.summary.row(0, named=True)
    assert row["resolved"] == 4  # every site's series, all four registered
    assert row["effective"] == 3  # ParkC/S4 reports nothing
    assert row["zero_observation"] == 1
    assert row["unconfirmed"] == 0
    assert result.unconfirmed.is_empty()


def test_complete_read_reports_complete(client):
    result = client.read_ranges({"parkA": "P/ParkA", "parkB": "P/ParkB"})

    assert result.complete is True
    assert result.unconfirmed.is_empty()
    assert set(result.unconfirmed.columns) == {"entry", "path", "data_type", "name", "reason"}


# ---------------------------------------------------------------------------
# on_missing: default report, opt-in strictness
# ---------------------------------------------------------------------------


def test_raise_mode_names_a_missing_range_root(client):
    with pytest.raises(NodeNotFoundError) as excinfo:
        client.read_ranges({"ghost": "P/NoSuchPark"}, on_missing="raise")

    assert excinfo.value.path == "P/NoSuchPark"


def test_raise_mode_names_every_unmatched_member(client):
    with pytest.raises(SeriesNotFoundError) as excinfo:
        client.read_ranges({"parkA": "P/ParkA"}, data_type="forecast", on_missing="raise")

    assert excinfo.value.route == "path"
    assert {key[0] for key in excinfo.value.missing} == {"P/ParkA/S1", "P/ParkA/S2"}


def test_raise_mode_passes_a_complete_read(client):
    result = client.read_ranges({"parkA": "P/ParkA"}, on_missing="raise")

    assert result.summary.row(0, named=True)["complete"] is True


# ---------------------------------------------------------------------------
# Output shapes and aggregation modes
# ---------------------------------------------------------------------------


def test_return_value_is_a_named_tuple(client):
    result = client.read_ranges({"parkA": "P/ParkA"})

    data, summary, unconfirmed = result
    assert isinstance(result, RangeRead)
    assert data is result.data
    assert summary is result.summary
    assert unconfirmed is result.unconfirmed


def test_frame_output_columns(client):
    result = client.read_ranges({"parkA": "P/ParkA"})

    assert list(result.data.columns) == ["entry", "valid_time", "value", "effective_count"]
    assert result.data["entry"].unique().to_list() == ["parkA"]
    assert result.data["valid_time"].dtype == pl.Datetime("us", "UTC")
    assert result.data["effective_count"].dtype == pl.Int64


def test_by_entry_output_carries_every_entry(client):
    result = client.read_ranges(
        {"parkA": "P/ParkA", "silent": "P/ParkC"},
        output="by_entry",
        agg="mean",
    )

    assert set(result.data) == {"parkA", "silent"}
    assert list(result.data["parkA"].columns) == ["valid_time", "value", "effective_count"]
    assert result.data["parkA"]["value"].to_list() == [5.5, 11.0]
    # An entry with no observations still gets its documented empty shape.
    assert result.data["silent"].is_empty()
    assert result.data["silent"].columns == ["valid_time", "value", "effective_count"]


def test_pandas_backend_converts_all_three_frames(client):
    result = client.read_ranges({"parkA": "P/ParkA", "ghost": "P/Nope"}, backend="pandas")

    assert isinstance(result.data, pd.DataFrame)
    assert isinstance(result.summary, pd.DataFrame)
    assert isinstance(result.unconfirmed, pd.DataFrame)
    assert result.complete is False


def test_pandas_backend_by_entry(client):
    result = client.read_ranges({"parkA": "P/ParkA"}, output="by_entry", backend="pandas")

    assert isinstance(result.data, dict)
    assert isinstance(result.data["parkA"], pd.DataFrame)


def test_bucket_averages_each_series_before_combining(client):
    """A series reporting twice in one bucket must not be weighted twice: the
    bucket mean is over *sites*, not over rows."""
    client.get_node("P/ParkA/S1").register_series(
        name="load", canonical_unit="MW", data_type="actual", timeseries_type="FLAT", retention="forever"
    )
    client.write(
        pl.DataFrame(
            {
                "path": ["P/ParkA/S1"] * 4,
                "data_type": ["actual"] * 4,
                "name": ["load"] * 4,
                "valid_time": [
                    BASE + timedelta(minutes=10),
                    BASE + timedelta(minutes=20),
                    BASE + timedelta(minutes=70),
                    BASE + timedelta(minutes=80),
                ],
                "value": [1.0, 3.0, 5.0, 7.0],
            }
        )
    )

    result = client.read_ranges(
        {"site1": "P/ParkA/S1"},
        name="load",
        bucket_us=3600 * 1_000_000,
        agg="mean",
    )

    # Two rows per bucket, one site: each bucket averages its two readings.
    assert result.data["valid_time"].to_list() == [BASE, BASE + timedelta(hours=1)]
    assert result.data["value"].to_list() == [2.0, 6.0]
    assert result.data["effective_count"].to_list() == [1, 1]


def test_bucket_mean_is_unweighted_across_series(client):
    """S1 reports twice in the bucket, S2 once; the mean is over the two series."""
    client.get_node("P/ParkA/S1").register_series(
        name="load", canonical_unit="MW", data_type="actual", timeseries_type="FLAT", retention="forever"
    )
    client.get_node("P/ParkA/S2").register_series(
        name="load", canonical_unit="MW", data_type="actual", timeseries_type="FLAT", retention="forever"
    )
    client.write(
        pl.DataFrame(
            {
                "path": ["P/ParkA/S1", "P/ParkA/S1", "P/ParkA/S2"],
                "data_type": ["actual"] * 3,
                "name": ["load"] * 3,
                "valid_time": [BASE + timedelta(minutes=10), BASE + timedelta(minutes=20), BASE + timedelta(minutes=30)],
                "value": [2.0, 4.0, 10.0],
            }
        )
    )

    result = client.read_ranges({"park": "P/ParkA"}, name="load", bucket_us=3600 * 1_000_000, agg="mean")

    # S1 → 3, S2 → 10, then mean = 6.5 (not (2+4+10)/3 = 5.33).
    assert result.data["value"].to_list() == [6.5]
    assert result.data["effective_count"].to_list() == [2]



@pytest.mark.parametrize(
    ("agg", "expected"),
    [("sum", [11.0, 22.0]), ("mean", [5.5, 11.0]), ("min", [1.0, 2.0]), ("max", [10.0, 20.0]), ("count", [2.0, 2.0])],
)
def test_aggregation_modes(client, agg, expected):
    result = client.read_ranges({"parkA": "P/ParkA"}, agg=agg)

    assert result.data["value"].to_list() == expected


def test_count_counts_observations_not_series(client):
    """``agg="count"`` counts rows; ``effective_count`` counts distinct series.
    They coincide here, and diverge as soon as any series has more than one row
    per valid_time — the docs call that out, so pin the distinction."""
    result = client.read_ranges({"parkA": "P/ParkA"}, agg="count")

    assert result.data["value"].to_list() == [2.0, 2.0]
    assert result.data["effective_count"].to_list() == [2, 2]
    assert result.summary.row(0, named=True)["effective"] == 2


def test_valid_time_window_filters_values_but_keeps_the_denominator(client):
    """A series with no rows *in the window* is a zero observation, not a gap."""
    result = client.read_ranges(
        {"parkA": "P/ParkA"},
        start_valid=BASE + timedelta(days=30),
        end_valid=BASE + timedelta(days=31),
    )

    assert result.data.is_empty()
    row = result.summary.row(0, named=True)
    assert row["resolved"] == 2
    assert row["effective"] == 0
    assert row["zero_observation"] == 2
    assert row["unconfirmed"] == 0


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_unit_converts_each_series_before_aggregating(client):
    result = client.read_ranges({"parkA": "P/ParkA"}, unit="kW", agg="sum")

    # 1 MW + 10 MW = 11 MW = 11_000 kW.
    assert result.data["value"].to_list() == [11_000.0, 22_000.0]


def test_mixed_canonical_units_without_a_target_raise(client):
    client.delete()
    client.create()
    client.register_tree(
        edb.Portfolio(
            name="P",
            members=[edb.Site(name="Park", members=[edb.Sensor(name="A"), edb.Sensor(name="B")])],
        )
    )
    for path, unit in (("P/Park/A", "MW"), ("P/Park/B", "kW")):
        client.get_node(path).register_series(
            name="power", canonical_unit=unit, data_type="actual", timeseries_type="FLAT", retention="forever"
        )
    _write(client, [("P/Park/A", [1.0]), ("P/Park/B", [500.0])])

    with pytest.raises(ValidationError, match="different canonical units"):
        client.read_ranges({"park": "P/Park"})

    result = client.read_ranges({"park": "P/Park"}, unit="MW")
    assert result.data["value"].to_list() == [1.5]


# ---------------------------------------------------------------------------
# Selection resolution
# ---------------------------------------------------------------------------


def test_uuid_addressed_scope_resolves(client):
    node_uuid = client.get_node("P/ParkA/S1").get_raw()["uuid"]
    result = client.read_ranges({"site1": client.get_node(uuid=node_uuid)})

    assert result.data["value"].to_list() == [1.0, 2.0]
    assert result.summary.row(0, named=True)["resolved"] == 1


def test_where_filter_narrows_the_selected_subtree(client):
    result = client.read_ranges({"c_only": client.get_node("P").where(name="S3")})

    assert result.summary.row(0, named=True)["resolved"] == 1
    assert result.data["value"].to_list() == [100.0, 200.0]


def test_uuid_addressed_missing_root_is_unconfirmed(client):
    from uuid import uuid4

    result = client.read_ranges({"ghost": client.get_node(uuid=uuid4())})

    gap = result.unconfirmed.row(0, named=True)
    assert gap["reason"] == "range not found"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_empty_ranges_is_rejected(client):
    with pytest.raises(ValidationError, match="at least one range entry"):
        client.read_ranges({})


def test_duplicate_entry_names_are_rejected(client):
    with pytest.raises(ValidationError, match="Duplicate range entry name"):
        client.read_ranges([("park", "P/ParkA"), ("park", "P/ParkB")])


def test_non_string_entry_name_is_rejected(client):
    with pytest.raises(ValidationError, match="non-empty strings"):
        client.read_ranges({1: "P/ParkA"})


def test_unknown_selection_type_is_rejected(client):
    with pytest.raises(ValidationError, match="must be a NodeScope"):
        client.read_ranges({"park": 42})


def test_edge_scope_is_rejected_with_a_pointer(client):
    with pytest.raises(ValidationError, match="is an EdgeScope"):
        client.read_ranges({"edge": client.get_edge("P/ParkA/S1", "P/ParkB/S3", type="line")})


def test_bad_output_and_agg_are_rejected(client):
    with pytest.raises(ValidationError, match="output must be"):
        client.read_ranges({"park": "P/ParkA"}, output="bogus")
    with pytest.raises(ValidationError, match="agg must be one of"):
        client.read_ranges({"park": "P/ParkA"}, agg="median")
    with pytest.raises(ValidationError, match="on_missing must be"):
        client.read_ranges({"park": "P/ParkA"}, on_missing="ignore")


def test_scope_from_another_client_is_rejected(client):
    other = Client()
    try:
        other.delete()
        other.create()
        with pytest.raises(ValidationError, match="different client"):
            client.read_ranges({"park": other.get_node("P/ParkA")})
    finally:
        other.delete()
        other.close()
