"""Range reads: aggregate several (possibly overlapping) subtrees in one read.

A run report usually rolls site sensor data up **per park** while also
publishing a few sub-ranges as their own entries — the same site shows up
under the park *and* under its own heading. Two things make that correct and
cheap:

* **Identity is the stable ``series_id``.** Ranges are resolved to per-series
  membership in PostgreSQL, then every distinct series is read from ClickHouse
  *once*; the members of the requested entries are joined back onto that one
  result. A series selected by both the park and the sub-range therefore
  contributes exactly one value to each entry, and never twice to either —
  resolving by ``path`` / ``name`` instead would make two nodes that happen to
  share a name collide, and reading per range would pay for the overlap twice.
* **The aggregate carries its own denominator.** Every aggregate row reports
  ``effective_count``: the number of *distinct* series that actually produced an
  observation in that bucket. Resolved-but-empty series (the site reports
  nothing in the window) are counted separately, and members that could not be
  confirmed at all are retained structurally instead of being folded into
  either count. See :class:`RangeRead`.

Both facts matter for a report: "average across the park" over an unknown
number of contributing sites is not a number anyone can defend, and silently
treating a site that was never instrumented the same as one reporting zeros
makes a gap look like a measurement.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from uuid import UUID

import polars as pl

from energydb._frames import Backend, to_backend
from energydb._io import apply_per_series_unit
from energydb.errors import NodeNotFoundError, SeriesNotFoundError, ValidationError
from energydb.models import SQL_SCHEMA_PREFIX as P
from energydb.paths import (
    _MISSING_PREVIEW,
    OnMissing,
    _check_on_missing,
    build_filter_conditions,
    derived_prefix_like,
)
from energydb.scope import EdgeScope, NodeScope

if TYPE_CHECKING:
    import pandas as pd

    from energydb.client import AsyncClient

RangeOutput = Literal["frame", "by_entry"]
_RANGE_OUTPUTS = ("frame", "by_entry")

Aggregation = Literal["sum", "mean", "min", "max", "count"]
_AGGREGATIONS = ("sum", "mean", "min", "max", "count")

_ENTRY = "entry"
_TIME = "valid_time"
_VALUE = "value"
_EFFECTIVE = "effective_count"

_DATA_SCHEMA: dict[str, pl.DataType] = {
    _ENTRY: pl.Utf8,
    _TIME: pl.Datetime("us", "UTC"),
    _VALUE: pl.Float64,
    _EFFECTIVE: pl.Int64,
}

# Internal aggregate shape: keyed by entry position until the final rename, so
# entries may be emitted in the caller's order regardless of lexical sorting.
_AGG_SCHEMA: dict[str, pl.DataType] = {
    "entry_idx": pl.Int32,
    _TIME: pl.Datetime("us", "UTC"),
    _VALUE: pl.Float64,
    _EFFECTIVE: pl.Int64,
}

_UNCONFIRMED_SCHEMA: dict[str, pl.DataType] = {
    _ENTRY: pl.Utf8,
    "path": pl.Utf8,
    "data_type": pl.Utf8,
    "name": pl.Utf8,
    "reason": pl.Utf8,
}

_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    _ENTRY: pl.Utf8,
    "resolved": pl.Int64,
    "effective": pl.Int64,
    "zero_observation": pl.Int64,
    "unconfirmed": pl.Int64,
    "complete": pl.Boolean,
}

# Why a member could not be confirmed. Free text, but stable so callers can
# filter on it; both mean "we could not establish this member's reading".
_REASON_RANGE_NOT_FOUND = "range not found"
_REASON_NO_MATCHING_SERIES = "no matching series"


class RangeRead(NamedTuple):
    """The aggregates for every range entry, plus what could not be confirmed.

    Returned by :meth:`~energydb.AsyncClient.read_ranges`. Three frames, all
    following ``backend=`` like :class:`~energydb.ReadResult`:

    ``data``
        The aggregate rows: ``(entry, valid_time, value, effective_count)``,
        in the caller's entry order, then by ``valid_time``.
        ``effective_count`` is the number of *distinct* series that contributed
        an observation to that row — never the number of series that merely
        resolved. Under
        ``output="by_entry"`` this is a ``{entry: DataFrame}`` dict whose
        sub-frames carry ``(valid_time, value, effective_count)``; every entry
        is present, with the documented schema even when it has no rows.

    ``summary``
        One row per entry: ``(entry, resolved, effective, zero_observation,
        unconfirmed, complete)``. The denominators are disjoint by
        construction — ``resolved == effective + zero_observation`` — and
        ``unconfirmed`` sits *outside* them: an unconfirmed member was never
        counted as a zero observation. ``complete`` is ``unconfirmed == 0``.

    ``unconfirmed``
        The members that were selected but could not be confirmed, one row
        each: ``(entry, path, data_type, name, reason)``. ``reason`` is
        ``"range not found"`` (the addressed node does not exist) or
        ``"no matching series"`` (the node carries series, but none matching
        the requested ``data_type`` / ``name``). ``path`` is the addressed
        node path, or the uuid string for a uuid-addressed range whose node
        does not exist. Zero-row with the right schema when everything was
        confirmed.

    A :class:`~typing.NamedTuple`, so ``data, summary, unconfirmed =
    client.read_ranges(...)`` unpacks.
    """

    data: (
        pl.DataFrame
        | pd.DataFrame
        | dict[str, pl.DataFrame]
        | dict[str, pd.DataFrame]
    )
    summary: pl.DataFrame | pd.DataFrame
    unconfirmed: pl.DataFrame | pd.DataFrame

    @property
    def complete(self) -> bool:
        """True when no member was left unconfirmed."""
        if not isinstance(self.unconfirmed, pl.DataFrame):
            # pandas backend: height is the same fact on both.
            return len(self.unconfirmed) == 0
        return self.unconfirmed.is_empty()


class _Root(NamedTuple):
    """One resolved-address root of one entry (a selection may have several)."""

    entry_idx: int
    root_path: str | None
    rel_path: str | None
    root_uuid: UUID | None
    where_conds: tuple[str, ...]
    where_params: tuple[Any, ...]


def _check_output(output: str) -> None:
    if output not in _RANGE_OUTPUTS:
        raise ValidationError(f"output must be 'frame' or 'by_entry', got {output!r}")


def _check_agg(agg: str) -> None:
    if agg not in _AGGREGATIONS:
        raise ValidationError(f"agg must be one of {list(_AGGREGATIONS)}, got {agg!r}")


def _entry_items(ranges: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Normalize the ``ranges`` argument to ordered ``(name, selection)`` pairs.

    A ``Mapping`` gives insertion order, which is the report's entry order; a
    sequence of pairs is accepted when the caller builds the list
    programmatically. Names must be non-empty and unique: they are the entry
    identity in every returned frame.
    """
    items = list(ranges.items()) if isinstance(ranges, Mapping) else list(ranges)
    if not items:
        raise ValidationError("read_ranges needs at least one range entry.")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValidationError(
                "read_ranges takes a Mapping {entry: selection} or a sequence of (entry, selection) pairs; "
                f"got item {item!r}."
            )
        name = item[0]
        if not isinstance(name, str) or not name:
            raise ValidationError(f"Range entry names must be non-empty strings; got {name!r}.")
        if name in seen:
            raise ValidationError(f"Duplicate range entry name {name!r}; entry names identify rows and must be unique.")
        seen.add(name)
    return items


def _scopes_for(client: AsyncClient, name: str, selection: Any) -> list[NodeScope]:
    """Expand one entry's selection into node scopes owned by ``client``.

    A string is itself a sequence, so it is tested first: an entry selecting
    several sites is a *sequence* of scopes/paths, while one string is a single
    ``/``-joined path. Anything else is diagnosed by :func:`_as_scope`.
    """
    if isinstance(selection, Sequence) and not isinstance(selection, str):
        scopes = [_as_scope(client, name, item) for item in selection]
        if not scopes:
            raise ValidationError(f"Range entry {name!r} selects no scopes.")
        return scopes
    return [_as_scope(client, name, selection)]


def _as_scope(client: AsyncClient, name: str, selection: Any) -> NodeScope:
    if isinstance(selection, str):
        return client.get_node(selection)
    if isinstance(selection, EdgeScope):
        raise ValidationError(
            f"Range entry {name!r} is an EdgeScope. Ranges aggregate node (site) series; "
            f"address an edge's series directly with client.read(...)."
        )
    if isinstance(selection, NodeScope):
        if selection._client._pool is not client._pool:
            raise ValidationError(
                f"Range entry {name!r} was built by a different client; build it with "
                f"client.get_node(...) on this client."
            )
        if selection._txn is not None:
            raise ValidationError(
                f"Range entry {name!r} is bound to a transaction(), whose connection a range read "
                f"does not use. Call read_ranges() outside the transaction block."
            )
        return selection
    raise ValidationError(
        f"Range entry {name!r} must be a NodeScope or a '/' -joined path; got {type(selection).__name__}. "
        f"Ranges aggregate node (site) series; address edges directly instead."
    )


def _root_for(entry_idx: int, scope: NodeScope, name: str) -> _Root:
    """Turn a scope's accumulated address into one SQL root descriptor."""
    if scope._path and scope._node_uuid is None:
        root_path: str | None = "/".join(scope._path)
        rel_path: str | None = None
        root_uuid: UUID | None = None
    elif scope._node_uuid is not None:
        root_path = None
        rel_path = "/".join(scope._path) or None
        root_uuid = scope._node_uuid
    else:
        raise ValidationError(f"Range entry {name!r} has no path or uuid to resolve.")
    if scope._where_filters:
        conds, params = build_filter_conditions(scope._where_filters, type_col="node_type", table_alias="n")
    else:
        conds, params = [], []
    return _Root(entry_idx, root_path, rel_path, root_uuid, tuple(conds), tuple(params))


def _members_sql(*, node_conds: Sequence[str], series_conds: Sequence[str]) -> str:
    """One statement: every (root, node, matching series) row for a root group.

    A ``LEFT JOIN`` all the way down, so a root that does not exist and a node
    with no matching series still come back — that is what lets the caller
    retain them as unconfirmed instead of silently losing them. ``root_exists``
    separates "the addressed node is missing" from "the filter matched nothing
    under an existing node".
    """
    node_filter = f" AND ({' AND '.join(node_conds)})" if node_conds else ""
    series_filter = f" AND {' AND '.join(series_conds)}" if series_conds else ""
    # derived_prefix_like emits psycopg-escaped '%', so this string is only ever
    # handed to conn.execute as a parameterized query.
    return rf"""
        WITH raw AS (
            SELECT * FROM unnest(%s::int[], %s::text[], %s::text[], %s::uuid[])
                AS t(entry_idx, root_path, rel_path, root_uuid)
        ),
        roots AS (
            SELECT raw.entry_idx,
                   raw.root_path,
                   raw.root_uuid,
                   COALESCE(
                       raw.root_path,
                       CASE WHEN raw.rel_path IS NULL THEN n0.path
                            ELSE n0.path || '/' || raw.rel_path END
                   ) AS resolved_path
            FROM raw
            LEFT JOIN {P}node n0 ON n0.uuid = raw.root_uuid
        )
        SELECT rs.entry_idx,
               rs.root_path,
               rs.resolved_path,
               rs.root_uuid::text,
               (rs.resolved_path IS NOT NULL
                AND EXISTS (SELECT 1 FROM {P}node x WHERE x.path = rs.resolved_path)) AS root_exists,
               n.path AS node_path,
               CASE WHEN s.series_id IS NULL
                    THEN EXISTS (SELECT 1 FROM {P}series sx WHERE sx.node_uuid = n.uuid)
                    ELSE TRUE
               END AS instrumented,
               s.series_id, s.canonical_unit, s.retention
        FROM roots rs
        LEFT JOIN {P}node n
               ON rs.resolved_path IS NOT NULL
              AND (n.path = rs.resolved_path
                   OR n.path LIKE {derived_prefix_like("rs.resolved_path")} ESCAPE '\')
              {node_filter}
        LEFT JOIN {P}series s ON s.node_uuid = n.uuid {series_filter}
    """


async def _fetch_members(
    conn,
    roots: Sequence[_Root],
    *,
    data_type: str | None,
    name: str | None,
) -> list[tuple]:
    """Resolve every root to membership rows in as few statements as possible.

    Roots sharing one ``.where()`` predicate set go in one statement (the
    common, unfiltered case is exactly one round-trip); distinct predicates
    each get their own statement on the same connection. Returns the raw rows
    of :func:`_members_sql`.
    """
    series_conds: list[str] = []
    series_params: list[Any] = []
    if data_type is not None:
        series_conds.append("s.data_type = %s")
        series_params.append(data_type)
    if name is not None:
        series_conds.append("s.name = %s")
        series_params.append(name)

    groups: dict[tuple, list[_Root]] = {}
    for root in roots:
        groups.setdefault((root.where_conds, root.where_params), []).append(root)

    rows: list[tuple] = []
    for (node_conds, node_params), group in groups.items():
        sql = _members_sql(node_conds=node_conds, series_conds=series_conds)
        params = [
            [r.entry_idx for r in group],
            [r.root_path for r in group],
            [r.rel_path for r in group],
            [r.root_uuid for r in group],
            *node_params,
            *series_params,
        ]
        rows.extend(await (await conn.execute(sql, params)).fetchall())
    return rows


def _unconfirmed_frame(
    rows: Sequence[tuple],
    *,
    entries: Sequence[str],
    data_type: str | None,
    name: str | None,
) -> pl.DataFrame:
    """Extract the members that could not be confirmed, keyed by entry name.

    Two sources, both retained: a root that does not exist, and an instrumented
    node with no series matching the requested metric. A plain structural node
    (portfolio, park, site with no series at all) is *not* a member and is
    skipped — nothing was expected from it.
    """
    records: dict[tuple, dict[str, Any]] = {}
    for (
        entry_idx,
        root_path,
        resolved_path,
        root_uuid,
        root_exists,
        node_path,
        instrumented,
        series_id,
        _unit,
        _retention,
    ) in rows:
        entry = entries[entry_idx]
        if not root_exists:
            # Prefer the human-readable address: the path it was addressed by,
            # else the path the uuid resolved to (+ relative suffix), else the
            # raw uuid for a uuid that itself does not exist.
            addressed = root_path or resolved_path or root_uuid
            key = (entry, addressed, _REASON_RANGE_NOT_FOUND)
            records.setdefault(
                key,
                {
                    _ENTRY: entry,
                    "path": addressed,
                    "data_type": data_type,
                    "name": name,
                    "reason": _REASON_RANGE_NOT_FOUND,
                },
            )
        elif node_path is not None and series_id is None and instrumented:
            key = (entry, node_path, _REASON_NO_MATCHING_SERIES)
            records.setdefault(
                key,
                {
                    _ENTRY: entry,
                    "path": node_path,
                    "data_type": data_type,
                    "name": name,
                    "reason": _REASON_NO_MATCHING_SERIES,
                },
            )
    if not records:
        return pl.DataFrame(schema=_UNCONFIRMED_SCHEMA)
    return pl.DataFrame(list(records.values()), schema=_UNCONFIRMED_SCHEMA).sort([_ENTRY, "path", "reason"])


def _raise_for_unconfirmed(unconfirmed: pl.DataFrame) -> None:
    """Strict mode: report the gaps through the existing typed errors.

    A missing range root is a bad address, so it raises
    :class:`~energydb.errors.NodeNotFoundError`; a member without a matching
    series raises :class:`~energydb.errors.SeriesNotFoundError`, carrying every
    gap in ``.missing`` like the manifest path route does. Both can occur in
    one call; the missing root wins, since it explains the rest.
    """
    if unconfirmed.is_empty():
        return
    missing_roots = unconfirmed.filter(pl.col("reason") == _REASON_RANGE_NOT_FOUND)
    if not missing_roots.is_empty():
        first = missing_roots.row(0, named=True)
        raise NodeNotFoundError(f"Node not found: {first['path']}", path=first["path"])
    triples = [
        (str(row["path"]), row["data_type"] or "", row["name"] or "")
        for row in unconfirmed.iter_rows(named=True)
    ]
    rendered = [_render_member(t) for t in triples[:_MISSING_PREVIEW]]
    message = f"Series not registered for {rendered[0]}."
    if len(triples) > 1:
        rest, remaining = rendered[1:], len(triples) - 1
        truncated = f" ({len(rest)} of {remaining} shown)" if remaining > len(rest) else ""
        message += f" {len(triples)} range members are unresolved; also {'; '.join(rest)}{truncated}."
    # Same (owner, data_type, name) shape as the manifest path route: the
    # unconfirmed frame identifies a member by its path (or uuid string).
    raise SeriesNotFoundError(message, route="path", missing=triples)


def _render_member(triple: tuple[str, str, str]) -> str:
    path, data_type, name = triple
    parts = [f"path={path!r}"]
    if data_type:
        parts.append(f"data_type={data_type!r}")
    if name:
        parts.append(f"name={name!r}")
    return ", ".join(parts)


def _bucket(values: pl.DataFrame, bucket_us: int) -> pl.DataFrame:
    """Floor ``valid_time`` to a bucket, client-side.

    Done here rather than through timedb's ``bucket_us`` so the aggregate works
    on every timedb release this library supports; the bucket mean timedb
    computes is identical (one resolved row per series per valid_time).
    """
    if bucket_us <= 0:
        raise ValidationError(f"bucket_us must be a positive number of microseconds, got {bucket_us!r}.")
    epoch_us = pl.col(_TIME).dt.epoch("us")
    floored = (epoch_us // bucket_us) * bucket_us
    return values.with_columns(floored.cast(pl.Datetime("us", "UTC")).alias(_TIME))


_STAGE1_EXPRS = {
    "sum": lambda: pl.col(_VALUE).sum(),
    "mean": lambda: pl.col(_VALUE).mean(),
    "min": lambda: pl.col(_VALUE).min(),
    "max": lambda: pl.col(_VALUE).max(),
    "count": lambda: pl.col(_VALUE).len().cast(pl.Float64),
}

# Across series, "count" pools the per-series observation counts; every other
# mode applies the same reduction again.
_STAGE2_EXPRS = {**_STAGE1_EXPRS, "count": lambda: pl.col(_VALUE).sum()}


def _aggregate(joined: pl.DataFrame, agg: str) -> pl.DataFrame:
    """Collapse per-series observations into one row per entry × valid_time.

    Two stages, on purpose. First each series is reduced to one value per
    ``(series, valid_time)`` (identity unless ``bucket_us`` grouped several of
    its rows into one bucket), then those per-series values are reduced across
    the entry's series. Without the first stage a site that reports twice in a
    bucket would be weighted twice in the park's mean — the aggregate would
    depend on reporting frequency rather than on the sites.
    """
    if joined.is_empty():
        return pl.DataFrame(schema=_AGG_SCHEMA)
    per_series = joined.group_by(["entry_idx", "series_id", _TIME]).agg(
        _STAGE1_EXPRS[agg]().alias(_VALUE)
    )
    return (
        per_series.group_by(["entry_idx", _TIME])
        .agg(
            _STAGE2_EXPRS[agg]().alias(_VALUE),
            pl.col("series_id").n_unique().cast(pl.Int64).alias(_EFFECTIVE),
        )
        .sort(["entry_idx", _TIME])
    )


def _summary_frame(
    entries: Sequence[str],
    resolved: pl.DataFrame,
    effective: pl.DataFrame,
    unconfirmed: pl.DataFrame,
) -> pl.DataFrame:
    """Assemble the per-entry denominators, with zeros for entries that missed.

    ``resolved`` / ``effective`` are ``(entry_idx, series_count)`` frames; the
    difference is the zero-observation denominator. ``unconfirmed`` contributes
    its own column and never enters either count.
    """
    by_entry: dict[int, dict[str, int]] = {
        idx: {"resolved": 0, "effective": 0, "unconfirmed": 0} for idx in range(len(entries))
    }
    for row in resolved.iter_rows(named=True):
        by_entry[row["entry_idx"]]["resolved"] = int(row["n"])
    for row in effective.iter_rows(named=True):
        by_entry[row["entry_idx"]]["effective"] = int(row["n"])
    for row in unconfirmed.iter_rows(named=True):
        key = entries.index(row[_ENTRY])
        by_entry[key]["unconfirmed"] += 1
    records = []
    for idx, entry in enumerate(entries):
        counts = by_entry[idx]
        resolved_n = counts["resolved"]
        effective_n = counts["effective"]
        unconfirmed_n = counts["unconfirmed"]
        records.append(
            {
                _ENTRY: entry,
                "resolved": resolved_n,
                "effective": effective_n,
                "zero_observation": resolved_n - effective_n,
                "unconfirmed": unconfirmed_n,
                "complete": unconfirmed_n == 0,
            }
        )
    return pl.DataFrame(records, schema=_SUMMARY_SCHEMA)


async def read_ranges(
    client: AsyncClient,
    ranges: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    data_type: str | None = None,
    name: str | None = None,
    unit: str | None = None,
    agg: Aggregation = "sum",
    bucket_us: int | None = None,
    start_valid: datetime | None = None,
    end_valid: datetime | None = None,
    start_known: datetime | None = None,
    end_known: datetime | None = None,
    output: RangeOutput = "frame",
    backend: Backend = "polars",
    on_missing: OnMissing = "skip",
) -> RangeRead:
    """Aggregate several (possibly overlapping) node ranges, deduped by series.

    ``ranges`` maps an entry name to its selection: a :class:`~energydb.NodeScope`
    (e.g. ``client.get_node("Parks/ParkA")``), a ``/``-joined path, or a sequence
    of those for an entry that spans several sites. Entries may overlap freely:
    a series selected by both the park and one of its sites is resolved once,
    read once, and contributes to each entry exactly once.

    ``data_type`` / ``name`` narrow which series are members, exactly as on
    :meth:`~energydb.AsyncClient.read`. With them, an instrumented node (one
    that has *some* series) but none matching is retained as unconfirmed.

    Values are aggregated across an entry's series per ``valid_time`` (floored
    to ``bucket_us`` when given) with ``agg`` — one of ``"sum"`` (default),
    ``"mean"``, ``"min"``, ``"max"``, ``"count"``. Each series is reduced to one
    value per bucket *first*, then those are reduced across the entry's series,
    so a site reporting more often than its neighbours does not get extra
    weight in a mean. ``count`` counts observations (rows), which is not the
    same as ``effective_count`` (distinct series). ``unit`` converts each series
    from its canonical unit before aggregating; a read that would mix several
    canonical units without one raises
    :class:`~energydb.errors.ValidationError`, because such a sum is meaningless.

    ``on_missing`` defaults to ``"skip"`` — unlike
    :meth:`~energydb.AsyncClient.read` — because the return value *is* the
    completeness report: unconfirmed members are always retained on
    ``RangeRead.unconfirmed`` and counted in ``RangeRead.summary``, never
    silently dropped, so a default that raises would make a partial report
    impossible to produce. Pass ``"raise"`` to fail on the first gap instead
    (:class:`~energydb.errors.NodeNotFoundError` for a missing range root,
    :class:`~energydb.errors.SeriesNotFoundError` for an unregistered metric).

    See :class:`RangeRead` for the three returned frames.
    """
    _check_output(output)
    _check_agg(agg)
    _check_on_missing(on_missing)
    items = _entry_items(ranges)
    entries = [name for name, _ in items]

    roots: list[_Root] = []
    for entry_idx, (entry, selection) in enumerate(items):
        for scope in _scopes_for(client, entry, selection):
            roots.append(_root_for(entry_idx, scope, entry))

    data_type_filter = str(data_type).lower() if data_type else None

    async with client._read_conn() as conn:
        rows = await _fetch_members(conn, roots, data_type=data_type_filter, name=name)

    unconfirmed = _unconfirmed_frame(
        rows, entries=entries, data_type=data_type_filter, name=name
    )
    if on_missing == "raise":
        _raise_for_unconfirmed(unconfirmed)

    # Membership: one row per (entry, series_id). The dict key is the stable-id
    # dedupe — a series reachable through two roots of the same entry is still
    # one member of it. Raw row layout is (entry_idx, root_path, resolved_path,
    # root_uuid, root_exists, node_path, instrumented, series_id, canonical_unit,
    # retention); index access keeps this independent of named unpacking.
    member_records = {
        (row[0], row[7]): (row[0], row[7], row[8], row[9]) for row in rows if row[7] is not None
    }
    members = pl.DataFrame(
        list(member_records.values()),
        schema={
            "entry_idx": pl.Int32,
            "series_id": pl.Int64,
            "canonical_unit": pl.Utf8,
            "retention": pl.Utf8,
        },
        orient="row",
    )

    resolved_counts = (
        members.group_by("entry_idx")
        .agg(pl.len().cast(pl.Int64).alias("n"))
        if not members.is_empty()
        else pl.DataFrame({"entry_idx": [], "n": []}, schema={"entry_idx": pl.Int32, "n": pl.Int64})
    )

    values, effective_counts = await _read_values(
        client,
        members,
        unit=unit,
        agg=agg,
        bucket_us=bucket_us,
        start_valid=start_valid,
        end_valid=end_valid,
        start_known=start_known,
        end_known=end_known,
    )

    summary = _summary_frame(entries, resolved_counts, effective_counts, unconfirmed)

    if output == "by_entry":
        data: Any = _by_entry(values, entries)
    else:
        data = values.with_columns(
            pl.col("entry_idx")
            .replace_strict(list(range(len(entries))), list(entries), return_dtype=pl.Utf8)
            .alias(_ENTRY)
        ).select([_ENTRY, _TIME, _VALUE, _EFFECTIVE])
    return RangeRead(
        to_backend(data, backend),
        to_backend(summary, backend),
        to_backend(unconfirmed, backend),
    )


async def _read_values(
    client: AsyncClient,
    members: pl.DataFrame,
    *,
    unit: str | None,
    agg: str,
    bucket_us: int | None,
    start_valid: datetime | None,
    end_valid: datetime | None,
    start_known: datetime | None,
    end_known: datetime | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read every member series once, then aggregate per entry × valid_time.

    Returns ``(aggregate_rows, effective_per_entry)``. The ClickHouse read is a
    single call over the *union* of member ids — the dedupe that keeps an
    overlapping sub-range from paying for (and counting) its series twice.
    """
    empty_effective = pl.DataFrame({"entry_idx": [], "n": []}, schema={"entry_idx": pl.Int32, "n": pl.Int64})
    if members.is_empty():
        return pl.DataFrame(schema=_AGG_SCHEMA), empty_effective

    series_meta = members.select(["series_id", "canonical_unit"]).unique(subset=["series_id"])
    if unit is None:
        distinct_units = series_meta["canonical_unit"].unique().to_list()
        if len(distinct_units) > 1:
            raise ValidationError(
                "read_ranges would aggregate series with different canonical units "
                f"({', '.join(sorted(str(u) for u in distinct_units))}); pass unit=... to convert "
                "them to one unit before aggregating."
            )

    series_ids = series_meta["series_id"].to_list()
    retentions = members["retention"].unique().to_list()
    values = await asyncio.to_thread(
        client.td.read,
        series_ids=series_ids,
        retention=retentions,
        start_valid=start_valid,
        end_valid=end_valid,
        start_known=start_known,
        end_known=end_known,
    )
    if values.is_empty():
        return pl.DataFrame(schema=_AGG_SCHEMA), empty_effective

    values = values.with_columns(pl.col("series_id").cast(pl.Int64))
    # A null value is timedb's "no observation" (the NaN sentinel): it must not
    # inflate a denominator, so drop it before anything is counted.
    values = values.filter(pl.col(_VALUE).is_not_null())
    if unit is not None:
        values = apply_per_series_unit(values, series_meta, unit)
    if bucket_us is not None:
        values = _bucket(values, bucket_us)
    if values.is_empty():
        return pl.DataFrame(schema=_AGG_SCHEMA), empty_effective

    joined = values.join(members, on="series_id", how="inner")
    aggregates = _aggregate(joined, agg)
    effective = (
        joined.select(["entry_idx", "series_id"]).unique().group_by("entry_idx").agg(pl.len().cast(pl.Int64).alias("n"))
    )
    return aggregates, effective


def _by_entry(values: pl.DataFrame, entries: Sequence[str]) -> dict[str, pl.DataFrame]:
    """Split the aggregate frame into per-entry sub-frames, every entry present."""
    schema = {_TIME: _DATA_SCHEMA[_TIME], _VALUE: _DATA_SCHEMA[_VALUE], _EFFECTIVE: _DATA_SCHEMA[_EFFECTIVE]}
    out: dict[str, pl.DataFrame] = {}
    parts = values.partition_by("entry_idx", as_dict=True, include_key=True) if not values.is_empty() else {}
    for idx, entry in enumerate(entries):
        sub = parts.get((idx,))
        out[entry] = (
            pl.DataFrame(schema=schema)
            if sub is None
            else sub.drop("entry_idx").select([_TIME, _VALUE, _EFFECTIVE])
        )
    return out
