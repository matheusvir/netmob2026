"""Shared helpers for the NetMob 2026 analysis notebooks.

The module stays intentionally small: path constants, route/trip normalization,
lightweight loaders, and a couple of validation helpers that can be reused across
notebooks without introducing a package structure.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETMOB_DATA_ROOT = PROJECT_ROOT / "data"
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"

_DATA_ROOTS: tuple[Path, ...] = (NETMOB_DATA_ROOT, LOCAL_DATA_ROOT)

CARD_TYPE_MAP = {
    -1: "Cash",
    1: "Standard",
    2: "Student",
    3: "Labor",
    4: "Senior",
}

FARE_TYPE_MAP = {
    1: "Free",
    2: "Electronic",
    3: "Cash",
}


def _existing_roots() -> list[Path]:
    return [root for root in _DATA_ROOTS if root.exists()]


def _resolve_first_existing(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find any of the expected data files:\n{searched}")


def _collect_files(patterns: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    for root in _existing_roots():
        for pattern in patterns:
            matches.extend(root.glob(pattern))
    unique_matches = sorted({path.resolve() for path in matches})
    return [Path(path) for path in unique_matches]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _add_derived_columns(frame: pd.DataFrame, *, route_column: str | None = None, trip_column: str | None = None) -> pd.DataFrame:
    result = frame.copy()

    if route_column and route_column in result.columns:
        result["route_norm"] = result[route_column].map(normalize_route_code)

    if trip_column and trip_column in result.columns:
        result["trip_base"] = result[trip_column].map(extract_trip_base)

    return result


def normalize_route_code(value: object) -> str:
    """Normalize route identifiers for cross-source matching.

    Examples
    --------
    >>> normalize_route_code("024.A")
    '24A'
    >>> normalize_route_code("49.1")
    '49.1'
    >>> normalize_route_code("17 J")
    '17J'
    """

    if pd.isna(value):
        return ""

    route = str(value).strip().upper()
    route = route.replace("–", "-").replace("—", "-")
    route = re.sub(r"\s+", "", route)

    decimal_match = re.fullmatch(r"0*(\d+)\.(\d+)", route)
    if decimal_match:
        left, right = decimal_match.groups()
        return f"{int(left)}.{int(right)}"

    numbered_suffix_match = re.fullmatch(r"0*(\d+)[\._-]*([A-Z]+)", route)
    if numbered_suffix_match:
        number, suffix = numbered_suffix_match.groups()
        return f"{int(number)}{suffix}"

    prefixed_number_match = re.fullmatch(r"([A-Z]+)[\._-]*0*(\d+)", route)
    if prefixed_number_match:
        prefix, number = prefixed_number_match.groups()
        return f"{prefix}{int(number)}"

    pure_number_match = re.fullmatch(r"0*(\d+)", route)
    if pure_number_match:
        return str(int(pure_number_match.group(1)))

    return re.sub(r"[^A-Z0-9.]+", "", route)


def extract_trip_base(trip_id: object) -> str:
    """Normalize trip identifiers across GTFS and mobility feeds.

    GTFS trip ids end with an extra direction suffix (for example
    ``1219424_D_1_1``), while mobility ``tripId`` values already stop at the
    trip base (for example ``1219424_D_1``). Strip the trailing ``_0`` / ``_1``
    only for the GTFS shape with *four or more* underscore-separated parts so we
    do not accidentally truncate mobility trip ids.
    """

    if pd.isna(trip_id):
        return ""

    trip_text = str(trip_id).strip()
    parts = trip_text.rsplit("_", 1)
    if trip_text.count("_") >= 3 and len(parts) == 2 and parts[1] in {"0", "1"}:
        return parts[0]
    return trip_text


def assert_expected_columns(df: pd.DataFrame, expected: Iterable[str], dataset_name: str) -> None:
    """Raise a helpful error if a DataFrame is missing required columns."""

    expected_set = {column for column in expected}
    missing = sorted(expected_set - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def flag_partial_mobility_days(row_counts_df: pd.DataFrame) -> pd.DataFrame:
    """Flag days whose mobility row counts are substantially below the typical level.

    The helper accepts a small inventory table with a date column and a row-count
    column. It adds a few transparent diagnostics that notebooks can inspect
    before inferring headways or frequencies.
    """

    result = row_counts_df.copy()

    date_column = next((column for column in ("date", "day", "mobility_date") if column in result.columns), None)
    if date_column is None:
        raise KeyError("Expected a date column named 'date', 'day', or 'mobility_date'.")

    count_column = next((column for column in ("row_count", "n_rows", "rows", "count") if column in result.columns), None)
    if count_column is None:
        raise KeyError("Expected a row-count column named 'row_count', 'n_rows', 'rows', or 'count'.")

    counts = pd.to_numeric(result[count_column], errors="coerce")
    median_count = counts.median()
    q1 = counts.quantile(0.25)
    q3 = counts.quantile(0.75)
    iqr = q3 - q1
    lower_fence = max(0.0, q1 - 1.5 * iqr)
    relative_threshold = median_count * 0.6 if pd.notna(median_count) else counts.min()
    threshold = max(relative_threshold, lower_fence) if pd.notna(lower_fence) else relative_threshold

    result["row_count"] = counts
    result["row_count_ratio"] = counts / median_count if pd.notna(median_count) and median_count != 0 else pd.NA
    result["partial_day_threshold"] = threshold
    result["is_partial_day"] = counts < threshold
    result["partial_day_reason"] = result["is_partial_day"].map(
        {True: "row_count_below_threshold", False: "ok"}
    )
    return result


def load_weather(path: str | Path | None = None) -> pd.DataFrame:
    """Load the weather source used by the analysis notebooks."""

    if path is not None:
        weather_path = Path(path)
    else:
        weather_path = _resolve_first_existing(
            [
                LOCAL_DATA_ROOT / "meteorological_data.csv",
                LOCAL_DATA_ROOT / "preprocessed_meteorological_data.csv",
                NETMOB_DATA_ROOT / "auxiliar_data" / "meteorological_data.csv",
                NETMOB_DATA_ROOT / "auxiliar_data" / "preprocessed_meteorological_data.csv",
            ]
        )

    return _read_table(weather_path)


def load_ticket_files(path: str | Path | None = None) -> pd.DataFrame:
    """Load and concatenate all ticket CSV files available in the data roots."""

    if path is not None:
        ticket_path = Path(path)
        if ticket_path.is_dir():
            files = sorted(ticket_path.glob("*.csv"))
        else:
            files = [ticket_path]
    else:
        files = _collect_files(["ticket_data/*.csv"])

    if not files:
        raise FileNotFoundError("No ticket data files were found.")

    frames = []
    for file_path in files:
        frame = _read_table(file_path)
        frame.insert(0, "source_file", file_path.name)
        frame = _add_derived_columns(
            frame,
            route_column="route_name" if "route_name" in frame.columns else "view_type" if "view_type" in frame.columns else None,
        )
        if "card_type" in frame.columns:
            card_codes = pd.Series(
                pd.to_numeric(frame["card_type"], errors="coerce"),
                index=frame.index,
            )
            frame["card_label"] = card_codes.map(CARD_TYPE_MAP).fillna("Unknown")
        if "fare_type" in frame.columns:
            fare_codes = pd.Series(
                pd.to_numeric(frame["fare_type"], errors="coerce"),
                index=frame.index,
            )
            frame["fare_label"] = fare_codes.map(FARE_TYPE_MAP).fillna("Unknown")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_mobility_files(path: str | Path | None = None) -> pd.DataFrame:
    """Load and concatenate mobility telemetry files."""

    if path is not None:
        mobility_path = Path(path)
        if mobility_path.is_dir():
            files = sorted(mobility_path.glob("*.csv"))
        else:
            files = [mobility_path]
    else:
        files = _collect_files(["mobility_data/*.csv", "processed_data/*.csv"])

    if not files:
        raise FileNotFoundError("No mobility data files were found.")

    frames = []
    for file_path in files:
        frame = _read_table(file_path)
        frame.insert(0, "source_file", file_path.name)
        frames.append(
            _add_derived_columns(
                frame,
                route_column="lineId" if "lineId" in frame.columns else "route_name" if "route_name" in frame.columns else None,
                trip_column="tripId" if "tripId" in frame.columns else None,
            )
        )

    return pd.concat(frames, ignore_index=True)


def load_gtfs_routes(path: str | Path | None = None) -> pd.DataFrame:
    """Load, concatenate, and de-duplicate GTFS route tables."""

    if path is not None:
        routes_path = Path(path)
        if routes_path.is_dir():
            files = sorted(routes_path.glob("routes.txt"))
        else:
            files = [routes_path]
    else:
        files = _collect_files(["GTFS_data/*/routes.txt"])

    if not files:
        raise FileNotFoundError("No GTFS route files were found.")

    frames = []
    for file_path in files:
        frame = _read_table(file_path)
        frame.insert(0, "source_file", file_path.name)
        frame.insert(0, "source_feed", file_path.parent.name)
        frames.append(frame)

    routes = pd.concat(frames, ignore_index=True)
    if "route_id" in routes.columns:
        routes = routes.drop_duplicates(subset="route_id", keep="last").reset_index(drop=True)
    return routes


def load_gtfs_trips_union(path: str | Path | None = None) -> pd.DataFrame:
    """Load the union of all GTFS trips across the available snapshots."""

    if path is not None:
        trips_path = Path(path)
        if trips_path.is_dir():
            files = sorted(trips_path.glob("trips.txt"))
        else:
            files = [trips_path]
    else:
        files = _collect_files(["GTFS_data/*/trips.txt"])

    if not files:
        raise FileNotFoundError("No GTFS trip files were found.")

    frames = []
    for file_path in files:
        frame = _read_table(file_path)
        frame.insert(0, "source_file", file_path.name)
        frame.insert(0, "source_feed", file_path.parent.name)
        if "trip_id" in frame.columns:
            frame["trip_base"] = frame["trip_id"].map(extract_trip_base)
        frames.append(frame)

    trips = pd.concat(frames, ignore_index=True)
    if "trip_id" in trips.columns:
        trips = trips.drop_duplicates(subset="trip_id", keep="last").reset_index(drop=True)
    return trips


__all__ = [
    "PROJECT_ROOT",
    "NETMOB_DATA_ROOT",
    "LOCAL_DATA_ROOT",
    "assert_expected_columns",
    "extract_trip_base",
    "flag_partial_mobility_days",
    "load_gtfs_routes",
    "load_gtfs_trips_union",
    "load_mobility_files",
    "load_ticket_files",
    "load_weather",
    "normalize_route_code",
]
