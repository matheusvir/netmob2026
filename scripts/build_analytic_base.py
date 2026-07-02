# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportReturnType=false, reportOperatorIssue=false
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.analysis_utils import (  # noqa: E402
    LOCAL_DATA_ROOT,
    NETMOB_DATA_ROOT,
    extract_trip_base,
    flag_partial_mobility_days,
    load_gtfs_routes,
    load_gtfs_trips_union,
    normalize_route_code,
)

DERIVED_DIR = LOCAL_DATA_ROOT / "derived"
FIGURES_DIR = DERIVED_DIR / "figures"
MARCH_START = pd.Timestamp("2026-03-01")
MARCH_END = pd.Timestamp("2026-04-01")
BRT_TZ = "America/Sao_Paulo"

WEEKDAY_COLUMNS = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    6: "sunday",
}

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

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

MANUAL_ROUTE_REVIEW = [
    {
        "route_norm": "17",
        "suggested_gtfs_route_norm": pd.NA,
        "route_crosswalk_confidence": "low",
        "route_crosswalk_status": "unresolved",
        "rationale": "No GTFS 17/17* route is present in the delivered feeds; keep this route out of integrated models.",
    },
    {
        "route_norm": "17J",
        "suggested_gtfs_route_norm": pd.NA,
        "route_crosswalk_confidence": "low",
        "route_crosswalk_status": "unresolved",
        "rationale": "No GTFS 17J route is present in the delivered feeds; keep this route out of integrated models.",
    },
    {
        "route_norm": "24A",
        "suggested_gtfs_route_norm": "24",
        "route_crosswalk_confidence": "medium",
        "route_crosswalk_status": "suggested",
        "rationale": "Ticket route 24A appears to be a variant of GTFS route 24 and is safe enough for integrated analysis with a medium-confidence flag.",
    },
    {
        "route_norm": "33F",
        "suggested_gtfs_route_norm": "33",
        "route_crosswalk_confidence": "medium",
        "route_crosswalk_status": "suggested",
        "rationale": "Ticket route 33F matches the same corridor as GTFS route 33 and is carried with a medium-confidence flag.",
    },
    {
        "route_norm": "34",
        "suggested_gtfs_route_norm": "34B",
        "route_crosswalk_confidence": "high",
        "route_crosswalk_status": "suggested",
        "rationale": "Ticket route 34 matches GTFS route 34B by route description and is treated as a high-confidence manual resolution.",
    },
    {
        "route_norm": "42SL",
        "suggested_gtfs_route_norm": "42S",
        "route_crosswalk_confidence": "high",
        "route_crosswalk_status": "suggested",
        "rationale": "Ticket route 42SL aligns with GTFS route 42S by corridor and suffix meaning (São Lourenço).",
    },
    {
        "route_norm": "46P",
        "suggested_gtfs_route_norm": pd.NA,
        "route_crosswalk_confidence": "low",
        "route_crosswalk_status": "unresolved",
        "rationale": "Ticket route 46P cannot be mapped safely to the GTFS feeds and is excluded from integrated models.",
    },
    {
        "route_norm": "56",
        "suggested_gtfs_route_norm": pd.NA,
        "route_crosswalk_confidence": "low",
        "route_crosswalk_status": "unresolved",
        "rationale": "Ticket route 56 has no GTFS counterpart in the delivered feeds and is excluded from integrated models.",
    },
    {
        "route_norm": "580M",
        "suggested_gtfs_route_norm": pd.NA,
        "route_crosswalk_confidence": "low",
        "route_crosswalk_status": "unresolved",
        "rationale": "Ticket route 580M has no GTFS counterpart in the delivered feeds and is excluded from integrated models.",
    },
    {
        "route_norm": "62B",
        "suggested_gtfs_route_norm": "62",
        "route_crosswalk_confidence": "high",
        "route_crosswalk_status": "suggested",
        "rationale": "Ticket route 62B matches GTFS route 62 by origin/destination and is treated as a high-confidence manual resolution.",
    },
]


def ensure_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def weather_category(series: pd.Series) -> pd.Categorical:
    return pd.cut(
        series,
        bins=[-np.inf, 1, 10, 25, np.inf],
        labels=["Clear", "Light Rain", "Moderate Rain", "Heavy Rain / Storm"],
        right=False,
    )


def collapse_confidence(values: pd.Series) -> str:
    candidates = [
        str(value).lower()
        for value in values.dropna()
        if str(value).lower() in CONFIDENCE_ORDER
    ]
    if not candidates:
        return "unknown"
    return min(candidates, key=lambda value: CONFIDENCE_ORDER[value])


def build_gtfs_lookup() -> pd.DataFrame:
    gtfs = load_gtfs_routes()[
        ["route_id", "route_short_name", "route_long_name", "source_feed"]
    ].drop_duplicates().copy()
    gtfs["gtfs_route_norm"] = gtfs["route_short_name"].map(normalize_route_code)
    return gtfs.drop_duplicates(subset=["gtfs_route_norm"], keep="last").reset_index(drop=True)


def build_weather_hourly() -> pd.DataFrame:
    weather_path_candidates = [
        LOCAL_DATA_ROOT / "meteorological_data.csv",
        LOCAL_DATA_ROOT / "preprocessed_meteorological_data.csv",
        NETMOB_DATA_ROOT / "auxiliar_data" / "meteorological_data.csv",
        NETMOB_DATA_ROOT / "auxiliar_data" / "preprocessed_meteorological_data.csv",
    ]
    weather_path = next(path for path in weather_path_candidates if path.exists())
    weather = pd.read_csv(weather_path)
    weather = weather.rename(
        columns={
            "Timestamp (UTC)": "timestamp_utc",
            "Rain (mm)": "rain_mm",
            "Temp. Ins. (C)": "temp_c",
            "Temp. Max. (C)": "temp_max_c",
            "Temp. Min. (C)": "temp_min_c",
            "Wind Speed (m/s)": "wind_speed",
            "Wind Gust (m/s)": "wind_gust",
            "Rad. (KJ/m2)": "radiation",
            "Radiation (KJ/m²)": "radiation",
        }
    )
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    weather["timestamp_brt"] = weather["timestamp_utc"].dt.tz_convert(BRT_TZ)
    weather = weather.sort_values("timestamp_brt").reset_index(drop=True)

    start_brt = MARCH_START.tz_localize(BRT_TZ)
    end_brt = MARCH_END.tz_localize(BRT_TZ)
    weather = weather[
        weather["timestamp_brt"].between(start_brt, end_brt, inclusive="left")
    ].copy()

    full_hours = pd.DataFrame(
        {
            "timestamp_brt": pd.date_range(
                start=start_brt,
                end=end_brt - pd.Timedelta(hours=1),
                freq="h",
            )
        }
    )
    weather = full_hours.merge(weather, on="timestamp_brt", how="left")
    weather["date"] = weather["timestamp_brt"].dt.tz_localize(None).dt.normalize()
    weather["hour"] = weather["timestamp_brt"].dt.hour
    weather["weather_observation_missing"] = weather["timestamp_utc"].isna()
    weather["rain_3h"] = weather["rain_mm"].rolling(window=3, min_periods=1).sum()
    weather["rain_6h"] = weather["rain_mm"].rolling(window=6, min_periods=1).sum()
    weather["weather_cat"] = weather_category(weather["rain_mm"])
    weather["is_rain"] = weather["rain_mm"].fillna(0) > 0
    weather["is_adverse"] = weather["rain_mm"].fillna(0) >= 10
    weather.to_parquet(DERIVED_DIR / "weather_hourly.parquet", index=False)
    return weather


def ticket_inventory_and_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    ticket_dir = NETMOB_DATA_ROOT / "ticket_data"
    daily_inventory: list[dict[str, object]] = []
    hourly_parts: list[pd.DataFrame] = []

    for path in sorted(ticket_dir.glob("*.csv")):
        df = pd.read_csv(
            path,
            usecols=[
                "transaction_date",
                "route_name",
                "card_type",
                "fare_type",
                "debited_amount",
                "integration_flag",
                "anon_user_id",
            ],
            low_memory=False,
        )
        dt = pd.to_datetime(df["transaction_date"], utc=True).dt.tz_convert(BRT_TZ)
        df["date"] = dt.dt.normalize().dt.tz_localize(None)
        df["hour"] = dt.dt.hour
        df["route_norm"] = df["route_name"].map(normalize_route_code)

        card_codes = pd.to_numeric(df["card_type"], errors="coerce").astype("Int64")
        fare_codes = pd.to_numeric(df["fare_type"], errors="coerce").astype("Int64")
        df["card_label"] = card_codes.map(CARD_TYPE_MAP).fillna("Unknown")
        df["fare_label"] = fare_codes.map(FARE_TYPE_MAP).fillna("Unknown")

        df["boardings"] = 1
        df["paid_boardings"] = (
            pd.to_numeric(df["debited_amount"], errors="coerce").fillna(0) > 0
        ).astype(int)
        df["free_rides"] = (
            pd.to_numeric(df["debited_amount"], errors="coerce").fillna(0) <= 0
        ).astype(int)
        df["transfers"] = (
            pd.to_numeric(df["integration_flag"], errors="coerce").fillna(0).astype(int)
        )
        df["revenue"] = pd.to_numeric(df["debited_amount"], errors="coerce").fillna(0)

        daily_inventory.append(
            {
                "date": df["date"].iloc[0],
                "row_count": int(len(df)),
                "source_file": path.name,
            }
        )

        hourly = (
            df.groupby(
                ["date", "hour", "route_norm", "card_label", "fare_label"],
                observed=True,
                as_index=False,
            )
            .agg(
                boardings=("boardings", "sum"),
                paid_boardings=("paid_boardings", "sum"),
                free_rides=("free_rides", "sum"),
                transfers=("transfers", "sum"),
                revenue=("revenue", "sum"),
                non_cash_users=("anon_user_id", "nunique"),
            )
        )
        hourly_parts.append(hourly)

    ticket_daily_inventory = (
        pd.DataFrame(daily_inventory).sort_values("date").reset_index(drop=True)
    )
    ticket_hourly = pd.concat(hourly_parts, ignore_index=True)
    ticket_hourly["is_weekend"] = ticket_hourly["date"].dt.dayofweek >= 5
    ticket_hourly["day_of_week"] = ticket_hourly["date"].dt.dayofweek
    ticket_hourly.to_parquet(DERIVED_DIR / "ticket_hourly_route_profile.parquet", index=False)
    return ticket_daily_inventory, ticket_hourly


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def scheduled_trip_counts(gtfs_lookup: pd.DataFrame) -> pd.DataFrame:
    route_lookup = gtfs_lookup[["route_id", "gtfs_route_norm", "route_short_name"]].copy()

    trips = load_gtfs_trips_union().copy()
    trips["trip_base"] = trips["trip_id"].map(extract_trip_base)
    trips = trips.merge(route_lookup, on="route_id", how="left")

    stop_time_frames = []
    for path in sorted((NETMOB_DATA_ROOT / "GTFS_data").glob("*/stop_times.txt")):
        df = pd.read_csv(path, usecols=["trip_id", "departure_time", "stop_sequence"])
        df = (
            df.sort_values(["trip_id", "stop_sequence"])
            .groupby("trip_id", as_index=False)
            .first()
        )
        df["source_feed"] = path.parent.name
        stop_time_frames.append(df)
    stop_times = pd.concat(stop_time_frames, ignore_index=True)
    stop_times["scheduled_hour"] = (
        stop_times["departure_time"].astype(str).str.split(":").str[0].astype(int) % 24
    )

    calendar_frames = []
    for path in sorted((NETMOB_DATA_ROOT / "GTFS_data").glob("*/calendar.txt")):
        df = pd.read_csv(path)
        df["source_feed"] = path.parent.name
        calendar_frames.append(df)
    calendar = pd.concat(calendar_frames, ignore_index=True)
    calendar["start_date"] = pd.to_datetime(
        calendar["start_date"].astype(str), format="%Y%m%d"
    )
    calendar["end_date"] = pd.to_datetime(
        calendar["end_date"].astype(str), format="%Y%m%d"
    )

    service_rows: list[dict[str, object]] = []
    for row in calendar.itertuples(index=False):
        start = max(row.start_date, MARCH_START)
        end = min(row.end_date, MARCH_END - pd.Timedelta(days=1))
        if start > end:
            continue
        for date in pd.date_range(start, end, freq="D"):
            weekday_col = WEEKDAY_COLUMNS[date.dayofweek]
            if int(getattr(row, weekday_col)) == 1:
                service_rows.append(
                    {
                        "service_id": row.service_id,
                        "source_feed": row.source_feed,
                        "date": date,
                    }
                )
    service_dates = pd.DataFrame(service_rows)

    schedule = trips.merge(
        stop_times[["trip_id", "scheduled_hour", "source_feed"]],
        on=["trip_id", "source_feed"],
        how="left",
    ).merge(service_dates, on=["service_id", "source_feed"], how="inner")

    schedule["direction"] = (
        pd.to_numeric(schedule["direction_id"], errors="coerce").fillna(0).astype(int)
    )
    schedule = schedule.dropna(subset=["gtfs_route_norm", "scheduled_hour"])

    grouped = (
        schedule.groupby(
            ["date", "scheduled_hour", "gtfs_route_norm", "direction"],
            observed=True,
            as_index=False,
        )
        .agg(scheduled_trip_count=("trip_base", "nunique"), operator=("source_feed", "first"))
        .rename(columns={"scheduled_hour": "hour", "gtfs_route_norm": "route_norm"})
    )
    return grouped


def mobility_inventory_and_hourly(
    mobility_quality: pd.DataFrame,
    scheduled_counts_df: pd.DataFrame,
) -> pd.DataFrame:
    mobility_dir = NETMOB_DATA_ROOT / "mobility_data"
    parts: list[pd.DataFrame] = []
    quality_lookup = mobility_quality[["date", "is_partial_day"]].copy()
    quality_lookup["date"] = pd.to_datetime(quality_lookup["date"])

    for path in sorted(mobility_dir.glob("*.csv")):
        df = pd.read_csv(
            path,
            usecols=["id", "timestamp", "tripId", "lat", "lng", "lineId", "direction"],
            low_memory=False,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)
        df["date"] = df["timestamp"].dt.normalize()
        df["hour"] = df["timestamp"].dt.hour
        df["route_norm"] = df["lineId"].map(normalize_route_code)
        df["trip_base"] = df["tripId"].map(extract_trip_base)
        df["direction"] = (
            pd.to_numeric(df["direction"], errors="coerce").fillna(0).astype(int)
        )

        grouped = (
            df.groupby(["date", "hour", "route_norm", "direction"], observed=True, as_index=False)
            .agg(
                active_vehicles=("id", "nunique"),
                gps_obs_count=("id", "size"),
                observed_trip_count=("trip_base", "nunique"),
            )
        )

        departures = (
            df.groupby(
                ["date", "hour", "route_norm", "direction", "trip_base"],
                observed=True,
                as_index=False,
            )
            .agg(start_ts=("timestamp", "min"))
            .sort_values(["date", "hour", "route_norm", "direction", "start_ts"])
        )
        departures["headway_min"] = (
            departures.groupby(["date", "hour", "route_norm", "direction"], observed=True)["start_ts"]
            .diff()
            .dt.total_seconds()
            .div(60)
        )
        departures = departures[
            departures["headway_min"].notna() & (departures["headway_min"] > 0)
        ]
        if len(departures):
            headway = (
                departures.groupby(["date", "hour", "route_norm", "direction"], observed=True)
                .agg(
                    headway_p50=("headway_min", "median"),
                    headway_p90=("headway_min", lambda s: float(np.nanpercentile(s, 90))),
                )
                .reset_index()
            )
            grouped = grouped.merge(
                headway,
                on=["date", "hour", "route_norm", "direction"],
                how="left",
            )

        prev_lat = df["lat"].shift()
        prev_lng = df["lng"].shift()
        prev_id = df["id"].shift()
        prev_ts = df["timestamp"].shift()
        prev_route = df["route_norm"].shift()
        prev_direction = df["direction"].shift()
        prev_date = df["date"].shift()
        same_chain = (
            (df["id"] == prev_id)
            & (df["route_norm"] == prev_route)
            & (df["direction"] == prev_direction)
            & (df["date"] == prev_date)
        )
        delta_seconds = (df["timestamp"] - prev_ts).dt.total_seconds()
        distance_km = haversine_km(prev_lat, prev_lng, df["lat"], df["lng"])
        speed_kmh = np.where(
            same_chain & (delta_seconds > 0) & (delta_seconds <= 180),
            distance_km / (delta_seconds / 3600.0),
            np.nan,
        )
        df["speed_kmh"] = speed_kmh
        df.loc[(df["speed_kmh"] <= 0) | (df["speed_kmh"] > 100), "speed_kmh"] = np.nan
        speed = (
            df.dropna(subset=["speed_kmh"])
            .groupby(["date", "hour", "route_norm", "direction"], observed=True)
            .agg(
                speed_p50=("speed_kmh", "median"),
                speed_p10=("speed_kmh", lambda s: float(np.nanpercentile(s, 10))),
            )
            .reset_index()
        )
        grouped = grouped.merge(speed, on=["date", "hour", "route_norm", "direction"], how="left")
        grouped = grouped.merge(
            quality_lookup.rename(columns={"is_partial_day": "coverage_flag"}),
            on="date",
            how="left",
        )
        grouped["coverage_flag"] = (
            grouped["coverage_flag"].map({True: "partial_day", False: "ok"}).fillna("ok")
        )
        grouped = grouped.merge(
            scheduled_counts_df,
            on=["date", "hour", "route_norm", "direction"],
            how="left",
        )
        grouped["scheduled_trip_count"] = grouped["scheduled_trip_count"].fillna(0)
        grouped["observed_vs_scheduled_trip_ratio"] = np.where(
            grouped["scheduled_trip_count"] > 0,
            grouped["observed_trip_count"] / grouped["scheduled_trip_count"],
            np.nan,
        )
        grouped["scheduled_headway_proxy"] = np.where(
            grouped["scheduled_trip_count"] > 0,
            60.0 / grouped["scheduled_trip_count"],
            np.nan,
        )
        grouped["service_gap_index"] = np.where(
            grouped["scheduled_trip_count"] > 0,
            np.clip(1 - grouped["observed_vs_scheduled_trip_ratio"], 0, None),
            np.nan,
        )
        parts.append(grouped)

    mobility_hourly = pd.concat(parts, ignore_index=True)
    mobility_hourly.to_parquet(DERIVED_DIR / "mobility_hourly_route_direction.parquet", index=False)
    return mobility_hourly


def mobility_quality_flags() -> pd.DataFrame:
    rows = []
    for path in sorted((NETMOB_DATA_ROOT / "mobility_data").glob("*.csv")):
        rows.append(
            {
                "date": pd.Timestamp(path.stem),
                "row_count": sum(1 for _ in open(path, "rb")) - 1,
                "source_file": path.name,
            }
        )
    inventory = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    flagged = flag_partial_mobility_days(inventory)
    flagged.to_csv(DERIVED_DIR / "mobility_day_quality_flags.csv", index=False)
    return flagged


def route_crosswalk(gtfs_lookup: pd.DataFrame) -> pd.DataFrame:
    ticket_routes = []
    for path in sorted((NETMOB_DATA_ROOT / "ticket_data").glob("*.csv")):
        df = pd.read_csv(path, usecols=["route_name"], low_memory=False)
        ticket_routes.append(
            pd.DataFrame({"route_name_raw": df["route_name"].astype("string")})
        )
    ticket = pd.concat(ticket_routes, ignore_index=True).drop_duplicates().reset_index(drop=True)
    ticket["route_norm"] = ticket["route_name_raw"].map(normalize_route_code)

    review = pd.DataFrame(MANUAL_ROUTE_REVIEW)
    review.to_csv(DERIVED_DIR / "route_crosswalk_review.csv", index=False)

    crosswalk = ticket.merge(
        gtfs_lookup,
        left_on="route_norm",
        right_on="gtfs_route_norm",
        how="left",
    ).merge(review, on="route_norm", how="left")

    direct_match = crosswalk["gtfs_route_norm"].notna()
    crosswalk["resolved_route_norm"] = np.where(
        direct_match,
        crosswalk["route_norm"],
        crosswalk["suggested_gtfs_route_norm"],
    )
    crosswalk["route_crosswalk_confidence"] = np.where(
        direct_match,
        "high",
        crosswalk["route_crosswalk_confidence"].fillna("low"),
    )
    crosswalk["route_crosswalk_status"] = np.where(
        direct_match,
        "direct_match",
        crosswalk["route_crosswalk_status"].fillna("unresolved"),
    )
    crosswalk["rationale"] = np.where(
        direct_match,
        "Exact normalized route match between ticketing and GTFS.",
        crosswalk["rationale"].fillna("No defensible GTFS mapping available."),
    )

    resolved_lookup = gtfs_lookup.add_prefix("resolved_")
    crosswalk = crosswalk.merge(
        resolved_lookup,
        left_on="resolved_route_norm",
        right_on="resolved_gtfs_route_norm",
        how="left",
    )

    crosswalk["route_short_name"] = crosswalk["route_short_name"].where(
        direct_match,
        crosswalk["resolved_route_short_name"],
    )
    crosswalk["route_long_name"] = crosswalk["route_long_name"].where(
        direct_match,
        crosswalk["resolved_route_long_name"],
    )
    crosswalk["source_feed"] = crosswalk["source_feed"].where(
        direct_match,
        crosswalk["resolved_source_feed"],
    )
    crosswalk["manual_review"] = crosswalk["route_crosswalk_status"].eq("unresolved")
    crosswalk["is_resolved"] = crosswalk["resolved_route_norm"].notna()

    columns = [
        "route_name_raw",
        "route_norm",
        "resolved_route_norm",
        "route_short_name",
        "route_long_name",
        "source_feed",
        "route_crosswalk_confidence",
        "route_crosswalk_status",
        "manual_review",
        "is_resolved",
        "rationale",
    ]
    crosswalk = crosswalk[columns].sort_values(["route_norm", "route_name_raw"]).reset_index(drop=True)
    crosswalk.to_csv(DERIVED_DIR / "route_crosswalk.csv", index=False)
    return crosswalk


def trip_coverage_summary() -> pd.DataFrame:
    gtfs_trips = load_gtfs_trips_union()[["trip_base"]].drop_duplicates().copy()
    gtfs_trips["in_gtfs"] = True

    mobility_base_parts = []
    for path in sorted((NETMOB_DATA_ROOT / "mobility_data").glob("*.csv")):
        df = pd.read_csv(path, usecols=["tripId"])
        mobility_base_parts.append(df["tripId"].map(extract_trip_base))

    mobility_bases = pd.Series(
        pd.concat(mobility_base_parts, ignore_index=True).dropna().unique(),
        name="trip_base",
    )
    coverage = pd.DataFrame({"trip_base": mobility_bases}).merge(
        gtfs_trips,
        on="trip_base",
        how="left",
    )
    coverage["in_gtfs"] = coverage["in_gtfs"].fillna(False)
    coverage = coverage.sort_values("trip_base").reset_index(drop=True)
    coverage.to_csv(DERIVED_DIR / "trip_base_coverage.csv", index=False)
    return coverage


def build_integrated_route_hour(
    weather_hourly: pd.DataFrame,
    ticket_hourly: pd.DataFrame,
    mobility_hourly: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    weather_columns = [
        "date",
        "hour",
        "rain_mm",
        "rain_3h",
        "rain_6h",
        "weather_cat",
        "is_rain",
        "is_adverse",
        "temp_c",
        "temp_max_c",
        "temp_min_c",
        "wind_speed",
        "wind_gust",
        "weather_observation_missing",
    ]
    if "radiation" in weather_hourly.columns:
        weather_columns.append("radiation")

    overlap_dates = pd.Index(pd.to_datetime(mobility_hourly["date"].unique())).sort_values()
    weather_route_hour = weather_hourly[weather_columns].copy()
    weather_route_hour = weather_route_hour[
        weather_route_hour["date"].isin(overlap_dates)
        & ~weather_route_hour["weather_observation_missing"]
    ].copy()

    ticket_enriched = ticket_hourly.merge(
        crosswalk[
            [
                "route_norm",
                "resolved_route_norm",
                "route_crosswalk_confidence",
                "route_crosswalk_status",
            ]
        ],
        on="route_norm",
        how="left",
    )
    ticket_enriched = ticket_enriched[
        ticket_enriched["date"].isin(overlap_dates)
        & ticket_enriched["resolved_route_norm"].notna()
    ].copy()
    ticket_enriched["ticket_route_norm"] = ticket_enriched["route_norm"]
    ticket_enriched["route_norm"] = ticket_enriched["resolved_route_norm"]

    ticket_route_hour = (
        ticket_enriched.groupby(["date", "hour", "route_norm"], observed=True, as_index=False)
        .agg(
            boardings=("boardings", "sum"),
            paid_boardings=("paid_boardings", "sum"),
            free_rides=("free_rides", "sum"),
            transfers=("transfers", "sum"),
            revenue=("revenue", "sum"),
            non_cash_users=("non_cash_users", "sum"),
            is_weekend=("is_weekend", "max"),
            day_of_week=("day_of_week", "first"),
            route_crosswalk_confidence=("route_crosswalk_confidence", collapse_confidence),
        )
    )

    mobility_route_hour = (
        mobility_hourly.groupby(["date", "hour", "route_norm"], observed=True, as_index=False)
        .agg(
            active_vehicles=("active_vehicles", "sum"),
            gps_obs_count=("gps_obs_count", "sum"),
            observed_trip_count=("observed_trip_count", "sum"),
            scheduled_trip_count=("scheduled_trip_count", "sum"),
            headway_p50=("headway_p50", "median"),
            headway_p90=("headway_p90", "median"),
            speed_p50=("speed_p50", "median"),
            speed_p10=("speed_p10", "median"),
            observed_vs_scheduled_trip_ratio=("observed_vs_scheduled_trip_ratio", "mean"),
            scheduled_headway_proxy=("scheduled_headway_proxy", "mean"),
            service_gap_index=("service_gap_index", "mean"),
            coverage_flag=(
                "coverage_flag",
                lambda s: "partial_day" if (s == "partial_day").any() else "ok",
            ),
            operator=("operator", "first"),
        )
    )

    integrated = ticket_route_hour.merge(
        weather_route_hour,
        on=["date", "hour"],
        how="inner",
    ).merge(
        mobility_route_hour,
        on=["date", "hour", "route_norm"],
        how="left",
    )
    integrated = integrated.sort_values(["date", "hour", "route_norm"]).reset_index(drop=True)
    integrated.to_parquet(DERIVED_DIR / "integrated_route_hour.parquet", index=False)
    return integrated


def main() -> None:
    ensure_dirs()
    gtfs_lookup = build_gtfs_lookup()
    weather_hourly = build_weather_hourly()
    _, ticket_hourly = ticket_inventory_and_hourly()
    quality = mobility_quality_flags()
    crosswalk = route_crosswalk(gtfs_lookup)
    trip_coverage_summary()
    schedules = scheduled_trip_counts(gtfs_lookup)
    mobility_hourly = mobility_inventory_and_hourly(quality, schedules)
    build_integrated_route_hour(weather_hourly, ticket_hourly, mobility_hourly, crosswalk)
    print("Built derived outputs in", DERIVED_DIR)


if __name__ == "__main__":
    main()