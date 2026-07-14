"""High-throughput parser for NOAA GHCN-Daily metadata and observations."""

from __future__ import annotations

import calendar
import io
import json
import logging
import tarfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from climate_forecast.config import (
    COUNTRIES_FILE,
    INVENTORY_FILE,
    RAW_TAR_ARCHIVE,
    STATIONS_FILE,
    TARGET_ELEMENTS,
    VALUE_SCALE_FACTOR,
)

logger = logging.getLogger(__name__)


@dataclass
class StationMetadata:
    station_id: str
    latitude: float
    longitude: float
    elevation: float
    state: str
    name: str
    country_code: str
    country_name: str
    is_gsn: bool
    wmo_id: str
    climate_zone: str = "Temperate"


def load_countries_metadata(countries_path: Path | str = COUNTRIES_FILE) -> Dict[str, str]:
    """Parse FIPS country codes and names from ghcnd-countries.txt."""
    path = Path(countries_path)
    if not path.exists():
        logger.warning(f"Countries file not found at {path}")
        return {}

    countries = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if len(line) >= 3:
                code = line[:2].strip()
                name = line[3:].strip()
                countries[code] = name
    return countries


def estimate_climate_zone(lat: float, elevation: float) -> str:
    """Heuristic climate classification based on latitude and elevation."""
    abs_lat = abs(lat)
    if abs_lat < 23.5:
        return "Highland Tropical" if elevation > 1500 else "Tropical"
    elif abs_lat < 35.0:
        return "Subtropical Arid" if elevation < 1000 else "Subtropical Highland"
    elif abs_lat < 55.0:
        return "Temperate Continental" if abs_lat > 45.0 else "Temperate Maritime"
    elif abs_lat < 66.5:
        return "Boreal/Subarctic"
    else:
        return "Polar/Tundra"


def load_stations_metadata(
    stations_path: Path | str = STATIONS_FILE,
    countries_path: Path | str = COUNTRIES_FILE,
) -> Dict[str, StationMetadata]:
    """Parse all station metadata from ghcnd-stations.txt."""
    path = Path(stations_path)
    countries = load_countries_metadata(countries_path)
    if not path.exists():
        logger.warning(f"Stations file not found at {path}")
        return {}

    stations = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 40:
                continue
            station_id = line[0:11].strip()
            try:
                lat = float(line[12:20].strip())
                lon = float(line[21:30].strip())
                elev_str = line[31:37].strip()
                elevation = float(elev_str) if elev_str and elev_str != "-999.9" else 0.0
            except ValueError:
                continue

            state = line[38:40].strip() if len(line) > 40 else ""
            name = line[41:71].strip() if len(line) > 71 else ""
            gsn_flag = line[72:75].strip() if len(line) > 75 else ""
            wmo_id = line[80:85].strip() if len(line) > 85 else ""

            country_code = station_id[:2]
            country_name = countries.get(country_code, country_code)
            climate_zone = estimate_climate_zone(lat, elevation)

            stations[station_id] = StationMetadata(
                station_id=station_id,
                latitude=lat,
                longitude=lon,
                elevation=elevation,
                state=state,
                name=name,
                country_code=country_code,
                country_name=country_name,
                is_gsn=bool(gsn_flag),
                wmo_id=wmo_id,
                climate_zone=climate_zone,
            )
    return stations


def select_global_representative_stations(
    inventory_path: Path | str = INVENTORY_FILE,
    stations_metadata: Optional[Dict[str, StationMetadata]] = None,
    min_start_year: int = 1995,
    min_end_year: int = 2024,
    target_count: int = 500,
) -> List[str]:
    """Select a diverse, globally distributed grid of high-quality stations with complete records."""
    if stations_metadata is None:
        stations_metadata = load_stations_metadata()

    # Read inventory to identify stations having TMAX, TMIN, and PRCP
    station_elements: Dict[str, Dict[str, Tuple[int, int]]] = {}
    with open(inventory_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 45:
                continue
            station_id = line[0:11].strip()
            elem = line[31:35].strip()
            try:
                start_yr = int(line[36:40])
                end_yr = int(line[41:45])
            except ValueError:
                continue

            if elem in TARGET_ELEMENTS:
                if station_id not in station_elements:
                    station_elements[station_id] = {}
                station_elements[station_id][elem] = (start_yr, end_yr)

    # Filter candidates with all 3 elements covering the target range
    candidate_ids = []
    for st_id, elems in station_elements.items():
        if len(elems) == 3:  # has TMAX, TMIN, PRCP
            has_coverage = all(
                start <= min_start_year and end >= min_end_year
                for start, end in elems.values()
            )
            if has_coverage and st_id in stations_metadata:
                candidate_ids.append(st_id)

    logger.info(f"Identified {len(candidate_ids)} eligible candidate stations with full records.")

    # Spatial binning (latitude / longitude grid cells) to ensure global coverage across all continents
    bins: Dict[Tuple[int, int], List[str]] = {}
    for st_id in candidate_ids:
        meta = stations_metadata[st_id]
        # 10x10 degree spatial binning
        lat_bin = int(np.floor(meta.latitude / 10.0))
        lon_bin = int(np.floor(meta.longitude / 10.0))
        key = (lat_bin, lon_bin)
        if key not in bins:
            bins[key] = []
        bins[key].append(st_id)

    # Pick top GSN or longest-record stations from each spatial bin
    selected = []
    # First pass: pick 1-2 per bin
    for key, bin_stations in bins.items():
        # Prefer GSN stations
        gsn_st = [s for s in bin_stations if stations_metadata[s].is_gsn]
        if gsn_st:
            selected.append(gsn_st[0])
        else:
            selected.append(bin_stations[0])

    # Second pass: fill up to target_count evenly from multi-station bins
    idx = 1
    while len(selected) < target_count:
        added_in_round = 0
        for key, bin_stations in bins.items():
            if len(bin_stations) > idx:
                cand = bin_stations[idx]
                if cand not in selected:
                    selected.append(cand)
                    added_in_round += 1
                    if len(selected) >= target_count:
                        break
        if added_in_round == 0:
            break
        idx += 1

    logger.info(f"Selected {len(selected)} globally distributed stations across {len(bins)} spatial bins.")
    return selected


def parse_dly_content(
    lines: Iterable[str],
    allowed_elements: Set[str] = set(TARGET_ELEMENTS),
    start_year: int = 1990,
    end_year: int = 2026,
) -> Generator[Dict, None, None]:
    """Parse raw ASCII lines from a NOAA GHCN .dly file into structured daily records."""
    for line in lines:
        if len(line) < 21:
            continue
        station_id = line[:11]
        try:
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21].strip()
        except ValueError:
            continue

        if element not in allowed_elements or year < start_year or year > end_year:
            continue

        num_days = calendar.monthrange(year, month)[1]
        for day in range(1, num_days + 1):
            start_pos = 21 + (day - 1) * 8
            end_pos = start_pos + 8
            if len(line) < end_pos:
                continue

            field = line[start_pos:end_pos]
            val_str = field[:5].strip()
            if not val_str or val_str == "-9999":
                continue

            try:
                raw_val = int(val_str)
            except ValueError:
                continue

            qflag = field[6].strip() if len(field) > 6 else ""
            # Discard failed quality check observations
            if qflag != "":
                continue

            # Scale to standard units (°C for temp, mm for prcp)
            value = float(raw_val) * VALUE_SCALE_FACTOR

            yield {
                "station_id": station_id,
                "date": date(year, month, day),
                "year": year,
                "month": month,
                "day": day,
                "day_of_year": date(year, month, day).timetuple().tm_yday,
                "element": element,
                "value": value,
            }


def extract_global_stations_dataset(
    tar_archive_path: Path | str = RAW_TAR_ARCHIVE,
    selected_stations: Optional[Set[str]] = None,
    output_parquet_path: Optional[Path | str] = None,
    start_year: int = 1990,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Extract and pivot daily observations for selected stations from raw tar archive."""
    tar_path = Path(tar_archive_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Raw archive not found at {tar_path}")

    selected_set = set(selected_stations) if selected_stations else None
    target_names = {f"ghcnd_all/{st}.dly" for st in selected_set} if selected_set else None

    logger.info(f"Opening archive {tar_path.name} to extract observations...")
    records = []

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".dly"):
                continue
            if target_names and member.name not in target_names:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            lines = io.TextIOWrapper(extracted, encoding="ascii", errors="ignore")
            for rec in parse_dly_content(lines, start_year=start_year, end_year=end_year):
                records.append(rec)

    logger.info(f"Parsed {len(records)} raw element records. Building DataFrame...")
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        logger.warning("No observations found matching criteria.")
        return pd.DataFrame()

    # Pivot elements into [station_id, date, TMAX, TMIN, PRCP]
    pivoted = df_raw.pivot_table(
        index=["station_id", "date", "year", "month", "day", "day_of_year"],
        columns="element",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivoted.columns.name = None
    pivoted["date"] = pd.to_datetime(pivoted["date"])
    pivoted = pivoted.sort_values(["station_id", "date"]).reset_index(drop=True)

    # Save to Parquet if output path requested
    if output_parquet_path:
        out_p = Path(output_parquet_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pivoted.to_parquet(out_p, engine="pyarrow", compression="zstd", index=False)
        logger.info(f"Saved observations dataset to {out_p} ({len(pivoted)} rows)")

    return pivoted
