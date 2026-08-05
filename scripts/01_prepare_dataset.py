"""Step 1: Extract and curate NOAA GHCN global stations dataset and compute 30-year climatology."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from climate_forecast.config import (
    COUNTRIES_FILE,
    INVENTORY_FILE,
    PROCESSED_DATA_DIR,
    RAW_TAR_ARCHIVE,
    STATIONS_FILE,
)
from climate_forecast.data.climatology import ClimatologyEngine
from climate_forecast.data.parser import (
    extract_global_stations_dataset,
    load_stations_metadata,
    select_global_representative_stations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare NOAA GHCN dataset & climatology")
    parser.add_argument("--num-stations", type=int, default=120, help="Number of representative global stations")
    parser.add_argument("--start-year", type=int, default=2000, help="Start year of observation records")
    parser.add_argument("--end-year", type=int, default=2026, help="End year of observation records")
    parser.add_argument("--output-parquet", default=str(PROCESSED_DATA_DIR / "observations.parquet"))
    parser.add_argument("--output-climatology", default=str(PROCESSED_DATA_DIR / "stations_climatology.json"))
    args = parser.parse_args()

    logger.info("1. Loading NOAA GHCN metadata...")
    stations_meta = load_stations_metadata(STATIONS_FILE, COUNTRIES_FILE)
    logger.info(f"Loaded {len(stations_meta)} station records from {STATIONS_FILE}")

    logger.info(f"2. Selecting {args.num_stations} diverse global representative stations across all climate zones...")
    selected_station_ids = select_global_representative_stations(
        inventory_path=INVENTORY_FILE,
        stations_metadata=stations_meta,
        min_start_year=1995,
        min_end_year=2024,
        target_count=args.num_stations,
    )
    logger.info(f"Selected {len(selected_station_ids)} stations. Sample: {selected_station_ids[:10]}")

    logger.info(f"3. Extracting daily observation records from archive {RAW_TAR_ARCHIVE}...")
    df_obs = extract_global_stations_dataset(
        tar_archive_path=RAW_TAR_ARCHIVE,
        selected_stations=set(selected_station_ids),
        output_parquet_path=args.output_parquet,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    logger.info(f"4. Computing 30-year climatological normals (1991-2020) and daily extreme thresholds...")
    climatology = ClimatologyEngine.fit_from_dataframe(
        df=df_obs,
        stations_metadata=stations_meta,
    )
    climatology.save_json(args.output_climatology)
    logger.info(f"Climatology catalog successfully saved to {args.output_climatology}")
    logger.info("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
