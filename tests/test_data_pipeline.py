"""Tests for NOAA GHCN data parsing, climatology engine, and scaling."""

import numpy as np
import pandas as pd
import pytest

from climate_forecast.data.climatology import ClimatologyEngine, DayClimatology, StationClimatology
from climate_forecast.data.dataset import ClimateDataScaler, SpatiotemporalClimateDataset
from climate_forecast.data.parser import parse_dly_content


def test_parse_dly_content():
    # 8 chars per day: 5 value chars + 3 flag chars
    sample_dly_line = (
        "USW00023183199507TMAX"
        + "  350   " * 31  # 35.0 C for all 31 days (8 chars each)
    )
    records = list(parse_dly_content([sample_dly_line], allowed_elements={"TMAX"}))
    assert len(records) == 31
    assert records[0]["station_id"] == "USW00023183"
    assert records[0]["element"] == "TMAX"
    assert np.isclose(records[0]["value"], 35.0)


def test_climatology_engine():
    # Build toy dataframe
    dates = pd.date_range("2000-01-01", "2020-12-31", freq="D")
    df = pd.DataFrame({
        "station_id": "TEST0001",
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "day_of_year": dates.dayofyear,
        "TMAX": 25.0 + 10.0 * np.sin(2.0 * np.pi * (dates.dayofyear - 80) / 365.25),
        "TMIN": 15.0 + 10.0 * np.sin(2.0 * np.pi * (dates.dayofyear - 80) / 365.25),
        "PRCP": np.random.exponential(2.0, size=len(dates)),
    })

    engine = ClimatologyEngine.fit_from_dataframe(df)
    st = engine.get_station("TEST0001")
    assert st is not None
    assert st.all_time_tmax_record >= 30.0

    # Day 171 is near peak summer: 25 + 10*sin(pi/2) ~ 35.0 C
    norm_summer = engine.get_norm("TEST0001", 171)
    assert norm_summer is not None
    assert norm_summer.tmax_mean > 32.0


def test_data_scaler():
    df = pd.DataFrame({
        "TMAX": [10.0, 20.0, 30.0],
        "TMIN": [0.0, 10.0, 20.0],
        "PRCP": [0.0, 5.0, 20.0],
    })
    scaler = ClimateDataScaler().fit(df)

    tmax_arr = np.array([20.0])
    norm = scaler.transform_tmax(tmax_arr)
    recon = scaler.inverse_transform_tmax(norm)
    assert np.isclose(recon[0], 20.0)
