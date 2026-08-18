"""FastAPI REST server for Global Extreme Climate & Heatwave Forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from climate_forecast.config import (
    CHECKPOINTS_DIR,
    HF_EXPORT_DIR,
    PROCESSED_DATA_DIR,
    STATIC_DIR,
)
from climate_forecast.inference.predictor import ClimatePredictor

app = FastAPI(
    title="Global Extreme Climate & Heatwave Forecaster API",
    description="Probabilistic spatiotemporal climate forecasting and extreme weather detection API trained on NOAA GHCN-Daily.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor instance (lazy-loaded)
_predictor: Optional[ClimatePredictor] = None
_world_geojson_cache: Optional[Dict] = None
_regions_geojson_cache: Optional[Dict] = None
_polygon_lookup_cache: Optional[List[Tuple[str, str, List]]] = None


def get_predictor() -> ClimatePredictor:
    global _predictor
    if _predictor is None:
        _predictor = ClimatePredictor()
    return _predictor


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    """Ray-casting algorithm for 2D point-in-polygon test."""
    n = len(poly)
    inside = False
    if n < 3:
        return False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _get_location_name(lat: float, lon: float) -> str:
    """Resolve latitude and longitude to subnational region and country name."""
    global _polygon_lookup_cache
    if _polygon_lookup_cache is None:
        # 1. Admin-1 comprehensive subnational regions (Europe, Americas, Asia, Japan, Africa, Oceania)
        admin1_path = STATIC_DIR / "geojson" / "admin1_regions.geojson"
        if admin1_path.exists():
            with open(admin1_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                for feat in d.get("features", []):
                    p = feat.get("properties", {})
                    rname = p.get("name") or p.get("NAME_1") or ""
                    cname = p.get("admin") or p.get("adm0_name") or ""
                    if rname:
                        cache.append((rname, cname, feat.get("geometry", {})))

        # 2. World countries fallback
        c_path = STATIC_DIR / "geojson" / "countries.geojson"
        if c_path.exists():
            with open(c_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                for feat in d.get("features", []):
                    p = feat.get("properties", {})
                    cname = p.get("name") or "Country"
                    cache.append(("", cname, feat.get("geometry", {})))

        _polygon_lookup_cache = cache

    # Check match
    for rname, cname, geom in _polygon_lookup_cache:
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        matched = False
        try:
            if gtype == "Polygon" and coords:
                matched = _point_in_polygon(lon, lat, coords[0])
            elif gtype == "MultiPolygon" and coords:
                for poly in coords:
                    if _point_in_polygon(lon, lat, poly[0]):
                        matched = True
                        break
        except Exception:
            matched = False

        if matched:
            if rname and cname:
                return f"{rname}, {cname}"
            elif cname:
                return f"{cname}"

    return f"Coordinates ({lat:.2f}°, {lon:.2f}°)"


class CustomPointRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees (-180 to 180)")
    elevation: float = Field(100.0, description="Elevation in meters")
    reference_date: Optional[str] = Field("2024-07-15", description="Reference date YYYY-MM-DD")
    location_name: Optional[str] = Field(None, description="Optional city or region name")


class ClimateShiftSimulationRequest(BaseModel):
    station_id: str = Field(..., description="Target NOAA Station ID")
    delta_warming_c: float = Field(1.5, ge=-5.0, le=8.0, description="Global temperature anomaly shift in degrees C")
    reference_date: Optional[str] = Field("2024-07-15", description="Reference date YYYY-MM-DD")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Global Extreme Climate & Heatwave Forecaster API</h1>")


@app.get("/health")
async def health_check():
    predictor = get_predictor()
    num_stations = len(predictor.climatology.stations_clim)
    return {
        "status": "healthy",
        "service": "climate-forecast",
        "device": str(predictor.device),
        "total_stations_available": num_stations,
    }


@app.get("/api/stations")
async def get_stations():
    """Retrieve catalog of all global weather stations with metadata and all-time records."""
    predictor = get_predictor()
    stations_list = []
    for st_id, st in predictor.climatology.stations_clim.items():
        stations_list.append({
            "station_id": st.station_id,
            "name": st.name,
            "country_name": st.country_name,
            "latitude": st.latitude,
            "longitude": st.longitude,
            "elevation": st.elevation,
            "climate_zone": st.climate_zone,
            "all_time_tmax_record": st.all_time_tmax_record,
            "all_time_tmin_record": st.all_time_tmin_record,
            "all_time_prcp_record": st.all_time_prcp_record,
        })
    return {"count": len(stations_list), "stations": stations_list}


@app.get("/api/world_choropleth")
async def get_world_choropleth():
    """Serve GeoJSON world countries with climate thermal regime and centroid coordinates for click forecast."""
    global _world_geojson_cache
    if _world_geojson_cache is not None:
        return _world_geojson_cache

    geojson_path = STATIC_DIR / "geojson" / "countries.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="countries.geojson not found")

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictor = get_predictor()
    country_stats: Dict[str, Dict] = {}
    for st_id, st in predictor.climatology.stations_clim.items():
        c_code = st_id[:2]
        if c_code not in country_stats:
            country_stats[c_code] = {
                "tmax_records": [],
                "climate_zones": set(),
                "stations": [],
            }
        country_stats[c_code]["tmax_records"].append(st.all_time_tmax_record)
        country_stats[c_code]["climate_zones"].add(st.climate_zone)
        country_stats[c_code]["stations"].append(st_id)

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        c_alpha2 = props.get("ISO3166-1-Alpha-2", "")

        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        avg_lat, avg_lon = 20.0, 0.0
        try:
            if geom.get("type") == "Polygon" and coords:
                avg_lat = sum(p[1] for p in coords[0]) / max(1, len(coords[0]))
                avg_lon = sum(p[0] for p in coords[0]) / max(1, len(coords[0]))
            elif geom.get("type") == "MultiPolygon" and coords:
                all_pts = [p for poly in coords for p in poly[0]]
                avg_lat = sum(p[1] for p in all_pts) / max(1, len(all_pts))
                avg_lon = sum(p[0] for p in all_pts) / max(1, len(all_pts))
        except Exception:
            avg_lat, avg_lon = 20.0, 0.0

        stats = country_stats.get(c_alpha2)
        if stats and stats["tmax_records"]:
            avg_tmax_rec = round(float(sum(stats["tmax_records"]) / len(stats["tmax_records"])), 1)
            czone = list(stats["climate_zones"])[0]
            st_ids = stats["stations"]
        else:
            abs_lat = abs(avg_lat)
            if abs_lat < 23.5:
                avg_tmax_rec = 38.0
                czone = "Tropical"
            elif abs_lat < 35.0:
                avg_tmax_rec = 35.0
                czone = "Subtropical"
            elif abs_lat < 55.0:
                avg_tmax_rec = 30.0
                czone = "Temperate"
            else:
                avg_tmax_rec = 22.0
                czone = "Boreal/Polar"
            st_ids = []

        props["avg_tmax_record"] = avg_tmax_rec
        props["climate_zone"] = czone
        props["stations_count"] = len(st_ids)
        props["primary_station_id"] = st_ids[0] if st_ids else None
        props["centroid_lat"] = round(avg_lat, 4)
        props["centroid_lon"] = round(avg_lon, 4)

    _world_geojson_cache = data
    return data


@app.get("/api/regions_choropleth")
async def get_regions_choropleth():
    """Serve Subnational Admin-1 Climate Regions (States, Provinces, Prefectures)."""
    global _regions_geojson_cache
    if _regions_geojson_cache is not None:
        return _regions_geojson_cache

    geojson_path = STATIC_DIR / "geojson" / "admin1_regions.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="admin1_regions.geojson not found")

    with open(geojson_path, "r", encoding="utf-8") as f:
        admin1_data = json.load(f)

    _regions_geojson_cache = admin1_data
    return _regions_geojson_cache


@app.get("/api/forecast/{station_id}")
async def get_station_forecast(
    station_id: str,
    date: Optional[str] = Query("2024-07-15", description="Reference date YYYY-MM-DD"),
    horizon: int = Query(14, ge=1, le=14, description="Forecast horizon days"),
    delta_warming_c: float = Query(0.0, description="Optional temperature anomaly shift in C"),
):
    predictor = get_predictor()
    try:
        result = predictor.predict_station(
            station_id=station_id,
            reference_date=date,
            horizon_days=horizon,
            delta_warming_c=delta_warming_c,
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/forecast/custom-point")
async def get_custom_point_forecast(req: CustomPointRequest):
    predictor = get_predictor()
    try:
        result = predictor.predict_custom_coordinates(
            latitude=req.latitude,
            longitude=req.longitude,
            elevation=req.elevation,
            reference_date=req.reference_date,
        )
        # Identify location name from spatial boundaries
        loc_name = req.location_name or _get_location_name(req.latitude, req.longitude)
        result["location_name"] = loc_name
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spatial interpolation error: {str(e)}")


@app.post("/api/simulate-climate-shift")
async def simulate_climate_shift(req: ClimateShiftSimulationRequest):
    predictor = get_predictor()
    try:
        base_res = predictor.predict_station(
            station_id=req.station_id,
            reference_date=req.reference_date,
            delta_warming_c=0.0,
        )
        shifted_res = predictor.predict_station(
            station_id=req.station_id,
            reference_date=req.reference_date,
            delta_warming_c=req.delta_warming_c,
        )

        return {
            "station_id": req.station_id,
            "station_name": base_res.station_name,
            "delta_warming_c": req.delta_warming_c,
            "baseline": {
                "max_tmax": base_res.max_forecast_tmax,
                "ehf_index": base_res.excess_heat_factor_ehf,
                "ehf_tier": base_res.ehf_alert_tier,
                "has_heatwave_warning": base_res.has_active_heatwave_warning,
            },
            "simulated": {
                "max_tmax": shifted_res.max_forecast_tmax,
                "ehf_index": shifted_res.excess_heat_factor_ehf,
                "ehf_tier": shifted_res.ehf_alert_tier,
                "has_heatwave_warning": shifted_res.has_active_heatwave_warning,
            },
            "daily_shift_comparison": [
                {
                    "date": pt_base.forecast_date,
                    "baseline_tmax_p50": pt_base.tmax_p50,
                    "simulated_tmax_p50": pt_shift.tmax_p50,
                    "baseline_hw_prob": pt_base.heatwave_hazard["probability"],
                    "simulated_hw_prob": pt_shift.heatwave_hazard["probability"],
                    "simulated_hw_level": pt_shift.heatwave_hazard["level"],
                }
                for pt_base, pt_shift in zip(base_res.daily_rollout, shifted_res.daily_rollout)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/api/extremes-summary")
async def get_extremes_summary():
    predictor = get_predictor()
    heatwaves = []
    frosts = []
    deluges = []

    for st_id, st in predictor.climatology.stations_clim.items():
        try:
            res = predictor.predict_station(st_id, reference_date="2024-07-15", horizon_days=7)
            if res.has_active_heatwave_warning:
                heatwaves.append({
                    "station_id": st_id,
                    "name": st.name,
                    "country": st.country_name,
                    "lat": st.latitude,
                    "lon": st.longitude,
                    "max_tmax": res.max_forecast_tmax,
                    "ehf_tier": res.ehf_alert_tier,
                })
            if res.has_active_frost_warning:
                frosts.append({
                    "station_id": st_id,
                    "name": st.name,
                    "country": st.country_name,
                    "lat": st.latitude,
                    "lon": st.longitude,
                    "min_tmin": res.min_forecast_tmin,
                })
            if res.has_active_deluge_warning:
                deluges.append({
                    "station_id": st_id,
                    "name": st.name,
                    "country": st.country_name,
                    "lat": st.latitude,
                    "lon": st.longitude,
                    "total_rain": res.total_forecast_prcp_mm,
                })
        except Exception:
            continue

    return {
        "active_heatwave_stations": heatwaves,
        "active_frost_stations": frosts,
        "active_deluge_stations": deluges,
        "counts": {
            "heatwaves": len(heatwaves),
            "frosts": len(frosts),
            "deluges": len(deluges),
        }
    }


@app.get("/api/metrics")
async def get_model_metrics():
    metrics_path = CHECKPOINTS_DIR / "benchmark_report.json"
    if not metrics_path.exists():
        metrics_path = CHECKPOINTS_DIR / "test_evaluation_metrics.json"

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Benchmark metrics will be available after training"}
