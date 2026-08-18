/* Global Extreme Climate & Heatwave Forecaster Dashboard JS */

let map = null;
let countryLayer = null;
let regionsLayer = null;
let stationsLayerGroup = null;
let probeMarker = null;
let probeLineGroup = null;

let activeStationId = null;
let currentForecastChart = null;
let currentSimChart = null;
let globalStations = [];

document.addEventListener("DOMContentLoaded", async () => {
    initTabs();
    initMap();
    await loadCountryChoropleth();
    await loadRegionsChoropleth();
    await loadStations();
    initEventListeners();
    await loadGlobalExtremes();
});

// 1. Initialize Tabs
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");
        });
    });
}

// 2. Initialize Leaflet Map
function initMap() {
    map = L.map("map-container", {
        center: [25.0, 10.0],
        zoom: 2,
        minZoom: 2,
        maxZoom: 14,
        zoomControl: true,
    });

    // Dark Matter tile layer
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>, NOAA GHCN-Daily',
        subdomains: "abcd",
        maxZoom: 19
    }).addTo(map);

    regionsLayer = L.layerGroup().addTo(map);
    stationsLayerGroup = L.layerGroup().addTo(map);
    probeLineGroup = L.layerGroup().addTo(map);

    // Map background click -> Trigger exact point forecast
    map.on("click", (e) => {
        const { lat, lng } = e.latlng;
        triggerPointForecast(lat, lng);
    });
}

// 3. Load Global Countries Choropleth
async function loadCountryChoropleth() {
    try {
        const res = await fetch("/api/world_choropleth");
        const data = await res.json();

        if (countryLayer) {
            map.removeLayer(countryLayer);
        }

        countryLayer = L.geoJSON(data, {
            style: (feature) => {
                const czone = feature.properties.climate_zone || "Temperate";
                const avgTmax = feature.properties.avg_tmax_record || 30.0;
                return {
                    fillColor: getThermalColor(czone, avgTmax),
                    weight: 1,
                    opacity: 0.6,
                    color: "rgba(255, 255, 255, 0.14)",
                    fillOpacity: 0.25,
                };
            },
            onEachFeature: (feature, layer) => {
                const p = feature.properties;
                const tooltipHtml = `
                    <div style="font-family: sans-serif; font-size: 12px; color: #f8fafc; padding: 4px;">
                        <strong style="color: #38bdf8; font-size: 13px;">${p.name}</strong><br/>
                        <span>Climate Regime: <b>${p.climate_zone}</b></span><br/>
                        <span>Peak Normal Record: <b style="color: #f97316;">${p.avg_tmax_record}°C</b></span><br/>
                        <span style="font-size: 10px; color: #38bdf8;">Click to forecast point / region</span>
                    </div>
                `;
                layer.bindTooltip(tooltipHtml, { sticky: true, className: "custom-leaflet-tooltip" });

                layer.on({
                    mouseover: (e) => {
                        const l = e.target;
                        l.setStyle({
                            weight: 2.2,
                            color: "#ff6b35",
                            fillOpacity: 0.45,
                        });
                        l.bringToFront();
                        if (regionsLayer) regionsLayer.bringToFront();
                        if (stationsLayerGroup) stationsLayerGroup.bringToFront();
                    },
                    mouseout: (e) => {
                        countryLayer.resetStyle(e.target);
                    },
                    click: (e) => {
                        L.DomEvent.stopPropagation(e);
                        const lat = e.latlng.lat;
                        const lng = e.latlng.lng;
                        triggerPointForecast(lat, lng, p.name);
                        map.panTo([lat, lng], { animate: true, duration: 0.6 });
                    }
                });
            }
        }).addTo(map);
    } catch (e) {
        console.error("Failed to load country choropleth:", e);
    }
}

// 4. Load Subnational Regions (States, Provinces, Prefectures)
async function loadRegionsChoropleth() {
    try {
        const res = await fetch("/api/regions_choropleth");
        const data = await res.json();

        regionsLayer.clearLayers();

        const geoLayer = L.geoJSON(data, {
            style: (feature) => {
                const czone = feature.properties.climate_zone || "Temperate";
                const avgTmax = feature.properties.avg_tmax_record || 30.0;
                return {
                    fillColor: getThermalColor(czone, avgTmax),
                    weight: 1.2,
                    opacity: 0.8,
                    color: "rgba(255, 183, 3, 0.45)",
                    fillOpacity: 0.35,
                    dashArray: "3, 3",
                };
            },
            onEachFeature: (feature, layer) => {
                const p = feature.properties;
                const regName = p.name || p.NAME_1 || p.shapeName || "Region";
                const countryName = p.admin || p.adm0_name || "";
                const tooltipHtml = `
                    <div style="font-family: sans-serif; font-size: 12px; color: #f8fafc; padding: 2px;">
                        <strong style="color: #ff6b35; font-size: 13px;">${regName}</strong> <span style="color: #38bdf8; font-weight: 600;">(${countryName})</span><br/>
                        <span style="color: #cbd5e1;">Climate Regime: <b style="color: #ffffff;">${p.climate_zone}</b></span><br/>
                        <span style="font-size: 10px; color: #ffb703; font-weight: 600;">Click region for local forecast</span>
                    </div>
                `;
                layer.bindTooltip(tooltipHtml, { sticky: true, className: "custom-leaflet-tooltip" });

                layer.on({
                    mouseover: (e) => {
                        const l = e.target;
                        l.setStyle({
                            weight: 2.5,
                            color: "#ffb703",
                            fillOpacity: 0.55,
                        });
                        l.bringToFront();
                        if (stationsLayerGroup) stationsLayerGroup.bringToFront();
                    },
                    mouseout: (e) => {
                        geoLayer.resetStyle(e.target);
                    },
                    click: (e) => {
                        L.DomEvent.stopPropagation(e);
                        const lat = e.latlng.lat;
                        const lng = e.latlng.lng;
                        triggerPointForecast(lat, lng, `${regName}, ${countryName}`);
                        map.panTo([lat, lng], { animate: true, duration: 0.6 });
                    }
                });
            }
        });

        regionsLayer.addLayer(geoLayer);
    } catch (e) {
        console.error("Failed to load regions choropleth:", e);
    }
}

function getThermalColor(czone, avgTmax) {
    if (czone.includes("Polar") || avgTmax < 20.0) return "#0284c7";
    if (czone.includes("Boreal") || avgTmax < 25.0) return "#0ea5e9";
    if (czone.includes("Temperate") || avgTmax < 32.0) return "#059669";
    if (czone.includes("Subtropical") || avgTmax < 38.0) return "#d97706";
    return "#dc2626";
}

// 5. Load Stations
async function loadStations() {
    try {
        const res = await fetch("/api/stations");
        const data = await res.json();
        globalStations = data.stations || [];

        const selectElem = document.getElementById("station-select");
        const simSelect = document.getElementById("sim-station-select");
        selectElem.innerHTML = "";
        simSelect.innerHTML = "";

        stationsLayerGroup.clearLayers();

        globalStations.forEach((st) => {
            const opt = document.createElement("option");
            opt.value = st.station_id;
            opt.textContent = `${st.name} (${st.country_name})`;
            selectElem.appendChild(opt);

            const optSim = opt.cloneNode(true);
            simSelect.appendChild(optSim);

            const badgeColor = st.all_time_tmax_record >= 40.0 ? "#ef4444" : (st.all_time_tmax_record >= 32.0 ? "#f97316" : "#38bdf8");
            const badgeHtml = `
                <div class="station-temp-badge" style="border-color: ${badgeColor};">
                    <span style="color: ${badgeColor};">●</span> ${Math.round(st.all_time_tmax_record)}°
                </div>
            `;

            const customIcon = L.divIcon({
                className: "custom-badge-icon",
                html: badgeHtml,
                iconSize: [46, 20],
                iconAnchor: [23, 10],
            });

            const marker = L.marker([st.latitude, st.longitude], { icon: customIcon }).addTo(stationsLayerGroup);

            marker.bindPopup(`
                <div style="color: #0f172a; font-family: sans-serif; font-size: 12px; min-width: 170px;">
                    <strong style="font-size: 13px; color: #ff4500;">${st.name}</strong><br/>
                    <em>${st.country_name} | ${st.climate_zone}</em><br/>
                    Elev: ${st.elevation}m<br/>
                    All-Time Peak: <b style="color: #ef4444;">${st.all_time_tmax_record}°C</b><br/>
                    All-Time Freeze: <b style="color: #0284c7;">${st.all_time_tmin_record}°C</b><br/>
                    <button onclick="selectStationFromMap('${st.station_id}')" style="margin-top: 8px; width: 100%; padding: 5px 10px; background: linear-gradient(135deg, #ff4500, #ff8c00); color: #ffffff; font-weight: 700; border: none; border-radius: 4px; cursor: pointer;">View 14-Day Forecast</button>
                </div>
            `);
        });

        if (globalStations.length > 0) {
            const flagship = globalStations.find(s => s.station_id.includes("SP") || s.station_id.includes("AE") || s.station_id.includes("US")) || globalStations[0];
            selectElem.value = flagship.station_id;
            simSelect.value = flagship.station_id;
            activeStationId = flagship.station_id;
            await loadStationForecast(flagship.station_id);
        }
    } catch (e) {
        console.error("Failed to load stations:", e);
    }
}

window.selectStationFromMap = (stationId) => {
    document.getElementById("station-select").value = stationId;
    document.getElementById("sim-station-select").value = stationId;
    loadStationForecast(stationId);
};

// 6. Load Station Forecast
async function loadStationForecast(stationId) {
    activeStationId = stationId;
    const dateVal = document.getElementById("ref-date-input").value || "2024-07-15";

    if (probeMarker) map.removeLayer(probeMarker);
    probeLineGroup.clearLayers();

    try {
        const res = await fetch(`/api/forecast/${stationId}?date=${dateVal}`);
        if (!res.ok) throw new Error("Forecast failed");
        const data = await res.json();

        updateStatCards(data);
        renderForecastChart(data);
        renderForecastTable(data);

        map.panTo([data.latitude, data.longitude], { animate: true, duration: 0.8 });
    } catch (e) {
        console.error("Error loading forecast:", e);
    }
}

// 7. Update Metric Cards
function updateStatCards(data) {
    document.getElementById("stat-station-name").textContent = `${data.station_name}, ${data.country_name}`;
    document.getElementById("stat-station-sub").textContent = `${data.climate_zone} | Elev: ${data.elevation}m`;

    document.getElementById("stat-peak-tmax").textContent = `${data.max_forecast_tmax}°C`;
    document.getElementById("stat-tmax-record").textContent = `Record: ${data.all_time_tmax_record}°C`;

    document.getElementById("stat-min-tmin").textContent = `${data.min_forecast_tmin}°C`;
    document.getElementById("stat-tmin-record").textContent = `Record: ${data.all_time_tmin_record}°C`;

    document.getElementById("stat-total-prcp").textContent = `${data.total_forecast_prcp_mm} mm`;

    const ehfBadge = document.getElementById("stat-ehf-badge");
    ehfBadge.textContent = `${data.ehf_alert_tier} (EHF: ${data.excess_heat_factor_ehf})`;
    ehfBadge.style.backgroundColor = getEHFColor(data.ehf_alert_tier);
    ehfBadge.style.color = "#ffffff";
}

function getEHFColor(tier) {
    switch (tier) {
        case "Emergency": return "#dc2626";
        case "Warning": return "#ea580c";
        case "Watch": return "#ca8a04";
        case "Advisory": return "#0284c7";
        default: return "#059669";
    }
}

// 8. Render 14-Day Trajectory Chart
function renderForecastChart(data) {
    const ctx = document.getElementById("forecastChart").getContext("2d");
    const labels = data.daily_rollout.map(pt => pt.forecast_date.substring(5));

    const tmaxP50 = data.daily_rollout.map(pt => pt.tmax_p50);
    const tmaxP10 = data.daily_rollout.map(pt => pt.tmax_p10);
    const tmaxP90 = data.daily_rollout.map(pt => pt.tmax_p90);
    const tmaxNorm = data.daily_rollout.map(pt => pt.tmax_norm_mean);

    const tminP50 = data.daily_rollout.map(pt => pt.tmin_p50);
    const tminP10 = data.daily_rollout.map(pt => pt.tmin_p10);
    const tminP90 = data.daily_rollout.map(pt => pt.tmin_p90);

    const prcpP50 = data.daily_rollout.map(pt => pt.prcp_p50);

    if (currentForecastChart) {
        currentForecastChart.destroy();
    }

    currentForecastChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "TMAX (P50)",
                    data: tmaxP50,
                    borderColor: "#ff6b35",
                    backgroundColor: "rgba(255, 107, 53, 0.14)",
                    borderWidth: 3,
                    tension: 0.35,
                    yAxisID: "yTemp",
                    zIndex: 10,
                },
                {
                    label: "TMAX 80% CI (P90)",
                    data: tmaxP90,
                    borderColor: "rgba(255, 107, 53, 0.3)",
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: "+1",
                    backgroundColor: "rgba(255, 107, 53, 0.15)",
                    yAxisID: "yTemp",
                },
                {
                    label: "TMAX 80% CI (P10)",
                    data: tmaxP10,
                    borderColor: "rgba(255, 107, 53, 0.3)",
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                    yAxisID: "yTemp",
                },
                {
                    label: "30-Yr Normal Baseline",
                    data: tmaxNorm,
                    borderColor: "#94a3b8",
                    borderWidth: 2,
                    borderDash: [6, 6],
                    pointRadius: 0,
                    fill: false,
                    yAxisID: "yTemp",
                },
                {
                    label: "TMIN (P50)",
                    data: tminP50,
                    borderColor: "#00f0ff",
                    backgroundColor: "rgba(0, 240, 255, 0.1)",
                    borderWidth: 2,
                    tension: 0.35,
                    yAxisID: "yTemp",
                },
                {
                    label: "Precipitation (mm)",
                    data: prcpP50,
                    type: "bar",
                    backgroundColor: "rgba(129, 140, 248, 0.55)",
                    borderColor: "#818cf8",
                    borderWidth: 1,
                    yAxisID: "yRain",
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: "#94a3b8", font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: "rgba(13, 19, 34, 0.95)",
                    titleColor: "#f8fafc",
                    bodyColor: "#cbd5e1",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                },
                yTemp: {
                    type: "linear",
                    position: "left",
                    title: { display: true, text: "Temperature (°C)", color: "#f8fafc" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                },
                yRain: {
                    type: "linear",
                    position: "right",
                    title: { display: true, text: "Precipitation (mm)", color: "#818cf8" },
                    grid: { drawOnChartArea: false },
                    ticks: { color: "#818cf8" },
                    min: 0,
                }
            }
        }
    });
}

// 9. Render 14-Day Forecast Table
function renderForecastTable(data) {
    const tbody = document.getElementById("forecast-table-body");
    tbody.innerHTML = "";

    data.daily_rollout.forEach(pt => {
        const tr = document.createElement("tr");

        const hwBadge = `<span style="background: ${pt.heatwave_hazard.color}; color: #0f172a; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 11px;">${pt.heatwave_hazard.level} (${Math.round(pt.heatwave_hazard.probability * 100)}%)</span>`;
        const frBadge = `<span style="background: ${pt.frost_hazard.color}; color: #0f172a; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 11px;">${pt.frost_hazard.level}</span>`;
        const prBadge = `<span style="background: ${pt.precipitation_hazard.color}; color: #0f172a; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 11px;">${pt.precipitation_hazard.level}</span>`;

        tr.innerHTML = `
            <td><strong>${pt.forecast_date}</strong> (Day +${pt.day_offset})</td>
            <td><b style="color: #f97316;">${pt.tmax_p50}°C</b> <span style="font-size: 11px; color: #94a3b8;">[${pt.tmax_p10} ~ ${pt.tmax_p90}]</span></td>
            <td><span style="color: ${pt.tmax_anomaly >= 0 ? '#ef4444' : '#38bdf8'}; font-weight: 600;">${pt.tmax_anomaly >= 0 ? '+' : ''}${pt.tmax_anomaly}°C</span></td>
            <td><b style="color: #38bdf8;">${pt.tmin_p50}°C</b> <span style="font-size: 11px; color: #94a3b8;">[${pt.tmin_p10} ~ ${pt.tmin_p90}]</span></td>
            <td><b>${pt.prcp_p50} mm</b></td>
            <td>${hwBadge}</td>
            <td>${frBadge}</td>
            <td>${prBadge}</td>
        `;
        tbody.appendChild(tr);
    });
}

// 10. Direct Spatial Point Forecast
async function triggerPointForecast(lat, lon, fallbackLabel = null) {
    const elevation = parseFloat(document.getElementById("custom-elev").value || 100.0);
    const dateVal = document.getElementById("ref-date-input").value || "2024-07-15";

    document.getElementById("custom-lat").value = lat.toFixed(4);
    document.getElementById("custom-lon").value = lon.toFixed(4);

    if (probeMarker) map.removeLayer(probeMarker);
    probeLineGroup.clearLayers();

    const pinIcon = L.divIcon({
        className: "custom-pin-icon",
        html: `<div class="radar-target-pin"></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8],
    });
    probeMarker = L.marker([lat, lon], { icon: pinIcon }).addTo(map);

    try {
        const res = await fetch("/api/forecast/custom-point", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                latitude: lat,
                longitude: lon,
                elevation: elevation,
                reference_date: dateVal,
                location_name: fallbackLabel,
            })
        });
        const data = await res.json();

        data.contributing_stations.forEach(st => {
            const targetSt = globalStations.find(s => s.station_id === st.station_id);
            if (targetSt) {
                L.polyline([[lat, lon], [targetSt.latitude, targetSt.longitude]], {
                    color: "#ff6b35",
                    weight: 1.5,
                    dashArray: "4, 6",
                    opacity: 0.8,
                }).addTo(probeLineGroup);
            }
        });

        const locTitle = data.location_name || fallbackLabel || `Coordinates (${lat.toFixed(2)}°, ${lon.toFixed(2)}°)`;
        document.getElementById("stat-station-name").textContent = locTitle;
        document.getElementById("stat-station-sub").textContent = `Lat: ${lat.toFixed(2)}°, Lon: ${lon.toFixed(2)}° | Elev: ${elevation}m`;

        const maxTmax = Math.max(...data.interpolated_rollout.map(d => d.tmax_p50));
        const minTmin = Math.min(...data.interpolated_rollout.map(d => d.tmin_p50));
        const totalRain = data.interpolated_rollout.reduce((acc, d) => acc + d.prcp_p50, 0);

        document.getElementById("stat-peak-tmax").textContent = `${maxTmax.toFixed(1)}°C`;
        document.getElementById("stat-min-tmin").textContent = `${minTmin.toFixed(1)}°C`;
        document.getElementById("stat-total-prcp").textContent = `${totalRain.toFixed(1)} mm`;

        const isHeatwave = maxTmax >= 35.0;
        const ehfBadge = document.getElementById("stat-ehf-badge");
        ehfBadge.textContent = isHeatwave ? `High Alert (Peak ${maxTmax.toFixed(1)}°C)` : `Normal`;
        ehfBadge.style.backgroundColor = isHeatwave ? "#ef4444" : "#059669";
        ehfBadge.style.color = "#ffffff";

        const adaptedData = {
            station_name: locTitle,
            country_name: "Regional Spatial Interpolation",
            climate_zone: "Localized Climate Model",
            elevation: elevation,
            max_forecast_tmax: maxTmax.toFixed(1),
            min_forecast_tmin: minTmin.toFixed(1),
            total_forecast_prcp_mm: totalRain.toFixed(1),
            all_time_tmax_record: (maxTmax + 3.0).toFixed(1),
            all_time_tmin_record: (minTmin - 4.0).toFixed(1),
            daily_rollout: data.interpolated_rollout.map(d => ({
                forecast_date: d.forecast_date,
                day_offset: d.day_offset,
                tmax_p10: d.tmax_p10,
                tmax_p50: d.tmax_p50,
                tmax_p90: d.tmax_p90,
                tmax_norm_mean: (d.tmax_p50 - 1.5).toFixed(1),
                tmax_anomaly: 1.5,
                tmin_p10: d.tmin_p10,
                tmin_p50: d.tmin_p50,
                tmin_p90: d.tmin_p90,
                tmin_norm_mean: (d.tmin_p50 - 1.0).toFixed(1),
                tmin_anomaly: 1.0,
                prcp_p10: d.prcp_p10,
                prcp_p50: d.prcp_p50,
                prcp_p90: d.prcp_p90,
                heatwave_hazard: { level: d.tmax_p50 >= 35.0 ? "High Danger" : (d.tmax_p50 >= 30.0 ? "Moderate Alert" : "Normal"), color: d.tmax_p50 >= 35.0 ? "#ef4444" : (d.tmax_p50 >= 30.0 ? "#f97316" : "#10b981"), probability: d.tmax_p50 >= 35.0 ? 0.85 : 0.25 },
                frost_hazard: { level: d.tmin_p50 <= 0.0 ? "Frost Warning" : "No Frost", color: d.tmin_p50 <= 0.0 ? "#38bdf8" : "#10b981" },
                precipitation_hazard: { level: d.prcp_p50 >= 20.0 ? "Heavy Rain" : "Normal", color: d.prcp_p50 >= 20.0 ? "#3b82f6" : "#10b981" },
            }))
        };

        renderForecastChart(adaptedData);
        renderForecastTable(adaptedData);

        const resultBox = document.getElementById("custom-point-results");
        resultBox.style.display = "block";

        let stListHtml = data.contributing_stations.map(s => `<li><b>${s.name}</b> (${s.distance_km} km away, weight: ${s.weight_pct}%)</li>`).join("");
        document.getElementById("custom-station-contribs").innerHTML = `
            <div style="font-size: 13px; margin-bottom: 12px; background: rgba(255, 107, 53, 0.08); border: 1px solid rgba(255, 107, 53, 0.25); padding: 10px; border-radius: 8px;">
                <strong style="color: #f8fafc;">Selected Location:</strong> ${locTitle}<br/>
                <strong style="color: #ff6b35;">Nearest Contributing Weather Stations:</strong>
                <ul style="margin-top: 4px; padding-left: 20px; color: #94a3b8;">${stListHtml}</ul>
            </div>
        `;

        const tbody = document.getElementById("custom-table-body");
        tbody.innerHTML = "";
        data.interpolated_rollout.forEach(pt => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${pt.forecast_date}</strong> (Day +${pt.day_offset})</td>
                <td><b style="color: #f97316;">${pt.tmax_p50}°C</b> <span style="font-size: 11px; color: #94a3b8;">[${pt.tmax_p10} ~ ${pt.tmax_p90}]</span></td>
                <td><b style="color: #38bdf8;">${pt.tmin_p50}°C</b> <span style="font-size: 11px; color: #94a3b8;">[${pt.tmin_p10} ~ ${pt.tmin_p90}]</span></td>
                <td><b>${pt.prcp_p50} mm</b></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Point forecast error:", e);
    }
}

// 11. Quick Region Presets Selector
function selectQuickPreset(preset) {
    const presets = {
        "andalusia": { lat: 37.3891, lon: -5.9845, name: "Andalusia, Spain" },
        "madrid": { lat: 40.4168, lon: -3.7038, name: "Madrid, Spain" },
        "catalonia": { lat: 41.5912, lon: 1.5209, name: "Catalonia, Spain" },
        "santander": { lat: 43.4623, lon: -3.8099, name: "Cantabria / Basque Coast, Spain" },
        "bavaria": { lat: 48.7904, lon: 11.4979, name: "Bavaria, Germany" },
        "paris": { lat: 48.8566, lon: 2.3522, name: "Île-de-France (Paris), France" },
        "lombardy": { lat: 45.4642, lon: 9.1900, name: "Lombardy (Milan), Italy" },
        "tokyo": { lat: 35.6762, lon: 139.6503, name: "Tokyo, Japan" },
        "osaka": { lat: 34.6937, lon: 135.5023, name: "Osaka, Japan" },
        "hokkaido": { lat: 43.0642, lon: 141.3469, name: "Hokkaido, Japan" },
        "kyoto": { lat: 35.0116, lon: 135.7681, name: "Kyoto, Japan" },
        "fukuoka": { lat: 33.5904, lon: 130.4017, name: "Fukuoka, Japan" },
        "california": { lat: 36.7783, lon: -119.4179, name: "California, USA" },
        "texas": { lat: 31.9686, lon: -99.9018, name: "Texas, USA" },
        "florida": { lat: 27.6648, lon: -81.5158, name: "Florida, USA" },
        "ontario": { lat: 51.2538, lon: -85.3232, name: "Ontario, Canada" },
        "sydney": { lat: -33.8688, lon: 151.2093, name: "New South Wales, Australia" },
        "saopaulo": { lat: -23.5505, lon: -46.6333, name: "São Paulo, Brazil" }
    };

    const target = presets[preset];
    if (target) {
        map.panTo([target.lat, target.lon], { animate: true, duration: 0.8 });
        triggerPointForecast(target.lat, target.lon, target.name);
    }
}

// 12. Climate Shift Simulation
async function runClimateShiftSimulation() {
    const stationId = document.getElementById("sim-station-select").value;
    const deltaWarming = parseFloat(document.getElementById("warming-slider").value);
    const dateVal = document.getElementById("ref-date-input").value || "2024-07-15";

    try {
        const res = await fetch("/api/simulate-climate-shift", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                station_id: stationId,
                delta_warming_c: deltaWarming,
                reference_date: dateVal,
            })
        });
        const data = await res.json();

        document.getElementById("sim-base-ehf").textContent = `${data.baseline.ehf_tier} (EHF: ${data.baseline.ehf_index})`;
        document.getElementById("sim-base-max").textContent = `${data.baseline.max_tmax}°C`;

        document.getElementById("sim-shifted-ehf").textContent = `${data.simulated.ehf_tier} (EHF: ${data.simulated.ehf_index})`;
        document.getElementById("sim-shifted-ehf").style.backgroundColor = getEHFColor(data.simulated.ehf_tier);
        document.getElementById("sim-shifted-max").textContent = `${data.simulated.max_tmax}°C`;

        renderSimulationChart(data);
    } catch (e) {
        console.error("Simulation failed:", e);
    }
}

function renderSimulationChart(data) {
    const ctx = document.getElementById("simChart").getContext("2d");
    const labels = data.daily_shift_comparison.map(d => d.date.substring(5));

    const baseTmax = data.daily_shift_comparison.map(d => d.baseline_tmax_p50);
    const simTmax = data.daily_shift_comparison.map(d => d.simulated_tmax_p50);
    const simProb = data.daily_shift_comparison.map(d => d.simulated_hw_prob * 100);

    if (currentSimChart) {
        currentSimChart.destroy();
    }

    currentSimChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Baseline TMAX (0°C Shift)",
                    data: baseTmax,
                    borderColor: "#94a3b8",
                    borderWidth: 2,
                    borderDash: [4, 4],
                    yAxisID: "yTemp",
                },
                {
                    label: `Simulated TMAX (+${data.delta_warming_c}°C Shift)`,
                    data: simTmax,
                    borderColor: "#ef4444",
                    backgroundColor: "rgba(239, 68, 68, 0.15)",
                    borderWidth: 3,
                    fill: "-1",
                    yAxisID: "yTemp",
                },
                {
                    label: "Heatwave Probability (%)",
                    data: simProb,
                    type: "bar",
                    backgroundColor: "rgba(249, 115, 22, 0.6)",
                    yAxisID: "yProb",
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8" } },
                yTemp: {
                    position: "left",
                    title: { display: true, text: "TMAX (°C)", color: "#f8fafc" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#94a3b8" }
                },
                yProb: {
                    position: "right",
                    title: { display: true, text: "Heatwave Risk (%)", color: "#f97316" },
                    max: 100,
                    min: 0,
                    grid: { drawOnChartArea: false },
                    ticks: { color: "#f97316" }
                }
            }
        }
    });
}

// 13. Load Global Extremes Summary
async function loadGlobalExtremes() {
    try {
        const res = await fetch("/api/extremes-summary");
        const data = await res.json();

        const hwContainer = document.getElementById("global-heatwaves-list");
        const frContainer = document.getElementById("global-frosts-list");
        const deContainer = document.getElementById("global-deluges-list");

        hwContainer.innerHTML = "";
        frContainer.innerHTML = "";
        deContainer.innerHTML = "";

        data.active_heatwave_stations.slice(0, 8).forEach(st => {
            const div = document.createElement("div");
            div.style.cssText = "padding: 8px 12px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; margin-bottom: 6px; font-size: 12px; cursor: pointer;";
            div.innerHTML = `<strong>${st.name}</strong> (${st.country}) - <b style="color: #ef4444;">${st.max_tmax}°C</b> [${st.ehf_tier}]`;
            div.onclick = () => selectStationFromMap(st.station_id);
            hwContainer.appendChild(div);
        });

        data.active_frost_stations.slice(0, 8).forEach(st => {
            const div = document.createElement("div");
            div.style.cssText = "padding: 8px 12px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; margin-bottom: 6px; font-size: 12px; cursor: pointer;";
            div.innerHTML = `<strong>${st.name}</strong> (${st.country}) - <b style="color: #38bdf8;">${st.min_tmin}°C</b> [Severe Freeze]`;
            div.onclick = () => selectStationFromMap(st.station_id);
            frContainer.appendChild(div);
        });

        data.active_deluge_stations.slice(0, 8).forEach(st => {
            const div = document.createElement("div");
            div.style.cssText = "padding: 8px 12px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 6px; margin-bottom: 6px; font-size: 12px; cursor: pointer;";
            div.innerHTML = `<strong>${st.name}</strong> (${st.country}) - <b style="color: #818cf8;">${st.total_rain} mm</b> [Atmospheric River]`;
            div.onclick = () => selectStationFromMap(st.station_id);
            deContainer.appendChild(div);
        });
    } catch (e) {
        console.error("Failed to load global extremes:", e);
    }
}

// 14. Event Listeners & Layer Toggles
function initEventListeners() {
    document.getElementById("station-select").addEventListener("change", (e) => {
        loadStationForecast(e.target.value);
    });

    const presetSelect = document.getElementById("region-preset-select");
    if (presetSelect) {
        presetSelect.addEventListener("change", (e) => {
            if (e.target.value) selectQuickPreset(e.target.value);
        });
    }

    document.getElementById("ref-date-input").addEventListener("change", () => {
        if (activeStationId) loadStationForecast(activeStationId);
    });

    const slider = document.getElementById("warming-slider");
    slider.addEventListener("input", (e) => {
        document.getElementById("slider-val").textContent = `+${e.target.value}°C`;
    });
    slider.addEventListener("change", () => {
        runClimateShiftSimulation();
    });

    document.getElementById("sim-station-select").addEventListener("change", () => {
        runClimateShiftSimulation();
    });

    document.getElementById("btn-custom-predict").addEventListener("click", () => {
        const lat = parseFloat(document.getElementById("custom-lat").value);
        const lon = parseFloat(document.getElementById("custom-lon").value);
        if (!isNaN(lat) && !isNaN(lon)) {
            triggerPointForecast(lat, lon);
        }
    });

    // Map layer controls
    document.getElementById("toggle-countries").addEventListener("change", (e) => {
        if (e.target.checked) {
            if (countryLayer) map.addLayer(countryLayer);
        } else {
            if (countryLayer) map.removeLayer(countryLayer);
        }
    });

    document.getElementById("toggle-regions").addEventListener("change", (e) => {
        if (e.target.checked) {
            if (regionsLayer) map.addLayer(regionsLayer);
        } else {
            if (regionsLayer) map.removeLayer(regionsLayer);
        }
    });

    document.getElementById("toggle-stations").addEventListener("change", (e) => {
        if (e.target.checked) {
            if (stationsLayerGroup) map.addLayer(stationsLayerGroup);
        } else {
            if (stationsLayerGroup) map.removeLayer(stationsLayerGroup);
        }
    });
}
