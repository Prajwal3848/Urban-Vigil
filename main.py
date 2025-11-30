# main.py - Urban Vigil Pro (restored + fixes)
# This file is the restored full backend with the requested fixes:
# - Adds safe imports (Body) and robust startup loads
# - Fixes duplicate __dt insertion bug in crime_trends
# - Keeps all original endpoints and logic; only small surgical fixes applied
# - Safe-route: OSRM attempt + fallback straight-line sampling with per-sample risk
#
# IMPORTANT: place your models and CSVs in the models/ folder:
# models/Final_model_fixed.pkl
# models/label_encoders_fixed.pkl
# models/place_lookup.csv
# models/bengaluru_dataset.csv
#
# Start with: uvicorn main:app --host 0.0.0.0 --port $PORT

from __future__ import annotations
import os
import sys
import math
import json
import logging
import traceback
from typing import Optional, Any, Dict, List, Tuple

import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ------------------ CONFIG ------------------
APP_TITLE = "Urban Vigil Pro - API"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_FILE = os.getenv("MODEL_FILE", "Final_model_fixed.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE)
ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders_fixed.pkl")
PLACE_LOOKUP_PATH = os.path.join(MODELS_DIR, "place_lookup.csv")
CRIME_CSV_PATH = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")
CRIME_MAP_PATH = os.path.join(MODELS_DIR, "crime_target_mapping.json")

# ------------------ LOGGING ------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("urban-vigil")

# ------------------ APP ------------------
app = FastAPI(title=APP_TITLE)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve frontend if present under /public
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
else:
    logger.warning("No public/ directory found. Place your index.html in public/ to serve frontend.")

# ------------------ GLOBALS ------------------
model = None
encoders: Dict[str, Any] = {}
crime_mapping: Dict[str, Any] = {}
place_lookup_df: pd.DataFrame = pd.DataFrame()
crime_df: Optional[pd.DataFrame] = None

location_cache: Dict[str, Dict[str, Any]] = {}
zones_cache: List[Dict[str, Any]] = []

# Sample police stations (Bengaluru)
BANGALORE_POLICE_STATIONS = [
    {"name": "Banashankari PS", "lat": 12.9250, "lon": 77.5670, "phone": "080-26771290"},
    {"name": "Koramangala PS", "lat": 12.9352, "lon": 77.6245, "phone": "080-25537466"},
    {"name": "Indiranagar PS", "lat": 12.9719, "lon": 77.6412, "phone": "080-25213009"},
    {"name": "MG Road PS", "lat": 12.9763, "lon": 77.5995, "phone": "080-22212444"},
    {"name": "Jayanagar PS", "lat": 12.9250, "lon": 77.5938, "phone": "080-26633292"},
    {"name":"Hebbal PS","lat":13.0350,"lon":77.5970,"phone":"080-23637700"},
    {"name":"Whitefield PS","lat":12.9698,"lon":77.7490,"phone":"080-28452905"},
    {"name":"Electronic City PS","lat":12.8419,"lon":77.6605,"phone":"080-27835215"},
    {"name":"JP Nagar PS","lat":12.9085,"lon":77.5850,"phone":"080-26493399"},
    {"name":"BTM Layout PS","lat":12.9165,"lon":77.6101,"phone":"080-26685336"},
    {"name":"Marathahalli PS","lat":12.9592,"lon":77.6974,"phone":"080-25221744"},
]

# ------------------ UTILITIES ------------------
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    aa = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(aa), math.sqrt(1-aa))

def geocode_location(q: str) -> Tuple[Optional[float], Optional[float]]:
    """Use Nominatim to geocode free text location names (best effort)."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "UrbanVigil/1.0 (contact: no-reply)"}
        resp = requests.get(url, params={"q": f"{q}, Bangalore, India", "format": "json", "limit": 1}, headers=headers, timeout=8)
        if resp.ok:
            data = resp.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.debug("geocode error: %s", e)
    return None, None

# ------------------ DATA LOADING & CACHE ------------------
def build_location_cache_from_df(df: pd.DataFrame):
    global location_cache, zones_cache
    location_cache = {}
    zones_cache = []
    if df is None or df.empty:
        logger.info("No crime_df to build location cache from.")
        return
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    type_col = next((c for c in df.columns if c.lower() in ("type", "crime", "offence")), None)
    if not lat_col or not lon_col:
        logger.warning("Dataframe lacks lat/lon columns required for cache.")
        return
    d = df.copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d = d.dropna(subset=[lat_col, lon_col])
    if d.empty:
        logger.info("No numeric lat/lon values in crime dataframe.")
        return
    d["lat_bin"] = (d[lat_col] * 100).round() / 100
    d["lon_bin"] = (d[lon_col] * 100).round() / 100
    agg = d.groupby(["lat_bin", "lon_bin"]).agg(
        crime_count=(type_col if type_col else lat_col, "count"),
        most_common=(type_col if type_col else lat_col, lambda x: x.mode().iloc[0] if len(x) else "Unknown")
    ).reset_index().rename(columns={"lat_bin":"lat","lon_bin":"lon"})
    for _, r in agg.iterrows():
        key = f"{r.lat:.4f},{r.lon:.4f}"
        location_cache[key] = {"lat": float(r.lat), "lon": float(r.lon), "crime_count": int(r.crime_count), "most_common": str(r.most_common)}
    if not agg.empty:
        maxc = agg["crime_count"].max()
        for _, r in agg.sort_values("crime_count", ascending=False).head(150).iterrows():
            radius = 200 + (float(r.crime_count) / maxc) * 1800 if maxc > 0 else 400
            level = "high" if r.crime_count >= max(3, 0.6 * maxc) else ("medium" if r.crime_count >= 0.2 * maxc else "low")
            zones_cache.append({"lat": float(r.lat), "lon": float(r.lon), "radius_m": int(radius), "level": level, "crime_count": int(r.crime_count)})
    logger.info("Built location cache (clusters=%d, zones=%d)", len(location_cache), len(zones_cache))

def safe_startup_loads():
    """Load model, encoders, place lookup and crime CSV safely (no fatal exceptions)."""
    global model, encoders, place_lookup_df, crime_df, crime_mapping
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info("✅ Model loaded from %s", MODEL_PATH)
        else:
            logger.warning("Model file not found at %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Model load failed: %s", e)
        model = None

    try:
        if os.path.exists(ENCODERS_PATH):
            encoders = joblib.load(ENCODERS_PATH)
            logger.info("✅ Encoders loaded")
    except Exception as e:
        logger.debug("Encoders load issue (continued): %s", e)

    try:
        if os.path.exists(CRIME_MAP_PATH):
            with open(CRIME_MAP_PATH, "r", encoding="utf-8") as fh:
                crime_mapping = json.load(fh)
            logger.info("✅ Crime mapping loaded (%d items)", len(crime_mapping))
    except Exception as e:
        logger.debug("Crime mapping not loaded: %s", e)

    try:
        if os.path.exists(PLACE_LOOKUP_PATH):
            pl = pd.read_csv(PLACE_LOOKUP_PATH)
            name_col = next((c for c in pl.columns if "place" in c.lower() or "name" in c.lower()), pl.columns[0])
            lat_col = next((c for c in pl.columns if "lat" in c.lower()), pl.columns[1] if len(pl.columns)>1 else pl.columns[0])
            lon_col = next((c for c in pl.columns if "lon" in c.lower()), pl.columns[2] if len(pl.columns)>2 else pl.columns[0])
            pl = pl.rename(columns={name_col:"Place", lat_col:"Latitude", lon_col:"Longitude"})
            pl["Latitude"] = pd.to_numeric(pl["Latitude"], errors="coerce")
            pl["Longitude"] = pd.to_numeric(pl["Longitude"], errors="coerce")
            place_lookup_df = pl.dropna(subset=["Latitude","Longitude"])
            logger.info("✅ Place lookup loaded: %d rows", len(place_lookup_df))
        else:
            logger.warning("Place lookup missing at %s", PLACE_LOOKUP_PATH)
    except Exception as e:
        logger.exception("Place lookup load failed: %s", e)

    try:
        if os.path.exists(CRIME_CSV_PATH):
            crime_df_local = pd.read_csv(CRIME_CSV_PATH)
            global crime_df
            crime_df = crime_df_local
            try:
                build_location_cache_from_df(crime_df)
            except Exception as ee:
                logger.debug("build_location_cache_from_df failed: %s", ee)
            logger.info("✅ Crime CSV loaded: %d records", len(crime_df))
        else:
            logger.warning("Crime CSV missing at %s", CRIME_CSV_PATH)
    except Exception as e:
        logger.exception("Crime CSV load failed: %s", e)

# call safe loads on startup
safe_startup_loads()

# ------------------ RISK SCORING ------------------
def compute_risk(lat: float, lon: float, hour: Optional[int] = None, dow: Optional[int] = None) -> float:
    base = 30.0
    nearest, min_d = None, float("inf")
    for v in location_cache.values():
        d = haversine_km((lat, lon), (v["lat"], v["lon"]))
        if d < min_d:
            min_d, nearest = d, v
    if nearest:
        base += min(40, nearest.get("crime_count", 0) * 0.6)
        t = str(nearest.get("most_common", "")).lower()
        if "murder" in t or "rape" in t:
            base += 25
        elif "robbery" in t or "assault" in t:
            base += 15
        elif "theft" in t:
            base += 10
    if hour is not None:
        if hour >= 22 or hour <= 4:
            base += 20
        elif 18 <= hour <= 21:
            base += 10
    if dow is not None and dow >= 5:
        base += 5
    if min_d > 1.0:
        base += min(12, min_d * 2)
    return max(0, min(100, round(base, 1)))
# ------------------ RECOMMENDATIONS ------------------
def generate_recommendations(risk_score: float, hour: Optional[int], crime_type: Optional[str]) -> List[str]:
    recs: List[str] = []
    if risk_score >= 75:
        recs += [
            "Avoid the area if possible — very high risk.",
            "If you must be there, travel in groups and avoid isolated places.",
            "Prefer trusted transport (taxi/app); avoid walking late at night.",
            "Share live location with a trusted contact and set check-in times.",
            "Charge phone and have emergency numbers at hand."
        ]
    elif risk_score >= 55:
        recs += [
            "Moderate risk — stay alert and prefer busy routes.",
            "Avoid short-cuts through poorly lit alleys.",
            "Keep valuables hidden; secure bags."
        ]
    else:
        recs += ["Area relatively safe; still be aware of surroundings."]
    if hour is not None and (hour >= 22 or hour <= 5):
        recs.append("Avoid late-night travel alone; use well-lit main roads.")
    if crime_type and "theft" in str(crime_type).lower():
        recs.append("Do not display phones or jewelry; use secured bags.")
    if crime_type and ("murder" in str(crime_type).lower() or "rape" in str(crime_type).lower()):
        recs.append("Report suspicious activity immediately and avoid the area.")
    recs.append("Emergency numbers: Police 100 | Women Helpline 1091 | Ambulance 108")
    return recs[:12]

# ------------------ Pydantic MODELS ------------------
class PredictBody(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place: Optional[str] = None
    location_name: Optional[str] = None
    time: Optional[str] = None

class RouteReq(BaseModel):
    src: Any
    dst: Any

# ------------------ ROUTES / API ------------------
@app.get("/api")
def api_root():
    return {"status": "ok", "app": APP_TITLE}

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "places": int(len(place_lookup_df)) if isinstance(place_lookup_df, pd.DataFrame) else 0,
        "crime_records": int(len(crime_df)) if crime_df is not None else 0,
        "clusters": int(len(location_cache)),
        "zones": len(zones_cache)
    }

@app.get("/api/place-lookup")
def place_lookup():
    try:
        if place_lookup_df is None or place_lookup_df.empty:
            return []
        return place_lookup_df["Place"].dropna().unique().tolist()
    except Exception:
        return []

@app.get("/api/police-locator")
def police_locator(lat: Optional[float] = Query(None), lon: Optional[float] = Query(None), limit: int = Query(5, ge=1, le=10)):
    if lat is None or lon is None:
        return {"error": "lat/lon required"}
    stations = []
    for s in BANGALORE_POLICE_STATIONS:
        d = haversine_km((lat, lon), (s["lat"], s["lon"]))
        stations.append({**s, "distance_km": round(d, 2)})
    stations = sorted(stations, key=lambda x: x["distance_km"])[:limit]
    return {"nearest": stations}

# Predict endpoint accepts lat/lon OR place OR location_name
@app.post("/api/predict")
@app.get("/api/predict")
def predict(body: PredictBody = None,
            lat: Optional[float] = Query(None),
            lon: Optional[float] = Query(None),
            place: Optional[str] = Query(None),
            location_name: Optional[str] = Query(None),
            time: Optional[str] = Query(None)):
    # allow JSON body or query params
    if body:
        lat = body.latitude if body.latitude is not None else lat
        lon = body.longitude if body.longitude is not None else lon
        place = body.place or place
        location_name = body.location_name or location_name
        time = body.time or time

    # resolve location_name via nominatim if provided and lat/lon missing
    if location_name and (lat is None or lon is None):
        glat, glon = geocode_location(location_name)
        if glat is not None:
            lat, lon = glat, glon

    # resolve place using place_lookup_df
    if place and (lat is None or lon is None) and not place_lookup_df.empty:
        row = place_lookup_df[place_lookup_df["Place"].str.lower() == place.lower()]
        if not row.empty:
            lat = float(row.iloc[0]["Latitude"]); lon = float(row.iloc[0]["Longitude"])

    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Provide coordinates or a valid place/name")

    now = datetime.now()
    hour = now.hour
    if time:
        try:
            hour = int(time.split(":")[0])
        except Exception:
            hour = now.hour
    dow = now.weekday()

    risk_score = compute_risk(lat, lon, hour, dow)
    # find nearest cluster for predicted crime
    nearest, min_d = None, float("inf")
    for v in location_cache.values():
        d = haversine_km((lat, lon), (v["lat"], v["lon"]))
        if d < min_d:
            min_d, nearest = d, v
    predicted_crime = nearest.get("most_common") if nearest else "Insufficient data"
    confidence = max(40, 100 - (min_d * 8)) if min_d != float("inf") else 50
    recs = generate_recommendations(risk_score, hour, predicted_crime)
    return {
        "latitude": lat,
        "longitude": lon,
        "risk_score": risk_score,
        "predicted_crime": predicted_crime,
        "confidence": round(confidence, 1),
        "recommendations": recs,
        "nearby_crimes": nearest.get("crime_count", 0) if nearest else 0,
        "distance_to_data_km": round(min_d, 2) if min_d != float("inf") else None
    }

@app.get("/api/heatmap")
def heatmap(limit: int = Query(800, ge=50, le=5000)):
    pts: List[Dict[str, Any]] = []
    try:
        for v in list(location_cache.values())[:limit]:
            rs = compute_risk(v["lat"], v["lon"])
            pts.append({"lat": v["lat"], "lon": v["lon"], "risk_score": rs, "crime_count": v.get("crime_count", 0), "most_common": v.get("most_common", "Unknown")})
    except Exception as e:
        logger.exception("Error building heatmap payload: %s", e)
    max_r = max((p["risk_score"] for p in pts), default=1)
    for p in pts:
        p["intensity"] = (p["risk_score"] / max_r) if max_r > 0 else 0.1
    return {"heatmap": pts, "total": len(pts)}

@app.get("/api/zones")
def zones(limit: int = Query(100, ge=10, le=500)):
    return {"zones": zones_cache[:limit]}

# Safe route endpoint: OSRM + fallback
@app.post("/api/safe-route")
def safe_route(req: RouteReq = Body(...)):
    """
    Accepts src/dst as:
     - dict with lat & lon -> {"lat":..,"lon":..}
     - place name string -> resolved via place_lookup or nominatim
    Returns OSRM geometry + sampled risks OR fallback straight-line points with risks.
    """

    def resolve_point(p: Any) -> Tuple[float, float]:
        if isinstance(p, dict) and "lat" in p and "lon" in p:
            return float(p["lat"]), float(p["lon"])
        if isinstance(p, str):
            if not place_lookup_df.empty:
                row = place_lookup_df[place_lookup_df["Place"].str.lower() == p.lower()]
                if not row.empty:
                    return float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])
            g = geocode_location(p)
            if g[0] is not None:
                return g
        raise HTTPException(status_code=400, detail=f"Cannot resolve location: {p}")

    src_lat, src_lon = resolve_point(req.src)
    dst_lat, dst_lon = resolve_point(req.dst)

    # Try OSRM (public) first
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{src_lon},{src_lat};{dst_lon},{dst_lat}"
        resp = requests.get(url, params={"overview": "full", "steps": "true", "geometries": "geojson"}, timeout=12)
        resp.raise_for_status()
        j = resp.json()
        if j.get("routes"):
            route = j["routes"][0]
            coords = route["geometry"]["coordinates"]
            step = max(1, len(coords) // 120)
            samples = []
            for i in range(0, len(coords), step):
                lon, lat = coords[i][0], coords[i][1]
                samples.append({"lat": lat, "lon": lon, "risk": compute_risk(lat, lon)})
            avg_risk = round(sum(s["risk"] for s in samples) / len(samples), 1) if samples else 50
            return {"note": "osrm", "geometry": route["geometry"], "distance_m": route.get("distance"), "duration_s": route.get("duration"), "samples": samples, "average_risk": avg_risk}
    except Exception as e:
        logger.debug("OSRM failed: %s", e)

    # fallback: straight-line sampling
    n = 40
    pts = []
    for i in range(n + 1):
        t = i / n
        lat = src_lat + t * (dst_lat - src_lat)
        lon = src_lon + t * (dst_lon - src_lon)
        pts.append({"lat": lat, "lon": lon, "risk": compute_risk(lat, lon)})
    avg_risk = round(sum(p["risk"] for p in pts) / len(pts), 1)
    dist_km = round(haversine_km((src_lat, src_lon), (dst_lat, dst_lon)), 2)
    return {"note": "fallback", "route_points": pts, "distance_km": dist_km, "average_risk": avg_risk}

# Crime trends (safe __dt insertion)
def crime_trends_safe():
    if crime_df is None:
        return {"monthly_trends": [], "crime_types": {}, "hourly": {}}
    df_local = crime_df.copy()
    try:
        if "__dt" not in df_local.columns:
            if "Date_fixed" in df_local.columns:
                df_local["__dt"] = pd.to_datetime(df_local["Date_fixed"], errors="coerce", dayfirst=True)
            elif "Date" in df_local.columns:
                df_local["__dt"] = pd.to_datetime(df_local["Date"].astype(str), errors="coerce", dayfirst=True)
    except Exception as e:
        logger.debug("crime_trends date parse err: %s", e)

    monthly = []
    try:
        if "__dt" in df_local.columns and df_local["__dt"].notna().any():
            p = df_local.dropna(subset=["__dt"])["__dt"].dt.to_period("M")
            m = p.value_counts().sort_index().reset_index()
            m.columns = ["period", "count"]
            m["date"] = m["period"].dt.to_timestamp()
            m = m.sort_values("date").tail(12)
            monthly = [{"date": r["date"].strftime("%Y-%m"), "count": int(r["count"])} for _, r in m.iterrows()]
    except Exception as e:
        logger.debug("crime_trends monthly calc failed: %s", e)

    type_col = next((c for c in df_local.columns if c.lower() in ("type", "crime", "offence")), None)
    crime_types = {}
    try:
        if type_col:
            crime_types = df_local[type_col].value_counts().to_dict()
    except Exception as e:
        logger.debug("crime_types calc failed: %s", e)

    hourly = {}
    try:
        if "__dt" in df_local.columns and df_local["__dt"].notna().any():
            hourly = df_local.dropna(subset=["__dt"])["__dt"].dt.hour.value_counts().sort_index().to_dict()
    except Exception as e:
        logger.debug("crime_trends hourly failed: %s", e)

    return {"monthly_trends": monthly, "crime_types": {str(k): int(v) for k, v in (crime_types or {}).items()}, "hourly": {int(k): int(v) for k, v in (hourly or {}).items()}}

# route wrapper so original endpoint name still works
@app.get("/api/crime-trends")
def crime_trends():
    return crime_trends_safe()

@app.get("/api/dashboard")
def dashboard():
    total = len(crime_df) if crime_df is not None else 0
    heat = heatmap().get("heatmap", [])
    top_risky = sorted(heat, key=lambda x: x["risk_score"], reverse=True)[:8]
    recent_30 = 0
    try:
        if crime_df is not None:
            df_local = crime_df.copy()
            if "__dt" not in df_local.columns:
                if "Date_fixed" in df_local.columns:
                    df_local["__dt"] = pd.to_datetime(df_local["Date_fixed"], errors="coerce", dayfirst=True)
                elif "Date" in df_local.columns:
                    df_local["__dt"] = pd.to_datetime(df_local["Date"].astype(str), errors="coerce", dayfirst=True)
            if "__dt" in df_local.columns and df_local["__dt"].notna().any():
                recent_30 = int(len(df_local[df_local["__dt"] >= (datetime.now() - timedelta(days=30))]))
    except Exception as e:
        logger.debug("dashboard recent_30 calc failed: %s", e)
    avg_risk = round(np.mean([h["risk_score"] for h in heat]) if heat else 0, 1)
    return {"total_crimes": total, "recent_30_days": recent_30, "average_risk": avg_risk, "top_risky": top_risky, "zones_count": len(zones_cache)}
# ------------------ ROOT STATIC SERVING (OPTIONAL) ------------------
# Already mounted earlier, but keeping endpoint for safety if hosting without StaticFiles:
@app.get("/", include_in_schema=False)
def root_index():
    index_path = os.path.join("frontendhtml", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"message": "Urban Vigil backend running. Place index.html under /public."})


# ------------------ ERROR HANDLER (OPTIONAL, from your original file) ------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ------------------ STARTUP EVENT ------------------
@app.on_event("startup")
def on_startup():
    logger.info("🌐 Urban Vigil backend startup complete.")


# ------------------ MAIN ENTRYPOINT ------------------
if __name__ == "__main__":
    import uvicorn
    # Re-run safe loads on local dev
    safe_startup_loads()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
