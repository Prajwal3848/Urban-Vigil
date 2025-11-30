# main.py - COMPLETE FIXED VERSION
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import joblib, os, math, json, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

# ==================== CONFIG ====================
APP_TITLE = "Urban Vigil Pro - AI Safety API"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_FILE = os.getenv("MODEL_FILE", "Final_model_fixed.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE)
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders_fixed.pkl")
CRIME_MAP_PATH = os.path.join(MODELS_DIR, "crime_target_mapping.json")
PLACE_LOOKUP_PATH = os.path.join(MODELS_DIR, "place_lookup.csv")
CRIME_CSV_PATH = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")

# Bangalore Police Stations with coordinates
BANGALORE_POLICE_STATIONS = [
    {"name": "Banashankari PS", "lat": 12.9250, "lon": 77.5670, "phone": "080-26771290"},
    {"name": "Koramangala PS", "lat": 12.9352, "lon": 77.6245, "phone": "080-25537466"},
    {"name": "Indiranagar PS", "lat": 12.9719, "lon": 77.6412, "phone": "080-25213009"},
    {"name": "MG Road PS", "lat": 12.9763, "lon": 77.5995, "phone": "080-22212444"},
    {"name": "Jayanagar PS", "lat": 12.9250, "lon": 77.5938, "phone": "080-26633292"},
    {"name": "Hebbal PS", "lat": 13.0350, "lon": 77.5970, "phone": "080-23637700"},
    {"name": "Whitefield PS", "lat": 12.9698, "lon": 77.7490, "phone": "080-28452905"},
    {"name": "Electronic City PS", "lat": 12.8419, "lon": 77.6605, "phone": "080-27835215"},
    {"name": "Kammanahalli PS", "lat": 13.0101, "lon": 77.6197, "phone": "080-25452611"},
    {"name": "Rajajinagar PS", "lat": 13.0136, "lon": 77.5551, "phone": "080-23580549"},
    {"name": "Yeshwanthpur PS", "lat": 13.0287, "lon": 77.5412, "phone": "080-23578450"},
    {"name": "Malleswaram PS", "lat": 13.0034, "lon": 77.5707, "phone": "080-23340223"},
    {"name": "RT Nagar PS", "lat": 13.0191, "lon": 77.5959, "phone": "080-23636244"},
    {"name": "Yelahanka PS", "lat": 13.1007, "lon": 77.5963, "phone": "080-28468933"},
    {"name": "JP Nagar PS", "lat": 12.9085, "lon": 77.5850, "phone": "080-26493399"},
    {"name": "BTM Layout PS", "lat": 12.9165, "lon": 77.6101, "phone": "080-26685336"},
    {"name": "HSR Layout PS", "lat": 12.9116, "lon": 77.6380, "phone": "080-25727731"},
    {"name": "Marathahalli PS", "lat": 12.9592, "lon": 77.6974, "phone": "080-25221744"},
    {"name": "Bellandur PS", "lat": 12.9260, "lon": 77.6748, "phone": "080-49262626"},
    {"name": "Sarjapur PS", "lat": 12.9010, "lon": 77.7280, "phone": "080-27835901"},
]

# ==================== APP ====================
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Serve static files
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("urban-vigil")

# ==================== GLOBALS ====================
model = None
encoders = {}
crime_mapping: Dict[str, Any] = {}
place_lookup_df: pd.DataFrame = pd.DataFrame()
crime_df: Optional[pd.DataFrame] = None
location_cache: Dict[str, Dict[str, Any]] = {}

# ==================== UTILITIES ====================
def haversine(coord1, coord2):
    """Calculate distance between two coordinates in km"""
    R = 6371.0
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def geocode_location(place_name: str):
    """Geocode a place name to coordinates using Nominatim"""
    try:
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{place_name}, Bangalore, India",
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "UrbanVigilPro/1.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.ok:
            data = response.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
    return None, None

def calculate_risk_score(lat, lon, hour=None, day_of_week=None):
    """Calculate risk score using ML model or heuristics"""
    base_risk = 30.0
    
    # Get nearest crime data
    nearest, min_dist = None, float("inf")
    for v in location_cache.values():
        d = haversine((lat, lon), (v['lat'], v['lon']))
        if d < min_dist:
            min_dist, nearest = d, v
    
    if nearest:
        base_risk += min(40, nearest.get('crime_count', 0) * 0.5)
        crime_type = str(nearest.get('most_common', '')).lower()
        if 'murder' in crime_type or 'rape' in crime_type:
            base_risk += 25
        elif 'robbery' in crime_type or 'assault' in crime_type:
            base_risk += 15
        elif 'theft' in crime_type:
            base_risk += 10
    
    # Time factors
    if hour is not None:
        if 22 <= hour or hour <= 4:
            base_risk += 20
        elif 18 <= hour <= 21:
            base_risk += 10
    
    if day_of_week is not None and day_of_week >= 5:
        base_risk += 5
    
    if min_dist > 1.0:
        base_risk += min(10, min_dist * 2)
    
    return max(0, min(100, round(base_risk, 1)))

def get_safety_recommendations(risk_score, hour=None, crime_type=None):
    """Generate safety recommendations based on risk"""
    recommendations = []
    
    if risk_score >= 70:
        recommendations.extend([
            "⚠️ HIGH RISK AREA - Avoid if possible",
            "🚨 Travel in groups of 3 or more",
            "📱 Share live location with family/friends",
            "🚗 Use trusted transportation services"
        ])
    elif risk_score >= 50:
        recommendations.extend([
            "⚡ MODERATE RISK - Exercise caution",
            "👥 Prefer well-lit, populated routes",
            "📞 Keep emergency contacts ready"
        ])
    else:
        recommendations.extend([
            "✅ RELATIVELY SAFE AREA",
            "👁️ Stay aware of surroundings",
            "🎧 Avoid distractions while walking"
        ])
    
    if hour is not None:
        if 22 <= hour or hour <= 5:
            recommendations.append("🌙 Late hours - Use well-lit paths only")
    
    if crime_type and 'theft' in str(crime_type).lower():
        recommendations.append("💰 Secure valuables, avoid displaying phones/jewelry")
    
    recommendations.append("🚨 Emergency: Police 100 | Women Helpline 1091")
    
    return recommendations

def build_cache(df: pd.DataFrame):
    """Build location cache from crime data"""
    global location_cache
    if df is None or df.empty:
        return
    
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    type_col = next((c for c in df.columns if c.lower() in ('type', 'crime')), None)
    
    if not all([lat_col, lon_col, type_col]):
        return
    
    df = df.copy()
    df['lat_bin'] = (pd.to_numeric(df[lat_col], errors='coerce') * 100).round() / 100
    df['lon_bin'] = (pd.to_numeric(df[lon_col], errors='coerce') * 100).round() / 100
    
    agg = df.groupby(['lat_bin', 'lon_bin']).agg({
        type_col: ['count', lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown']
    }).reset_index()
    agg.columns = ['lat', 'lon', 'crime_count', 'most_common']
    
    location_cache = {}
    for _, r in agg.iterrows():
        key = f"{r.lat:.4f},{r.lon:.4f}"
        location_cache[key] = {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "crime_count": int(r.crime_count),
            "most_common": str(r.most_common)
        }

# ==================== DATA LOADING ====================
def safe_loads():
    global model, encoders, place_lookup_df, crime_df, crime_mapping, location_cache, zones_cache
    # model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info("✅ Model loaded from %s", MODEL_PATH)
        else:
            logger.warning("Model not found at %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to load model (continuing without model): %s", e)
        model = None

    # encoders
    try:
        if os.path.exists(ENCODER_PATH):
            encoders = joblib.load(ENCODER_PATH)
            logger.info("✅ Encoders loaded")
    except Exception as e:
        logger.debug("Encoders load error (continuing): %s", e)

    # crime mapping json (optional)
    try:
        if os.path.exists(CRIME_MAP_PATH):
            with open(CRIME_MAP_PATH, "r", encoding="utf-8") as fh:
                crime_mapping = json.load(fh)
            logger.info("✅ Crime mapping loaded (%d items)", len(crime_mapping))
    except Exception as e:
        logger.debug("Crime mapping load failed: %s", e)

    # place lookup CSV
    try:
        if os.path.exists(PLACE_LOOKUP_PATH):
            pl = pd.read_csv(PLACE_LOOKUP_PATH)
            name_col = next((c for c in pl.columns if "place" in c.lower() or "name" in c.lower()), pl.columns[0])
            lat_col = next((c for c in pl.columns if "lat" in c.lower()), pl.columns[1] if len(pl.columns) > 1 else pl.columns[0])
            lon_col = next((c for c in pl.columns if "lon" in c.lower()), pl.columns[2] if len(pl.columns) > 2 else pl.columns[0])
            pl = pl.rename(columns={name_col: "Place", lat_col: "Latitude", lon_col: "Longitude"})
            pl["Latitude"] = pd.to_numeric(pl["Latitude"], errors="coerce")
            pl["Longitude"] = pd.to_numeric(pl["Longitude"], errors="coerce")
            place_lookup_df = pl.dropna(subset=["Latitude", "Longitude"])
            logger.info("✅ Place lookup loaded: %d rows", len(place_lookup_df))
        else:
            logger.warning("Place lookup file missing at %s", PLACE_LOOKUP_PATH)
    except Exception as e:
        logger.exception("Failed to load place_lookup.csv: %s", e)

    # crime CSV (optional)
    try:
        if os.path.exists(CRIME_CSV_PATH):
            crime_df = pd.read_csv(CRIME_CSV_PATH)
            # rebuild caches (this uses your original cluster logic)
            try:
                build_location_cache_from_df(crime_df)
            except Exception as e:
                logger.debug("build_location_cache_from_df failed: %s", e)
            logger.info("✅ Crime CSV loaded: %d records", len(crime_df))
        else:
            logger.warning("Crime CSV not found at %s", CRIME_CSV_PATH)
    except Exception as e:
        logger.exception("Failed to load crime CSV: %s", e)

# call it once during startup (replace previous direct load call)
safe_loads()


@app.on_event("startup")
def startup_event():
    safe_loads()

# ==================== MODELS ====================
class PredictRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place: Optional[str] = None
    location_name: Optional[str] = None  # NEW: Accept plain text location
    time: Optional[str] = None

class RouteRequest(BaseModel):
    src: Any
    dst: Any

# ==================== ENDPOINTS ====================
@app.get("/api")
def api_root():
    return {
        "status": "online",
        "app": APP_TITLE,
        "version": "2.0",
        "endpoints": [
            "/api/health",
            "/api/place-lookup", 
            "/api/predict",
            "/api/heatmap",
            "/api/safe-route",
            "/api/dashboard",
            "/api/crime-trends",
            "/api/police-locator"
        ]
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "places": len(place_lookup_df),
        "crime_records": len(crime_df) if crime_df is not None else 0,
        "cached_clusters": len(location_cache),
        "police_stations": len(BANGALORE_POLICE_STATIONS),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/place-lookup")
def place_lookup():
    """Get list of known places"""
    try:
        if place_lookup_df is None or place_lookup_df.empty:
            return []
        return place_lookup_df["Place"].dropna().unique().tolist()
    except Exception as e:
        logger.exception("place_lookup error")
        return []

@app.get("/api/police-locator")
@app.post("/api/police-locator")
def police_locator(lat: Optional[float] = None, lon: Optional[float] = None):
    """Find nearest police stations - NEW FEATURE"""
    if lat is None or lon is None:
        return {"error": "Latitude and longitude required"}
    
    # Calculate distances
    stations_with_distance = []
    for station in BANGALORE_POLICE_STATIONS:
        distance = haversine((lat, lon), (station['lat'], station['lon']))
        stations_with_distance.append({
            **station,
            "distance_km": round(distance, 2)
        })
    
    # Sort by distance and get top 5
    stations_with_distance.sort(key=lambda x: x['distance_km'])
    top_5 = stations_with_distance[:5]
    
    return {
        "nearest_stations": top_5,
        "total_stations": len(BANGALORE_POLICE_STATIONS),
        "user_location": {"lat": lat, "lon": lon}
    }

@app.get("/api/predict")
@app.post("/api/predict")
def predict(req: Optional[PredictRequest] = None, 
            place: Optional[str] = Query(None),
            location_name: Optional[str] = Query(None),  # NEW
            lat: Optional[float] = Query(None),
            lon: Optional[float] = Query(None),
            time: Optional[str] = Query(None)):
    """Predict crime risk - ENHANCED"""
    
    # Handle both POST and GET
    if req:
        place = req.place or req.location_name
        location_name = req.location_name
        lat = req.latitude
        lon = req.longitude
        time = req.time
    
    # NEW: Handle plain text location
    if location_name and not lat and not lon:
        logger.info(f"Geocoding location: {location_name}")
        lat, lon = geocode_location(location_name)
        if not lat:
            # Check if location is in Bangalore
            return {
                "error": "location_not_found",
                "message": f"'{location_name}' not found in Bengaluru. We're expanding to more areas soon!",
                "suggestion": "Please try a well-known area like Koramangala, Indiranagar, or Whitefield"
            }
    
    # Resolve place to coordinates
    if place and not lat:
        row = place_lookup_df[place_lookup_df["Place"].str.lower() == place.lower()]
        if not row.empty:
            lat = float(row.iloc[0]["Latitude"])
            lon = float(row.iloc[0]["Longitude"])
    
    if not lat or not lon:
        raise HTTPException(400, "Please provide coordinates or a valid location name")
    
    # Parse time
    now = datetime.now()
    hour = now.hour
    if time:
        try:
            hour = datetime.strptime(time.strip(), "%H:%M").hour
        except:
            try:
                hour = int(time.strip().split(":")[0])
            except:
                pass
    
    day_of_week = now.weekday()
    
    # Calculate risk
    risk_score = calculate_risk_score(lat, lon, hour, day_of_week)
    
    # Get nearest crime data
    nearest, distance = None, None
    min_d = float("inf")
    for v in location_cache.values():
        d = haversine((lat, lon), (v['lat'], v['lon']))
        if d < min_d:
            min_d, nearest = d, v
    distance = min_d if min_d != float("inf") else None
    
    predicted_crime = nearest.get('most_common', 'Unknown') if nearest else 'Insufficient Data'
    confidence = max(40, 100 - (distance * 10)) if distance else 50
    
    # Get recommendations
    recommendations = get_safety_recommendations(risk_score, hour, predicted_crime)
    
    return {
        "latitude": lat,
        "longitude": lon,
        "predicted_crime": predicted_crime,
        "risk_score": round(risk_score, 1),
        "confidence": round(confidence, 1),
        "recommendations": recommendations,
        "nearby_crimes": nearest['crime_count'] if nearest else 0,
        "hour": hour,
        "day_of_week": day_of_week,
        "distance_to_data_km": round(distance, 2) if distance else None
    }

@app.get("/api/heatmap")
def heatmap(limit: int = Query(500, ge=1, le=2000)):
    """Get crime heatmap data"""
    if not location_cache:
        return {"heatmap": [], "total": 0}
    
    data = []
    for loc in list(location_cache.values())[:limit]:
        risk = calculate_risk_score(loc['lat'], loc['lon'])
        data.append({
            "lat": loc['lat'],
            "lon": loc['lon'],
            "risk_score": risk,
            "crime_count": loc['crime_count'],
            "most_common": loc['most_common']
        })
    
    data.sort(key=lambda x: x['risk_score'], reverse=True)
    return {"heatmap": data, "total": len(data)}

@app.post("/api/safe-route")
def safe_route(req: RouteReq = Body(...)):
    """
    Calculate safe route with OSRM if available; otherwise fallback to straight-line sampling.
    Accepts src and dst as either {"lat":..,"lon":..} OR place string OR location_name.
    """

    def resolve_point(p: Any) -> Tuple[float, float]:
        # Accept dicts with lat/lon or string place names (lookup or geocode)
        if isinstance(p, dict) and "lat" in p and "lon" in p:
            return float(p["lat"]), float(p["lon"])
        if isinstance(p, str):
            # try place lookup first
            if not place_lookup_df.empty:
                row = place_lookup_df[place_lookup_df["Place"].str.lower() == p.lower()]
                if not row.empty:
                    return float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])
            # geocode fallback
            g = geocode_location(p)
            if g[0] is not None:
                return g
        raise HTTPException(status_code=400, detail=f"Cannot resolve location: {p}")

    src_lat, src_lon = resolve_point(req.src)
    dst_lat, dst_lon = resolve_point(req.dst)

    # Try OSRM route first
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{src_lon},{src_lat};{dst_lon},{dst_lat}"
        resp = requests.get(url, params={"overview": "full", "steps": "true", "geometries": "geojson"}, timeout=12)
        resp.raise_for_status()
        j = resp.json()
        if j.get("routes"):
            route = j["routes"][0]
            coords = route["geometry"]["coordinates"]
            # sample points to keep payload small
            step = max(1, len(coords) // 120)
            samples = []
            for i in range(0, len(coords), step):
                lon, lat = coords[i][0], coords[i][1]
                samples.append({"lat": lat, "lon": lon, "risk": calculate_risk_score(lat, lon)})
            avg_risk = round(sum(s["risk"] for s in samples) / len(samples), 1) if samples else 50
            return {"note": "osrm", "geometry": route["geometry"], "distance_m": route.get("distance"), "duration_s": route.get("duration"), "samples": samples, "average_risk": avg_risk}
    except Exception as e:
        logger.debug("OSRM route failed or timed out: %s", e)

    # Fallback straight-line interpolation
    n = 40
    pts = []
    for i in range(n + 1):
        t = i / n
        lat = src_lat + t * (dst_lat - src_lat)
        lon = src_lon + t * (dst_lon - src_lon)
        pts.append({"lat": lat, "lon": lon, "risk": calculate_risk_score(lat, lon)})
    avg_risk = round(sum(p["risk"] for p in pts) / len(pts), 1)
    dist_km = round(haversine((src_lat, src_lon), (dst_lat, dst_lon)), 2)
    return {"note": "fallback", "route_points": pts, "distance_km": dist_km, "average_risk": avg_risk}


@app.get("/api/dashboard")
def dashboard():
    """Dashboard stats"""
    total = len(crime_df) if crime_df is not None else 0
    heat = heatmap(limit=100)['heatmap']
    top_risky = heat[:5]
    
    recent_30d = 0
    if crime_df is not None:
        date_cols = [c for c in crime_df.columns if 'date' in c.lower()]
        if date_cols:
            try:
                df = crime_df.copy()
                if 'Date_fixed' in df.columns:
                    df['__dt'] = pd.to_datetime(df['Date_fixed'], errors='coerce')
                elif 'Date' in df.columns:
                    df['__dt'] = pd.to_datetime(df['Date'], errors='coerce')
                if '__dt' in df.columns and df['__dt'].notna().any():
                    recent_30d = len(df[df['__dt'] >= (datetime.now() - timedelta(days=30))])
            except:
                pass
    
    avg_risk = round(np.mean([h['risk_score'] for h in heat]), 1) if heat else 0
    
    return {
        "total_crimes": total,
        "recent_crimes_30d": recent_30d,
        "top_risky": top_risky,
        "average_risk": avg_risk,
        "cached_clusters": len(location_cache)
    }

@app.get("/api/crime-trends")
def crime_trends():
    """
    Returns monthly trends (last 12 months), crime type counts and hourly distribution.
    Guard against repeated insertion of __dt — only create __dt if missing.
    """
    if crime_df is None:
        return {"monthly_trends": [], "crime_types": {}, "hourly": {}}
    df = crime_df.copy()

    # --- SAFE: only create __dt if it's not already present ---
    try:
        if "__dt" not in df.columns:
            if "Date_fixed" in df.columns:
                # parse with dayfirst=True to match dd-mm-yyyy formats
                df["__dt"] = pd.to_datetime(df["Date_fixed"], errors="coerce", dayfirst=True)
            elif "Date" in df.columns:
                df["__dt"] = pd.to_datetime(df["Date"].astype(str), errors="coerce", dayfirst=True)
    except Exception as e:
        logger.exception("Failed to parse dates for crime_trends: %s", e)

    # monthly trends (last 12 months)
    monthly = []
    try:
        if "__dt" in df.columns and df["__dt"].notna().any():
            p = df.dropna(subset=["__dt"])["__dt"].dt.to_period("M")
            m = p.value_counts().sort_index().reset_index()
            m.columns = ["period", "count"]
            m["date"] = m["period"].dt.to_timestamp()
            m = m.sort_values("date").tail(12)
            monthly = [{"date": r["date"].strftime("%Y-%m"), "count": int(r["count"])} for _, r in m.iterrows()]
    except Exception as e:
        logger.exception("Failed building monthly trends: %s", e)

    # crime types
    type_col = next((c for c in df.columns if c.lower() in ("type", "crime", "offence")), None)
    crime_types = {}
    try:
        if type_col is not None:
            crime_types = df[type_col].value_counts().to_dict()
    except Exception as e:
        logger.debug("crime_types calc failed: %s", e)

    # hourly distribution
    hourly = {}
    try:
        if "__dt" in df.columns and df["__dt"].notna().any():
            hourly = df.dropna(subset=["__dt"])["__dt"].dt.hour.value_counts().sort_index().to_dict()
    except Exception as e:
        logger.debug("hourly calc failed: %s", e)

    # ensure serializable ints
    crime_types = {str(k): int(v) for k, v in (crime_types or {}).items()}
    hourly = {int(k): int(v) for k, v in (hourly or {}).items()}

    return {"monthly_trends": monthly, "crime_types": crime_types, "hourly": hourly}


if __name__ == "__main__":
    import uvicorn
    safe_loads()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))