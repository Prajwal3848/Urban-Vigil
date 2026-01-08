# Urban Vigil Pro

AI-powered safety intelligence system for Bengaluru that provides real-time crime risk assessment, safe route planning, and location-based safety recommendations.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121.1-green.svg)

## Features

- **Interactive Risk Map** - Visualize crime hotspots and risk zones across Bengaluru
- **Safe Route Planning** - Get optimized routes with OSRM integration and risk-based path suggestions
- **Real-time Dashboard** - View crime statistics, trends, and high-risk zones
- **Location Risk Checker** - Assess safety levels for any location in Bengaluru
- **Emergency Contacts** - Quick access to police, ambulance, and helpline numbers
- **Nearest Police Stations** - Find nearby police stations with contact information

## Tech Stack

**Frontend:**
- HTML5, CSS3, JavaScript
- Tailwind CSS for styling
- Leaflet.js for interactive maps
- Chart.js for data visualization
- Leaflet MarkerCluster for performance optimization

**Backend:**
- FastAPI (Python)
- Machine Learning with scikit-learn and XGBoost
- Pandas and NumPy for data processing
- OSRM API for route optimization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone https://github.com/Kethan-hs/Urban-Vigil.git
cd Urban-Vigil
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Set up the models directory
```bash
mkdir models
```

4. Place the following files in the `models/` directory:
   - `Final_model_fixed.pkl` - Trained ML model
   - `label_encoders_fixed.pkl` - Label encoders
   - `place_lookup.csv` - Location database
   - `bengaluru_dataset.csv` - Crime dataset
   - `crime_target_mapping.json` - Crime type mappings

## Running the Application

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /api/health
```

### Location Risk Assessment
```
GET /api/predict?lat=12.9716&lon=77.5946
POST /api/predict
```

Request body:
```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "location_name": "Koramangala",
  "time": "22:00"
}
```

### Safe Route Planning
```
POST /api/safe-route
```

Request body:
```json
{
  "src": {"lat": 12.9716, "lon": 77.5946},
  "dst": "MG Road"
}
```

### Other Endpoints
- `GET /api/heatmap?limit=800` - Crime heatmap data
- `GET /api/dashboard` - Dashboard statistics
- `GET /api/police-locator?lat={lat}&lon={lon}` - Nearby police stations
- `GET /api/place-lookup` - Available places
- `GET /api/crime-trends` - Crime trends and statistics

## Project Structure

```
Urban-Vigil/
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── models/                 # ML models and data files
│   ├── Final_model_fixed.pkl
│   ├── label_encoders_fixed.pkl
│   ├── place_lookup.csv
│   ├── bengaluru_dataset.csv
│   └── crime_target_mapping.json
├── frontendhtml/
│   └── index.html         # Frontend interface
└── public/                # Static files (optional)
```

## Risk Scoring

The system calculates risk scores (0-100) based on:
- Historical crime data proximity
- Crime type severity
- Time of day
- Day of week
- Distance from known incident locations

## Route Planning

- Integrates with OSRM API for realistic road-based routing
- Falls back to straight-line sampling if OSRM is unavailable
- Provides per-checkpoint risk assessments
- Calculates average route risk score

## Environment Variables

```bash
MODELS_DIR=models                    # Directory for model files
MODEL_FILE=Final_model_fixed.pkl     # ML model filename
PORT=8000                            # Server port
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is developed for educational and safety awareness purposes.

## Contact

Kethan - [GitHub Profile](https://github.com/Kethan-hs)

Project Link: [https://github.com/Kethan-hs/Urban-Vigil](https://github.com/Kethan-hs/Urban-Vigil)

## Acknowledgments

- OpenStreetMap for mapping data
- OSRM Project for routing services
- Nominatim for geocoding services
- Bengaluru Police Department for crime data awareness

---

**Note:** This application is designed to provide safety awareness and should not be the sole factor in making safety decisions. Always use common sense and follow local law enforcement guidelines.