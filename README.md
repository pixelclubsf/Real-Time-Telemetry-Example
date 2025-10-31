# Solar Regatta - Real-Time Telemetry Analysis

A comprehensive Python package for analyzing and visualizing solar boat race telemetry data. Features interactive web dashboards, real-time performance metrics, GPS tracking, and VESC motor controller analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

🚤 **Real-Time Telemetry Visualization**
- Interactive dashboard with live data updates
- Speed, voltage, current, and efficiency charts
- GPS track visualization and path analysis

📊 **Performance Analytics**
- Speed calculations from GPS coordinates
- Battery voltage monitoring with cutoff warnings
- Motor current analysis and efficiency metrics
- Comprehensive performance statistics

🌐 **Web-Based Interface**
- Modern, responsive dashboard design
- Works on desktop and mobile devices
- No installation required for viewing data
- Real-time interactive charts with Plotly

🐍 **Python Library**
- Easy-to-use API for data analysis
- Can be used programmatically or via web interface
- Sample data generation for testing
- Matplotlib and Plotly visualizations

## Quick Start

### Installation

**Option 1: Install from GitHub**
```bash
git clone https://github.com/pixelclubsf/Real-Time-Telemetry-Example.git
cd Real-Time-Telemetry-Example
pip install -e .
```

**Option 2: Install normally (no development mode)**
```bash
pip install .
```

### Usage

#### As a Python Library

```python
from solar_regatta import (
    generate_sample_vesc_data,
    calculate_speeds,
    analyze_performance,
    plot_all_metrics
)

# Generate sample telemetry data
gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)

# Calculate speeds from GPS coordinates
speeds = calculate_speeds(gps_points, timestamps)

# Analyze performance metrics
metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

# Create visualizations
plot_all_metrics(speeds, battery_voltage, motor_current, timestamps, gps_points)
```

#### Run the Web Dashboard

```bash
python -m solar_regatta.web.app
```

Then open your browser to: **http://localhost:5001**

1. Click "Load Sample Data" to generate test data
2. View interactive charts for speed, voltage, current, and efficiency
3. Export data as JSON for further analysis

### Example Script

```python
from solar_regatta import calculate_speeds, plot_speed_vs_time
from datetime import datetime, timedelta

# Your VESC GPS data in MGRS format
gps_points = [
    "10SEG1234567890",
    "10SEG1234567891",
    "10SEG1234567892",
]

timestamps = [
    datetime(2025, 10, 30, 10, 0, 0),
    datetime(2025, 10, 30, 10, 0, 10),
    datetime(2025, 10, 30, 10, 0, 20),
]

# Calculate and plot speeds
speeds = calculate_speeds(gps_points, timestamps)
plot_speed_vs_time(speeds, timestamps, title="Solar Boat Speed Analysis")
```

Run the example:
```bash
python example_vesc_plot.py
```

## Core Modules

### `solar_regatta.core.analysis`

**Functions:**
- `calculate_speeds(gps_points, timestamps)` - Calculate speed from GPS coordinates
- `generate_sample_vesc_data(duration_seconds, interval)` - Generate realistic sample data
- `analyze_performance(speeds, battery_voltage, motor_current, timestamps)` - Calculate metrics
- `plot_speed_vs_time(speeds, timestamps, title)` - Create speed visualization
- `plot_with_coordinates(speeds, timestamps, gps_points, title)` - Plot with GPS info
- `plot_all_metrics(speeds, battery_voltage, motor_current, timestamps, gps_points)` - Dashboard visualization
- `dist(sp, ep)` - Calculate distance between MGRS coordinates

### `solar_regatta.web.app`

Flask web server providing:
- **GET /** - Main dashboard page
- **POST /api/load-sample-data** - Load sample VESC telemetry
- **GET /api/charts** - Retrieve all Plotly chart JSON
- **GET /api/metrics** - Get performance statistics
- **GET /api/export** - Export data as JSON

## API Reference

### Calculate Speeds from GPS

```python
from solar_regatta import calculate_speeds
from datetime import datetime, timedelta

# MGRS coordinates
gps_points = ["10SEG1234567890", "10SEG1234567891"]
# DateTime objects
timestamps = [datetime(2025, 10, 30, 10, 0, 0), datetime(2025, 10, 30, 10, 0, 10)]

# Returns list of speeds in m/s
speeds = calculate_speeds(gps_points, timestamps)
```

### Generate Sample Data

```python
from solar_regatta import generate_sample_vesc_data

gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)
```

**Parameters:**
- `duration_seconds` (int): Total simulation time in seconds (default: 300)
- `interval` (int): Time between GPS samples in seconds (default: 5)

**Returns:**
- `gps_points`: List of MGRS coordinates
- `timestamps`: List of datetime objects
- `speeds_raw`: List of simulated speeds
- `battery_voltage`: List of voltage readings
- `motor_current`: List of current readings

### Analyze Performance

```python
from solar_regatta import analyze_performance

metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

# metrics dict contains:
# - max_speed, min_speed, avg_speed (m/s)
# - max_voltage, min_voltage (V)
# - max_current, avg_current (A)
# - distance (m)
# - duration (seconds)
```

## Data Formats

### GPS Coordinates

The package uses **MGRS (Military Grid Reference System)** format for GPS coordinates:
- Format: `10SEG6400050000` (15 characters)
- Example: `"10SEG1234567890"`
- Converted internally to lat/lon for distance calculations

### Timestamps

Supports both:
- Python `datetime` objects
- Unix timestamps (seconds)

## Web Dashboard

The Flask web interface provides:

### Metrics Display
- Maximum, minimum, and average speeds
- Battery voltage range with low cutoff warning
- Motor current statistics
- Total distance and duration
- Start and end GPS positions

### Interactive Charts
1. **Speed vs Time** - Performance over the race
2. **Battery Voltage** - Power system monitoring
3. **Motor Current** - Power consumption analysis
4. **Efficiency Plot** - Speed vs current relationship
5. **GPS Track** - Location sequence visualization

### Data Export
Export all data as JSON for external analysis and processing.

## Architecture

```
Solar Regatta
├── Core Analysis Module
│   ├── GPS distance calculations
│   ├── Speed computations
│   ├── Performance metrics
│   └── Matplotlib visualizations
│
└── Web Dashboard
    ├── Flask backend
    ├── Plotly interactive charts
    ├── Real-time data processing
    └── JSON API endpoints
```

## Requirements

- Python 3.8 or higher
- Flask 3.0.0
- Plotly 5.17.0
- Matplotlib 3.8.0
- NumPy 1.24.3
- MGRS 1.4.6

All dependencies are automatically installed with the package.

## Installation Troubleshooting

**ModuleNotFoundError: No module named 'solar_regatta'**
```bash
pip install -e .
```

**Flask or other import errors**
```bash
pip install -r requirements.txt
```

**Template or static file issues**
```bash
pip uninstall solar-regatta
pip install -e .
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

## Project Structure

```
Real-Time-Telemetry-Example/
├── README.md                      # This file
├── INSTALLATION.md                # Installation guide
├── QUICKSTART.md                  # Quick start guide
├── FLASK_README.md                # Web dashboard documentation
├── setup.py                       # Package setup
├── pyproject.toml                 # Modern Python packaging
├── requirements.txt               # Dependencies
├── example_vesc_plot.py          # Example script
│
├── solar_regatta/                # Main package
│   ├── __init__.py               # Package exports
│   ├── core/
│   │   ├── __init__.py
│   │   └── analysis.py           # Core analysis functions
│   └── web/
│       ├── __init__.py
│       └── app.py                # Flask web server
│
├── templates/
│   └── dashboard.html            # Web UI template
│
├── static/
│   ├── css/
│   │   └── style.css             # Dashboard styles
│   └── js/
│       └── dashboard.js          # Client-side logic
```

## Usage Examples

### Basic Analysis Script

```python
from solar_regatta import (
    generate_sample_vesc_data,
    calculate_speeds,
    analyze_performance
)

# Generate sample data (300 second race)
gps, timestamps, speeds_raw, voltage, current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)

# Calculate speeds
speeds = calculate_speeds(gps, timestamps)

# Get metrics
metrics = analyze_performance(speeds, voltage, current, timestamps)

print(f"Max Speed: {metrics['max_speed']:.2f} m/s")
print(f"Avg Speed: {metrics['avg_speed']:.2f} m/s")
print(f"Distance: {metrics['distance']:.1f} m")
print(f"Duration: {metrics['duration']:.0f} seconds")
```

### Custom GPS Data

```python
from solar_regatta import calculate_speeds
from datetime import datetime, timedelta

# Your actual VESC GPS data
your_gps_points = ["10SEG...", "10SEG...", ...]
your_timestamps = [datetime(...), datetime(...), ...]

speeds = calculate_speeds(your_gps_points, your_timestamps)
```

### Data Export

```python
import json
from solar_regatta import generate_sample_vesc_data, calculate_speeds

# Generate data
gps, timestamps, speeds_raw, voltage, current = \
    generate_sample_vesc_data()

speeds = calculate_speeds(gps, timestamps)

# Export to JSON
data = {
    'gps_points': gps,
    'speeds': speeds,
    'battery_voltage': voltage,
    'motor_current': current
}

with open('telemetry_data.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions:
- GitHub Issues: [Open an issue](https://github.com/pixelclubsf/Real-Time-Telemetry-Example/issues)
- Documentation: See [FLASK_README.md](FLASK_README.md) for detailed feature docs

## Project Status

**Version:** 0.1.0 (Alpha)
**Status:** Active Development
**Last Updated:** October 2025

## Authors

- Charlie Cullen (@charlieijk)

## Acknowledgments

- Pixel Club SF for the solar boat racing initiative
- VESC (Vedder's ESC) community for motor controller telemetry data
- Plotly for interactive visualization
- Flask for web framework

---

**Get started now!**
```bash
pip install -e .
python -m solar_regatta.web.app
```

Then visit: http://localhost:5001
