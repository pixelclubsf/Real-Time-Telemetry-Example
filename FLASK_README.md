# Solar Boat Race Analysis - Flask Web Dashboard

A fully interactive web-based dashboard for analyzing solar boat race telemetry data with real-time VESC metrics visualization.

## Features

- **Interactive Charts** - Hover to see exact values
  - Speed vs Time
  - Battery Voltage Over Time
  - Motor Current Draw
  - Speed vs Current Efficiency Plot
  - GPS Track Points Visualization

- **Performance Metrics** - Key statistics at a glance
  - Max/Min/Average Speed
  - Distance and Duration
  - Battery Voltage Range
  - Motor Current Data

- **Data Export** - Export all data as JSON for further analysis

- **Responsive Design** - Works on desktop, tablet, and mobile

- **Sample Data Generation** - Built-in realistic VESC telemetry simulator

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/charlie/Desktop/Solar-Regatta
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:5000
   ```

3. **Click "Load Sample Data"** to generate example VESC telemetry data

## Usage

### Loading Sample Data
1. Adjust the duration and interval if needed:
   - **Duration:** Total race time in seconds (default: 300s = 5 minutes)
   - **Interval:** Time between GPS samples in seconds (default: 5s)
2. Click "Load Sample Data"
3. The dashboard will populate with:
   - Performance metrics cards
   - 5 interactive charts
   - All data points ready for analysis

### Interacting with Charts
- **Hover** over charts to see exact values
- **Zoom** by clicking and dragging
- **Pan** by holding Shift and dragging
- **Reset** by double-clicking

### Exporting Data
- Click "Export Data" to download a JSON file containing:
  - All performance metrics
  - Raw speed values
  - Battery voltage readings
  - Motor current data
  - GPS coordinates

## Project Structure

```
Solar-Regatta/
├── app.py                    # Flask application
├── solar.py                  # Core analysis functions
├── requirements.txt          # Python dependencies
├── templates/
│   └── dashboard.html        # Main dashboard page
└── static/
    ├── css/
    │   └── style.css         # Dashboard styling
    └── js/
        └── dashboard.js      # Interactive functionality
```

## API Endpoints

### `/` (GET)
Main dashboard page

### `/api/load-sample-data` (POST)
Load sample VESC data
- **Parameters:**
  - `duration`: Total race duration in seconds
  - `interval`: Time between GPS samples in seconds

### `/api/charts` (GET)
Get all chart data as JSON (Plotly format)

### `/api/metrics` (GET)
Get performance metrics

### `/api/export` (GET)
Export all data as JSON

## Customization

### Adding Your Own Data

To load your own VESC data instead of sample data, modify the `load_sample_data()` endpoint in `app.py`:

```python
# Instead of:
gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
    generate_sample_vesc_data(...)

# You can load from:
# - CSV file
# - JSON file
# - Direct API call
# - VESC tool export
```

### Modifying Chart Colors

Edit chart colors in `app.py`:
- Speed: `#2E86AB`
- Voltage: `#A23B72`
- Current: `#F18F01`
- Efficiency: `#C73E1D`

### Changing Layout

The dashboard is built with CSS Grid and Flexbox. Modify `static/css/style.css` to adjust:
- Chart grid layout
- Metric card sizes
- Colors and fonts
- Responsive breakpoints

## Troubleshooting

### Port already in use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Missing dependencies
Reinstall requirements:
```bash
pip install -r requirements.txt --force-reinstall
```

### Charts not loading
Check browser console (F12) for errors and ensure Plotly CDN is accessible

## Performance

- Handles up to 10,000+ data points smoothly
- Charts are responsive and update dynamically
- Data is cached in memory (can be extended to database)

## Future Enhancements

- Database backend for historical race data
- Multi-race comparison
- Advanced filtering and analytics
- Map integration with real GPS coordinates
- Real-time live race monitoring
- Mobile app version
- Authentication and user accounts

## License

MIT License

## Support

For issues or questions, check:
1. Browser console for JavaScript errors
2. Flask console for Python errors
3. Network tab in developer tools for API issues
