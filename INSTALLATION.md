# Installation Guide - Solar Regatta

This guide will help you install the Solar Regatta package on your system without needing to use file paths.

## Option 1: Install from GitHub (Recommended for Users)

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- git (for cloning the repository)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pixelclubsf/Real-Time-Telemetry-Example.git
   cd Real-Time-Telemetry-Example
   ```

2. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

   Or install normally:
   ```bash
   pip install .
   ```

3. **Verify installation:**
   ```bash
   python -c "import solar_regatta; print(solar_regatta.__version__)"
   ```

## Option 2: Install with pip directly (Once Published to PyPI)

```bash
pip install solar-regatta
```

## Usage After Installation

### Using as a Python Library

Once installed, you can import the package anywhere without worrying about file paths:

```python
from solar_regatta import (
    calculate_speeds,
    generate_sample_vesc_data,
    analyze_performance,
    plot_speed_vs_time,
    plot_with_coordinates,
    plot_all_metrics
)

# Generate sample data
gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)

# Calculate speeds
speeds = calculate_speeds(gps_points, timestamps)

# Analyze performance
metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

# Create visualizations
plot_all_metrics(speeds, battery_voltage, motor_current, timestamps, gps_points)
```

### Running the Web Dashboard

1. **Start the Flask server:**
   ```bash
   python -m solar_regatta.web.app
   ```

2. **Open your browser:**
   Navigate to `http://localhost:5001`

3. **Load sample data:**
   Click "Load Sample Data" in the dashboard and interact with the visualizations

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'solar_regatta'"

**Solution:** Make sure you've installed the package:
```bash
pip install -e .
```

### Issue: "Flask not found" or other import errors

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Templates or static files not loading in the web app

**Solution:** Ensure you're running the app from the installed package, not cloning multiple times. Reinstall:
```bash
pip uninstall solar-regatta
pip install -e .
```

## Development Installation

If you're contributing to the project:

```bash
git clone https://github.com/pixelclubsf/Real-Time-Telemetry-Example.git
cd Real-Time-Telemetry-Example
pip install -e ".[dev]"  # Installs with development dependencies
```

## Uninstallation

To remove the package:

```bash
pip uninstall solar-regatta
```

## What's New?

The Solar Regatta package is now properly structured as a Python package, which means:

✅ **No path issues** - Install once, use anywhere
✅ **Import anywhere** - Use `from solar_regatta import ...` from any directory
✅ **No relative paths** - All templates and static files are found automatically
✅ **Easy distribution** - Can be installed by others via pip
✅ **Version tracking** - Clear version management

## Next Steps

See [QUICKSTART.md](QUICKSTART.md) for usage examples and [FLASK_README.md](FLASK_README.md) for detailed feature documentation.
