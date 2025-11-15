# VESC Tool Integration Guide

This guide explains how to use Solar Regatta with VESC motor controllers for real-time telemetry collection and data annotation.

## Overview

The Solar Regatta package now includes utilities for:
- **Real-time data collection** from VESC motor controllers
- **Telemetry annotation** for machine learning and analysis
- **Code annotation** workflows for labeling data

## Installation

### Basic Installation

```bash
pip install -e .
```

### VESC Integration (Optional)

For real-time VESC data collection, install pyserial:

```bash
pip install pyserial
```

For advanced VESC protocol support, you can also install PyVESC:

```bash
pip install git+https://github.com/LiamBindle/PyVESC.git
```

## Quick Start

### 1. Collect Data from VESC

Connect your VESC motor controller via USB and run:

```bash
python examples/collect_vesc_data.py --port /dev/ttyUSB0 --duration 300 --output race_data.json
```

**Parameters:**
- `--port`: Serial port (e.g., `/dev/ttyUSB0` on Linux, `COM3` on Windows)
- `--duration`: Collection duration in seconds
- `--interval`: Sample interval in seconds (default: 1.0)
- `--output`: Output JSON file
- `--annotation`: Optional label for the session (e.g., "practice_run_1")

### 2. Annotate Collected Data

After collection, annotate your data for ML training:

```bash
python examples/annotate_telemetry.py --input race_data.json --visualize
```

This will:
1. Show performance metrics
2. Generate visualization plots (if `--visualize` is used)
3. Prompt for interactive annotations
4. Save annotated data to `race_data_annotated.json`

### 3. Analyze and Train Models

Use the annotated data for analysis:

```python
from solar_regatta import calculate_speeds, train_speed_model, analyze_performance
from solar_regatta.vesc import VESCDataCollector

# Load annotated data
collector = VESCDataCollector()
data = collector.load_from_file('race_data_annotated.json')

# Extract telemetry
gps_points = [p.gps_position for p in data]
timestamps = [p.timestamp for p in data]
voltages = [p.battery_voltage for p in data]
currents = [p.motor_current for p in data]

# Calculate speeds
speeds = calculate_speeds(gps_points, timestamps)

# Train predictive model
model = train_speed_model(speeds, voltages, currents, timestamps)
```

## Python API

### Basic VESC Connection

```python
from solar_regatta.vesc import connect_vesc

# Connect to VESC
collector = connect_vesc(port='/dev/ttyUSB0', baudrate=115200)

# Collect data for 60 seconds
collector.start_collection(duration=60, interval=1.0)

# Get collected data
data = collector.get_data()

# Save to file
collector.save_to_file('telemetry.json')

# Disconnect
collector.disconnect()
```

### Quick Data Collection

```python
from solar_regatta.vesc import read_telemetry

# Collect and save in one call
data = read_telemetry(
    port='/dev/ttyUSB0',
    duration=60,
    interval=1.0,
    output_file='quick_test.json'
)
```

### Data Structure

Each telemetry point contains:

```python
@dataclass
class VESCTelemetryPoint:
    timestamp: datetime
    gps_position: str          # MGRS format
    speed_gps: float           # m/s from GPS
    battery_voltage: float     # V
    motor_current: float       # A
    motor_rpm: Optional[float]
    duty_cycle: Optional[float]
    amp_hours: Optional[float]
    watt_hours: Optional[float]
    temp_fet: Optional[float]
    temp_motor: Optional[float]
```

## Annotation Workflow

### Session Annotations

The annotation tool captures:
- **Session type**: race, practice, test
- **Conditions**: weather and environmental factors
- **Boat configuration**: setup notes
- **Performance labels**: max speed, issues
- **Event markers**: specific events with timestamps

### Example Annotation Session

```bash
$ python examples/annotate_telemetry.py --input race_data.json

TELEMETRY ANNOTATION
======================================================================
Session type (race/practice/test): race
Conditions (sunny/cloudy/windy/calm): sunny
Boat configuration notes: New propeller, optimized trim
Additional notes: Strong headwind in second half

Performance Labels:
Max speed achieved (m/s) [optional]: 12.5
Any battery issues? (yes/no): no
Any motor issues? (yes/no): no

Event Markers:
Event timestamp (HH:MM:SS) or 'done': 14:32:15
Event description: Start of race
Event timestamp (HH:MM:SS) or 'done': 14:35:20
Event description: Strong wind gust, reduced speed
Event timestamp (HH:MM:SS) or 'done': done

Your name/ID: Charlie
```

## Use Cases

### 1. Training Data Collection

Collect labeled datasets for ML model training:

```bash
# Collect data with annotation
python examples/collect_vesc_data.py \
  --port /dev/ttyUSB0 \
  --duration 600 \
  --annotation "optimal_conditions" \
  --output training_data_1.json

# Annotate
python examples/annotate_telemetry.py \
  --input training_data_1.json
```

### 2. Performance Analysis

Compare different configurations:

```bash
# Collect baseline
python examples/collect_vesc_data.py \
  --annotation "baseline_prop" \
  --output baseline.json

# Collect with new setup
python examples/collect_vesc_data.py \
  --annotation "new_prop" \
  --output new_prop.json

# Analyze both
python -c "
from solar_regatta.vesc import VESCDataCollector
from solar_regatta import analyze_performance, calculate_speeds

for file in ['baseline.json', 'new_prop.json']:
    collector = VESCDataCollector()
    data = collector.load_from_file(file)
    # Extract and analyze...
"
```

### 3. Real-time Monitoring

Stream data for live monitoring:

```python
from solar_regatta.vesc import VESCDataCollector

collector = VESCDataCollector(port='/dev/ttyUSB0')
collector.connect()

# Continuous collection (Ctrl+C to stop)
collector.start_collection(duration=None, interval=0.5)
```

## Troubleshooting

### Permission Denied on Serial Port

On Linux, you may need to add yourself to the dialout group:

```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

Or temporarily:

```bash
sudo chmod 666 /dev/ttyUSB0
```

### VESC Not Detected

1. Check connection: `ls /dev/ttyUSB*` (Linux) or Device Manager (Windows)
2. Verify VESC Tool can connect
3. Check baud rate matches VESC configuration
4. Ensure no other program is using the port

### Missing Dependencies

```bash
# Install all optional dependencies
pip install pyserial
pip install git+https://github.com/LiamBindle/PyVESC.git
```

## Integration with VESC Tool

### Using VESC Tool's Real-time Data

If you're using VESC Tool software, you can export data and import it:

1. In VESC Tool, enable RT data logging
2. Export the log to CSV/JSON
3. Convert to Solar Regatta format (helper script coming soon)

### Direct Protocol Implementation

For advanced users, you can implement the VESC protocol directly using the PyVESC library. See the [PyVESC documentation](https://github.com/LiamBindle/PyVESC) for details.

## Next Steps

- [Main README](README.md) - General package documentation
- [Notebooks](notebooks/) - Interactive analysis examples
- [ML Guide](notebooks/Solar_Regatta_Quickstart.ipynb) - Model training tutorial

## Contributing

If you implement additional VESC features or protocol support, please contribute back to the project!
