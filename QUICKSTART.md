# Solar Regatta - Quick Start Guide

## Installation

```bash
cd /Users/charlie/Solar-Regatta
pip install -r requirements.txt
```

## Running the Application

### Option 1: Gradio Web Interface (Recommended)

```bash
python3 app.py
```

Then open http://localhost:7860 in your browser.

The app has 6 tabs:
1. **📊 Data Collection** - Simulate VESC telemetry collection
2. **🤖 Model Training** - Train ML models on telemetry
3. **⚡ Speed Predictions** - Real-time speed prediction
4. **📈 Analysis Dashboard** - Visualize telemetry data
5. **🌍 World Model** ⭐ NEW - Physics-based race simulation
6. **ℹ️ About** - Project information

### Option 2: Python Scripts

#### Quick Start Demo
```bash
python3 examples/world_model_quickstart.py
```

Shows 5 beginner-friendly examples with formatted output.

#### Comprehensive Demo
```bash
python3 examples/world_model_demo.py
```

Shows 5 advanced examples and saves visualizations to `/tmp/`.

### Option 3: Python Library

```python
from solar_regatta.ml.world_model import create_default_world_model, simulate_race

# Create world model
world_model = create_default_world_model()

# Simulate a race
sun_profile = [1000.0] * 600  # Full sun for 10 minutes
trajectory, metrics = simulate_race(
    world_model,
    race_distance=500.0,
    sun_profile=sun_profile,
    strategy='optimal'
)

print(f"Time: {metrics['total_time']:.1f}s")
print(f"Speed: {metrics['avg_velocity']:.2f} m/s")
print(f"Battery: {metrics['final_soc']*100:.1f}%")
```

## World Model Features

### 🎯 What You Can Do

1. **Simulate Races**
   - Physics-based boat dynamics
   - Multiple strategies (optimal, aggressive, conservative)
   - Custom boat configurations
   - Variable weather conditions

2. **Predict Trajectories**
   - Future state prediction
   - Uncertainty quantification
   - Monte Carlo simulation
   - Confidence intervals

3. **Optimize Strategies**
   - Automated control optimization
   - Energy management
   - Speed vs battery trade-offs

4. **Compare Designs**
   - Test different parameters
   - Performance benchmarking
   - Design exploration

## Documentation

- **[WORLD_MODEL.md](WORLD_MODEL.md)** - Complete technical documentation
- **[WORLD_MODEL_SUMMARY.md](WORLD_MODEL_SUMMARY.md)** - Implementation summary
- **[CHANGELOG_WORLD_MODEL.md](CHANGELOG_WORLD_MODEL.md)** - What's new
- **[README.md](README.md)** - Project overview

## Example Code

### Basic Simulation
```python
from solar_regatta.ml.world_model import create_default_world_model, simulate_race

world_model = create_default_world_model()
sun_profile = [1000.0] * 600
trajectory, metrics = simulate_race(world_model, 500.0, sun_profile, 'optimal')
```

### Custom Boat
```python
from solar_regatta.ml.world_model import WorldModel, PhysicsParameters

params = PhysicsParameters(
    mass=40.0,              # 40kg boat
    battery_capacity=120.0, # 120Wh battery
    solar_panel_area=0.7    # 0.7m² solar panel
)
world_model = WorldModel(params)
```

### Trajectory Prediction
```python
from solar_regatta.ml.world_model import BoatState
import numpy as np

initial_state = BoatState(
    time=0.0,
    position=np.array([0.0, 0.0]),
    velocity=0.0,
    heading=0.0,
    battery_voltage=13.0,
    battery_soc=1.0,
    motor_current=0.0,
    solar_power=0.0
)

control_sequence = [(5.0, 1000.0)] * 60  # 5A for 60 seconds
trajectory = world_model.predict_trajectory(initial_state, control_sequence, dt=1.0)
```

### Visualization
```python
from solar_regatta.viz.world_model_viz import (
    plot_trajectory_2d,
    plot_state_evolution,
    plot_strategy_comparison
)

# 2D trajectory
img = plot_trajectory_2d([trajectory], labels=['My Race'])
img.save('trajectory.png')

# State evolution
img = plot_state_evolution(trajectory)
img.save('states.png')
```

## Testing

Verify installation:
```bash
python3 -c "from solar_regatta.ml.world_model import create_default_world_model; print('✓ Working!')"
```

Run basic test:
```bash
python3 -c "
from solar_regatta.ml.world_model import create_default_world_model, simulate_race
wm = create_default_world_model()
traj, met = simulate_race(wm, 500.0, [1000]*600, 'optimal')
print(f'Distance: {met[\"total_distance\"]:.0f}m, Time: {met[\"total_time\"]:.0f}s')
"
```

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Module Not Found
```bash
pip install -e .
```

### Gradio Not Available
```bash
pip install gradio==4.32.0
```

## Next Steps

1. **Try the Gradio app**: `python3 app.py`
2. **Run quickstart demo**: `python3 examples/world_model_quickstart.py`
3. **Read full docs**: Open `WORLD_MODEL.md`
4. **Explore examples**: Check `examples/` directory

## Support

- **Documentation**: See WORLD_MODEL.md
- **Examples**: Run scripts in examples/
- **Issues**: https://github.com/charlieijk/SolarRegetta/issues

---

**Ready to race? 🚤⚡☀️**

```bash
python3 app.py
```
