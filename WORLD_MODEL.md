# World Model for Solar Boat Racing

## Overview

The Solar Regatta World Model is an advanced physics-based simulation system that models the complete dynamics of solar boat racing. Unlike simple regression models that only predict speed from telemetry, the world model understands the underlying physics and can:

- **Predict future states** over time horizons from seconds to minutes
- **Estimate uncertainty** in predictions using Monte Carlo methods
- **Optimize strategies** for energy management and race planning
- **Compare designs** by simulating different boat configurations
- **Learn from data** by fitting corrections to real telemetry

## Key Features

### 1. Physics-Based Dynamics

The world model simulates real physical processes:

- **Hydrodynamics**: Water drag proportional to velocity squared
- **Propulsion**: Motor/propeller efficiency and thrust generation
- **Energy**: Battery state of charge, internal resistance, power flow
- **Solar**: Panel area, efficiency, and sun intensity
- **Mass/Inertia**: Boat mass affects acceleration

### 2. Predictive Capabilities

Given an initial state and control inputs, predict:
- Boat velocity and position over time
- Battery voltage and state of charge
- Energy consumption and solar generation
- Complete trajectory with full state history

### 3. Uncertainty Quantification

Monte Carlo simulation provides:
- Confidence intervals on predictions
- Sensitivity to parameter uncertainty
- Risk assessment for race strategies

### 4. Strategy Optimization

Automated optimization finds:
- Optimal motor current profiles
- Energy management strategies
- Trade-offs between speed and battery life

## Architecture

```
WorldModel
├── PhysicsParameters     - Boat/environment configuration
├── BoatDynamicsModel    - Core physics simulation
│   ├── Drag calculation
│   ├── Thrust generation
│   ├── Energy balance
│   └── State updates
├── BoatState            - Complete state at one moment
└── Methods
    ├── predict_trajectory()          - Deterministic prediction
    ├── predict_with_uncertainty()   - Probabilistic prediction
    ├── optimize_control_strategy()  - Find optimal controls
    └── learn_from_telemetry()       - Fit to real data
```

## Usage Examples

### Basic Simulation

```python
from solar_regatta.ml.world_model import create_default_world_model, simulate_race
import numpy as np

# Create world model with default parameters
world_model = create_default_world_model()

# Define sun conditions (W/m²)
sun_profile = [1000.0] * 600  # Full sun for 10 minutes

# Simulate a 500m race with optimal strategy
trajectory, metrics = simulate_race(
    world_model,
    race_distance=500.0,
    sun_profile=sun_profile,
    strategy='optimal'  # or 'aggressive', 'conservative'
)

print(f"Race time: {metrics['total_time']:.1f}s")
print(f"Average speed: {metrics['avg_velocity']:.2f} m/s")
print(f"Energy used: {metrics['energy_used_wh']:.1f} Wh")
print(f"Efficiency: {metrics['efficiency_m_per_wh']:.2f} m/Wh")
```

### Custom Boat Configuration

```python
from solar_regatta.ml.world_model import WorldModel, PhysicsParameters

# Define custom boat parameters
params = PhysicsParameters(
    mass=45.0,                    # 45kg boat
    hull_drag_coeff=0.45,         # Low drag hull
    battery_capacity=120.0,       # 120 Wh battery
    solar_panel_area=0.6,         # 0.6 m² panel
    solar_efficiency=0.20,        # 20% efficient cells
    motor_efficiency=0.90,        # 90% motor efficiency
    prop_efficiency=0.70          # 70% propeller efficiency
)

# Create world model with custom parameters
world_model = WorldModel(params)

# Now use it for simulations
```

### Trajectory Prediction

```python
from solar_regatta.ml.world_model import BoatState
import numpy as np

# Define initial state
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

# Define control sequence: (motor_current, sun_intensity) for each second
control_sequence = [
    (5.0, 1000.0),  # 5A motor current, full sun
    (5.0, 1000.0),
    (6.0, 1000.0),
    # ... more time steps
]

# Predict trajectory
trajectory = world_model.predict_trajectory(
    initial_state,
    control_sequence,
    dt=1.0  # 1 second time steps
)

# Access predicted states
for state in trajectory:
    print(f"t={state.time}s: v={state.velocity:.2f} m/s, SOC={state.battery_soc*100:.1f}%")
```

### Uncertainty Estimation

```python
# Monte Carlo prediction with uncertainty
mean_trajectory, uncertainties = world_model.predict_with_uncertainty(
    initial_state,
    control_sequence,
    n_samples=100,  # Number of Monte Carlo samples
    dt=1.0
)

# Uncertainties contain [velocity_std, x_std, y_std, soc_std] at each time step
for state, unc in zip(mean_trajectory, uncertainties):
    velocity_std = unc[0]
    soc_std = unc[3]
    print(f"t={state.time}s: v={state.velocity:.2f}±{velocity_std:.2f} m/s")
```

### Control Optimization

```python
# Find optimal motor current profile
optimal_currents = world_model.optimize_control_strategy(
    initial_state,
    target_distance=500.0,      # 500m race
    max_time=300.0,             # 5 minute limit
    sun_intensity_profile=[1000.0] * 300,
    dt=1.0
)

# optimal_currents is a numpy array of motor currents (A) for each second
print(f"Optimal current range: {optimal_currents.min():.1f} - {optimal_currents.max():.1f} A")
```

### Learning from Telemetry

```python
# Fit world model to real telemetry data
telemetry_states = [...]  # List of BoatState objects from real data
telemetry_controls = [...]  # Corresponding control inputs

# Learn corrections from data
world_model.learn_from_telemetry(telemetry_states, telemetry_controls)

# Now predictions will include learned corrections
```

## Visualization

The package includes comprehensive visualization tools:

```python
from solar_regatta.viz.world_model_viz import (
    plot_trajectory_2d,
    plot_state_evolution,
    plot_strategy_comparison,
    plot_uncertainty_bands
)

# 2D trajectory plot
img = plot_trajectory_2d(
    [trajectory1, trajectory2],
    labels=['Strategy A', 'Strategy B'],
    title="Race Trajectories"
)
img.save('trajectories.png')

# Complete state evolution over time
img = plot_state_evolution(trajectory, title="State Evolution")
img.save('states.png')

# Compare multiple strategies
img = plot_strategy_comparison(
    trajectories_dict={'optimal': traj1, 'aggressive': traj2},
    metrics_dict={'optimal': metrics1, 'aggressive': metrics2}
)
img.save('comparison.png')

# Uncertainty bands
img = plot_uncertainty_bands(mean_trajectory, uncertainties)
img.save('uncertainty.png')
```

## Physics Models

### Drag Force

```
F_drag = 0.5 * ρ * C_d * A * v²

where:
  ρ = water density (kg/m³)
  C_d = hull drag coefficient
  A = frontal area (m²)
  v = velocity (m/s)
```

### Thrust Force

```
F_thrust = (P_motor * η_motor * η_prop) / v

where:
  P_motor = motor power (W)
  η_motor = motor efficiency
  η_prop = propeller efficiency
  v = velocity (m/s)
```

### Acceleration

```
a = (F_thrust - F_drag) / m

where:
  m = boat mass (kg)
```

### Battery Dynamics

```
SOC_new = SOC + (P_net * Δt) / (3600 * C_battery)

where:
  P_net = P_solar - P_motor (W)
  Δt = time step (s)
  C_battery = battery capacity (Wh)
```

### Solar Power

```
P_solar = A_panel * η_solar * I_sun

where:
  A_panel = panel area (m²)
  η_solar = solar efficiency
  I_sun = solar irradiance (W/m²)
```

## Default Parameters

```python
PhysicsParameters(
    # Boat
    mass=50.0,                  # kg
    hull_drag_coeff=0.5,
    frontal_area=0.3,           # m²

    # Motor/Propulsion
    motor_efficiency=0.85,
    prop_efficiency=0.65,
    max_motor_power=200.0,      # W

    # Battery
    battery_capacity=100.0,     # Wh
    battery_resistance=0.1,     # Ohms
    nominal_voltage=13.0,       # V

    # Solar
    solar_panel_area=0.5,       # m²
    solar_efficiency=0.18,      # 18%

    # Environment
    water_density=1000.0,       # kg/m³
    air_density=1.225,          # kg/m³
)
```

## Strategies

The world model includes three built-in racing strategies:

### Optimal Strategy
Uses numerical optimization to find the best motor current profile that:
- Maximizes speed
- Manages energy to avoid battery depletion
- Adapts to varying sun conditions

### Aggressive Strategy
- High constant motor current (8A)
- Prioritizes speed over energy efficiency
- Risk of battery depletion in long races

### Conservative Strategy
- Low constant motor current (3A)
- Prioritizes battery preservation
- Slower but more reliable

## Integration with Gradio App

The world model is integrated into the main Gradio dashboard at [app.py](app.py) as the "🌍 World Model" tab. Features include:

- Interactive race simulation
- Multiple strategy comparison
- Custom boat configuration
- Sun condition profiles
- Real-time visualization
- Performance metrics tables

## Performance Considerations

- **Standard simulation**: ~1ms per time step
- **Uncertainty estimation**: ~100ms for 100 Monte Carlo samples
- **Optimization**: ~5-30 seconds depending on race duration
- **Real-time capable**: Yes, for prediction and visualization

## Validation

The world model can be validated against real telemetry by:

1. Running simulation with recorded control inputs
2. Comparing predicted vs. actual states
3. Using `learn_from_telemetry()` to fit corrections
4. Re-validating with cross-validation

## Future Enhancements

Potential improvements include:

- **Machine learning corrections**: Neural network residual models
- **Weather integration**: Wind, waves, current prediction
- **Multi-boat racing**: Interaction dynamics
- **Route optimization**: Optimal path planning
- **Real-time MPC**: Model predictive control for autonomous racing
- **Hardware-in-the-loop**: Integration with real VESC controllers

## References

- Boat hydrodynamics: Principles of Naval Architecture
- Solar panel modeling: PV system design standards
- Battery dynamics: Lithium-ion battery models
- Optimization: SciPy optimization documentation

## Examples

See [examples/world_model_demo.py](examples/world_model_demo.py) for comprehensive demonstrations of all features.

## Support

For questions or issues with the world model:
- GitHub Issues: [Open an issue](https://github.com/charlieijk/SolarRegetta/issues)
- Documentation: This file and code docstrings
- Examples: `examples/world_model_demo.py`
