#!/usr/bin/env python3
"""
World Model Quick Start
=======================

A simple introduction to the Solar Regatta world model.
Perfect for getting started quickly!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solar_regatta.ml.world_model import (
    create_default_world_model,
    simulate_race,
    BoatState,
    PhysicsParameters
)
import numpy as np

print("""
╔════════════════════════════════════════════════════════════╗
║          Solar Regatta - World Model Quick Start          ║
╚════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# EXAMPLE 1: Your First Simulation
# =============================================================================
print("\n📊 EXAMPLE 1: Your First Race Simulation\n")

# Create a world model (uses realistic default parameters)
world_model = create_default_world_model()

# Define race conditions
sun_profile = [1000.0] * 600  # Full sun (1000 W/m²) for 10 minutes

# Simulate a 500 meter race
print("🚤 Simulating 500m race with optimal strategy...")
trajectory, metrics = simulate_race(
    world_model,
    race_distance=500.0,
    sun_profile=sun_profile,
    strategy='optimal'
)

# Print results
print(f"""
Results:
  ✓ Distance covered: {metrics['total_distance']:.1f} meters
  ✓ Race time: {metrics['total_time']:.1f} seconds
  ✓ Average speed: {metrics['avg_velocity']:.2f} m/s ({metrics['avg_velocity']*3.6:.1f} km/h)
  ✓ Max speed: {metrics['max_velocity']:.2f} m/s
  ✓ Energy used: {metrics['energy_used_wh']:.1f} Wh
  ✓ Efficiency: {metrics['efficiency_m_per_wh']:.2f} meters per Wh
  ✓ Final battery: {metrics['final_soc']*100:.1f}%
""")

# =============================================================================
# EXAMPLE 2: Compare Strategies
# =============================================================================
print("\n⚡ EXAMPLE 2: Comparing Racing Strategies\n")

strategies = ['optimal', 'aggressive', 'conservative']
print("Testing three strategies: optimal, aggressive, and conservative...\n")

for strategy in strategies:
    traj, met = simulate_race(world_model, 500.0, sun_profile, strategy)
    print(f"{strategy.upper():12} → Time: {met['total_time']:5.1f}s  |  "
          f"Speed: {met['avg_velocity']:.2f} m/s  |  "
          f"Battery: {met['final_soc']*100:4.1f}%  |  "
          f"Efficiency: {met['efficiency_m_per_wh']:.2f} m/Wh")

print("""
💡 Insight: Aggressive is fastest but uses more energy.
           Conservative preserves battery but is slower.
           Optimal balances both considerations.
""")

# =============================================================================
# EXAMPLE 3: Custom Boat Design
# =============================================================================
print("\n🔧 EXAMPLE 3: Testing a Custom Boat Design\n")

# Design a lightweight racing boat
racing_boat = PhysicsParameters(
    mass=40.0,                    # Lightweight: 40kg
    hull_drag_coeff=0.4,          # Sleek hull design
    battery_capacity=80.0,        # Smaller battery: 80 Wh
    solar_panel_area=0.7,         # Large solar panel: 0.7 m²
    solar_efficiency=0.22,        # High-efficiency cells: 22%
    motor_efficiency=0.88,        # Efficient motor: 88%
)

print("Designed a lightweight racing boat:")
print(f"  • Mass: {racing_boat.mass}kg (vs default 50kg)")
print(f"  • Drag coefficient: {racing_boat.hull_drag_coeff} (vs default 0.5)")
print(f"  • Solar panel: {racing_boat.solar_panel_area}m² @ {racing_boat.solar_efficiency*100}%")

# Create world model with custom parameters
from solar_regatta.ml.world_model import WorldModel
racing_model = WorldModel(racing_boat)

# Compare with default boat
print("\nRacing 500m with both designs:\n")

traj_default, met_default = simulate_race(
    world_model, 500.0, sun_profile, 'optimal'
)
traj_racing, met_racing = simulate_race(
    racing_model, 500.0, sun_profile, 'optimal'
)

print(f"Default boat:  Time={met_default['total_time']:.1f}s  |  "
      f"Speed={met_default['avg_velocity']:.2f}m/s  |  "
      f"SOC={met_default['final_soc']*100:.1f}%")
print(f"Racing boat:   Time={met_racing['total_time']:.1f}s  |  "
      f"Speed={met_racing['avg_velocity']:.2f}m/s  |  "
      f"SOC={met_racing['final_soc']*100:.1f}%")

time_improvement = met_default['total_time'] - met_racing['total_time']
speed_improvement = met_racing['avg_velocity'] - met_default['avg_velocity']

print(f"\n✓ Racing boat is {time_improvement:.1f}s faster!")
print(f"✓ Average speed improved by {speed_improvement:.2f} m/s")

# =============================================================================
# EXAMPLE 4: Variable Weather
# =============================================================================
print("\n🌤️  EXAMPLE 4: Simulating Variable Weather\n")

# Create different sun scenarios
scenarios = {
    "Perfect Sun ☀️ ": [1000.0] * 600,
    "Partly Cloudy ⛅": [700.0] * 600,
    "Morning Race 🌅": [400 + min(600, i) for i in range(600)],
    "Passing Cloud 🌥️ ": [1000 if (i < 200 or i > 400) else 300 for i in range(600)],
}

print("Testing 500m race under different conditions:\n")

for weather, sun_profile in scenarios.items():
    traj, met = simulate_race(world_model, 500.0, sun_profile, 'optimal')
    avg_sun = np.mean(sun_profile[:int(met['total_time'])])

    print(f"{weather} (avg {avg_sun:4.0f}W/m²) → "
          f"Time: {met['total_time']:5.1f}s  |  "
          f"Speed: {met['avg_velocity']:.2f}m/s  |  "
          f"Battery: {met['final_soc']*100:4.1f}%")

print("""
💡 Insight: Sun conditions significantly affect performance.
           The optimizer adapts strategy to available solar power.
""")

# =============================================================================
# EXAMPLE 5: Trajectory Prediction
# =============================================================================
print("\n🎯 EXAMPLE 5: Predicting Boat Trajectory\n")

# Start from specific initial state
initial_state = BoatState(
    time=0.0,
    position=np.array([0.0, 0.0]),
    velocity=1.5,                  # Already moving at 1.5 m/s
    heading=0.0,
    battery_voltage=13.0,
    battery_soc=0.8,              # Battery at 80%
    motor_current=5.0,
    solar_power=0.0
)

# Define control sequence (motor current, sun intensity)
control_sequence = [
    (5.0, 1000.0),  # 5A for first 10 seconds
    (5.0, 1000.0),
    (5.0, 1000.0),
    (5.0, 1000.0),
    (5.0, 1000.0),
    (5.0, 1000.0),
    (5.0, 1000.0),
    (5.0, 1000.0),
    (5.0, 1000.0),
    (5.0, 1000.0),
    (7.0, 1000.0),  # Increase to 7A for sprint
    (7.0, 1000.0),
    (7.0, 1000.0),
    (7.0, 1000.0),
    (7.0, 1000.0),
]

print("Predicting boat state over 15 seconds...")
print("Control: 5A for 10s, then 7A for 5s sprint\n")

trajectory = world_model.predict_trajectory(initial_state, control_sequence, dt=1.0)

print("Time  Velocity  Distance  Battery  Power Balance")
print("─" * 55)
for state in trajectory[::3]:  # Every 3 seconds
    power_net = state.solar_power - (state.motor_current * state.battery_voltage)
    print(f"{state.time:3.0f}s   {state.velocity:5.2f}m/s   "
          f"{np.linalg.norm(state.position):6.1f}m   "
          f"{state.battery_soc*100:5.1f}%   "
          f"{power_net:+6.1f}W")

print("""
💡 Insight: The model predicts how boat state evolves over time,
           accounting for acceleration, energy, and power balance.
""")

# =============================================================================
# Summary
# =============================================================================
print("""
╔════════════════════════════════════════════════════════════╗
║                       Summary                              ║
╚════════════════════════════════════════════════════════════╝

You've learned how to:
  ✓ Run basic race simulations
  ✓ Compare different strategies
  ✓ Design and test custom boats
  ✓ Simulate various weather conditions
  ✓ Predict detailed trajectories

Next steps:
  📚 Read WORLD_MODEL.md for complete documentation
  🎨 Check out world_model_demo.py for visualizations
  🌐 Try the Gradio app for interactive exploration

Happy racing! 🚤⚡☀️
""")
