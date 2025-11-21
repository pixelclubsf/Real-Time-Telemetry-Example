#!/usr/bin/env python3
"""
World Model Demonstration
=========================

This example demonstrates the advanced physics-based world model for solar boat racing.
It shows how to:
1. Create and configure a world model with custom parameters
2. Simulate race trajectories with different strategies
3. Predict with uncertainty estimation
4. Optimize control strategies
5. Learn from real telemetry data
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solar_regatta.ml.world_model import (
    WorldModel,
    BoatState,
    PhysicsParameters,
    create_default_world_model,
    simulate_race,
)

from solar_regatta.viz.world_model_viz import (
    plot_trajectory_2d,
    plot_state_evolution,
    plot_strategy_comparison,
    plot_uncertainty_bands,
)


def demo_basic_simulation():
    """Demonstrate basic race simulation."""
    print("=" * 60)
    print("DEMO 1: Basic Race Simulation")
    print("=" * 60)

    # Create world model with default parameters
    world_model = create_default_world_model()

    # Create sun intensity profile (full sun)
    sun_profile = [1000.0] * 600  # 10 minutes of full sun

    # Simulate a 500m race with different strategies
    print("\nSimulating 500m race with different strategies...")
    strategies = ['optimal', 'aggressive', 'conservative']
    trajectories = {}
    metrics = {}

    for strategy in strategies:
        traj, met = simulate_race(world_model, 500.0, sun_profile, strategy)
        trajectories[strategy] = traj
        metrics[strategy] = met

        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Distance: {met['total_distance']:.1f}m")
        print(f"  Time: {met['total_time']:.1f}s")
        print(f"  Avg Speed: {met['avg_velocity']:.2f} m/s")
        print(f"  Energy Used: {met['energy_used_wh']:.1f} Wh")
        print(f"  Efficiency: {met['efficiency_m_per_wh']:.2f} m/Wh")
        print(f"  Final SOC: {met['final_soc']*100:.1f}%")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_trajectory_2d(
        list(trajectories.values()),
        labels=list(trajectories.keys()),
        title="Race Trajectory Comparison"
    ).save("/tmp/trajectories.png")
    print("  Saved: /tmp/trajectories.png")

    plot_state_evolution(
        trajectories['optimal'],
        title="State Evolution - Optimal Strategy"
    ).save("/tmp/state_evolution.png")
    print("  Saved: /tmp/state_evolution.png")

    plot_strategy_comparison(
        trajectories,
        metrics,
        title="Strategy Performance"
    ).save("/tmp/strategy_comparison.png")
    print("  Saved: /tmp/strategy_comparison.png")


def demo_uncertainty_prediction():
    """Demonstrate uncertainty estimation with Monte Carlo simulation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Uncertainty Prediction")
    print("=" * 60)

    world_model = create_default_world_model()

    # Initial state
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

    # Control sequence (moderate constant current)
    control_sequence = [(5.0, 1000.0)] * 300  # 5A for 300 seconds

    print("\nRunning Monte Carlo simulation (50 samples)...")
    mean_trajectory, uncertainties = world_model.predict_with_uncertainty(
        initial_state,
        control_sequence,
        n_samples=50,
        dt=1.0
    )

    # Print uncertainty statistics
    print("\nUncertainty Statistics:")
    print(f"  Initial velocity uncertainty: {uncertainties[0][0]:.4f} m/s")
    print(f"  Final velocity uncertainty: {uncertainties[-1][0]:.4f} m/s")
    print(f"  Initial SOC uncertainty: {uncertainties[0][3]*100:.2f}%")
    print(f"  Final SOC uncertainty: {uncertainties[-1][3]*100:.2f}%")

    print("\nGenerating uncertainty visualization...")
    plot_uncertainty_bands(
        mean_trajectory,
        uncertainties,
        title="Prediction Uncertainty (Monte Carlo)"
    ).save("/tmp/uncertainty.png")
    print("  Saved: /tmp/uncertainty.png")


def demo_custom_boat():
    """Demonstrate custom boat configuration."""
    print("\n" + "=" * 60)
    print("DEMO 3: Custom Boat Configuration")
    print("=" * 60)

    # Create custom boat parameters
    lightweight_boat = PhysicsParameters(
        mass=35.0,  # Lighter boat
        hull_drag_coeff=0.4,  # Better hydrodynamics
        battery_capacity=80.0,  # Smaller battery
        solar_panel_area=0.6,  # Larger solar panel
        solar_efficiency=0.20,  # Better solar cells
    )

    heavyweight_boat = PhysicsParameters(
        mass=65.0,  # Heavier boat
        hull_drag_coeff=0.6,  # More drag
        battery_capacity=150.0,  # Larger battery
        solar_panel_area=0.4,  # Smaller solar panel
        solar_efficiency=0.15,  # Standard solar cells
    )

    print("\nComparing two boat designs:")
    print("\nLightweight Boat:")
    print(f"  Mass: {lightweight_boat.mass}kg")
    print(f"  Drag Coefficient: {lightweight_boat.hull_drag_coeff}")
    print(f"  Battery: {lightweight_boat.battery_capacity}Wh")
    print(f"  Solar Panel: {lightweight_boat.solar_panel_area}m² @ {lightweight_boat.solar_efficiency*100}%")

    print("\nHeavyweight Boat:")
    print(f"  Mass: {heavyweight_boat.mass}kg")
    print(f"  Drag Coefficient: {heavyweight_boat.hull_drag_coeff}")
    print(f"  Battery: {heavyweight_boat.battery_capacity}Wh")
    print(f"  Solar Panel: {heavyweight_boat.solar_panel_area}m² @ {heavyweight_boat.solar_efficiency*100}%")

    # Simulate both
    sun_profile = [1000.0] * 600
    trajectories = {}
    metrics_dict = {}

    for name, params in [("Lightweight", lightweight_boat), ("Heavyweight", heavyweight_boat)]:
        wm = WorldModel(params)
        traj, met = simulate_race(wm, 500.0, sun_profile, 'optimal')
        trajectories[name] = traj
        metrics_dict[name] = met

    print("\n\nRace Results (500m, optimal strategy):")
    for name in ["Lightweight", "Heavyweight"]:
        met = metrics_dict[name]
        print(f"\n{name}:")
        print(f"  Time: {met['total_time']:.1f}s")
        print(f"  Avg Speed: {met['avg_velocity']:.2f} m/s")
        print(f"  Max Speed: {met['max_velocity']:.2f} m/s")
        print(f"  Energy Used: {met['energy_used_wh']:.1f} Wh")
        print(f"  Efficiency: {met['efficiency_m_per_wh']:.2f} m/Wh")

    print("\nGenerating comparison visualization...")
    plot_strategy_comparison(
        trajectories,
        metrics_dict,
        title="Boat Design Comparison"
    ).save("/tmp/boat_comparison.png")
    print("  Saved: /tmp/boat_comparison.png")


def demo_control_optimization():
    """Demonstrate optimal control strategy."""
    print("\n" + "=" * 60)
    print("DEMO 4: Control Strategy Optimization")
    print("=" * 60)

    world_model = create_default_world_model()

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

    # Variable sun conditions (cloud passes overhead)
    sun_profile = [1000.0 - 500.0 * np.sin(t * 0.02)**2 for t in range(300)]

    print("\nOptimizing control strategy for 300m race...")
    print("(This may take a moment...)")

    optimal_currents = world_model.optimize_control_strategy(
        initial_state,
        target_distance=300.0,
        max_time=300.0,
        sun_intensity_profile=sun_profile,
        dt=1.0
    )

    print(f"\nOptimal control computed!")
    print(f"  Current range: {optimal_currents.min():.2f} - {optimal_currents.max():.2f} A")
    print(f"  Average current: {optimal_currents.mean():.2f} A")

    # Simulate with optimal control
    control_seq = [(curr, sun_profile[min(i, len(sun_profile)-1)])
                   for i, curr in enumerate(optimal_currents)]
    trajectory = world_model.predict_trajectory(initial_state, control_seq, dt=1.0)
    metrics = world_model.get_performance_metrics(trajectory)

    print(f"\nOptimized Race Performance:")
    print(f"  Distance: {metrics['total_distance']:.1f}m")
    print(f"  Time: {metrics['total_time']:.1f}s")
    print(f"  Avg Speed: {metrics['avg_velocity']:.2f} m/s")
    print(f"  Final SOC: {metrics['final_soc']*100:.1f}%")


def demo_variable_conditions():
    """Demonstrate simulation under varying environmental conditions."""
    print("\n" + "=" * 60)
    print("DEMO 5: Variable Environmental Conditions")
    print("=" * 60)

    world_model = create_default_world_model()

    # Different sun scenarios
    scenarios = {
        "Full Sun": [1000.0] * 600,
        "Morning Race": [400 + 600 * min(1.0, t/300) for t in range(600)],
        "Cloudy Day": [300.0] * 600,
        "Variable": [1000.0 - 400 * np.sin(t * 0.01) for t in range(600)],
    }

    print("\nSimulating 500m race under different conditions...")
    results = {}

    for scenario_name, sun_profile in scenarios.items():
        traj, met = simulate_race(world_model, 500.0, sun_profile, 'optimal')
        results[scenario_name] = met

        print(f"\n{scenario_name}:")
        print(f"  Time: {met['total_time']:.1f}s")
        print(f"  Avg Speed: {met['avg_velocity']:.2f} m/s")
        print(f"  Energy Used: {met['energy_used_wh']:.1f} Wh")
        print(f"  Final SOC: {met['final_soc']*100:.1f}%")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Solar Regatta - World Model Demonstration")
    print("="*60)

    try:
        demo_basic_simulation()
        demo_uncertainty_prediction()
        demo_custom_boat()
        demo_control_optimization()
        demo_variable_conditions()

        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        print("\nVisualization files saved to /tmp/")
        print("  - trajectories.png")
        print("  - state_evolution.png")
        print("  - strategy_comparison.png")
        print("  - uncertainty.png")
        print("  - boat_comparison.png")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
