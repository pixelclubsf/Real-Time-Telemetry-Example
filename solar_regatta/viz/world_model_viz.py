"""
Visualization tools for World Model predictions and analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import List, Optional, Tuple
import io
from PIL import Image

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_trajectory_2d(trajectories: List, labels: Optional[List[str]] = None,
                       uncertainties: Optional[List] = None,
                       title: str = "Boat Trajectory") -> Image.Image:
    """
    Plot 2D trajectory with optional uncertainty bounds.

    Args:
        trajectories: List of trajectory lists (each trajectory is list of BoatStates)
        labels: Optional labels for each trajectory
        uncertainties: Optional uncertainty bounds for each trajectory
        title: Plot title

    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    for idx, trajectory in enumerate(trajectories):
        positions = np.array([state.position for state in trajectory])
        label = labels[idx] if labels else f"Trajectory {idx+1}"

        # Plot main trajectory
        ax.plot(positions[:, 0], positions[:, 1],
               marker='o', markersize=3, label=label,
               color=colors[idx], linewidth=2)

        # Plot start and end points
        ax.scatter(positions[0, 0], positions[0, 1],
                  s=200, marker='o', color='green',
                  edgecolor='black', linewidth=2, zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1],
                  s=200, marker='s', color='red',
                  edgecolor='black', linewidth=2, zorder=5)

        # Add uncertainty ellipses if provided
        if uncertainties and idx < len(uncertainties):
            unc = uncertainties[idx]
            # Plot every 10th uncertainty ellipse to avoid clutter
            for i in range(0, len(positions), 10):
                if i < len(unc):
                    # Position uncertainty is in indices 1, 2
                    ellipse = Ellipse(
                        xy=positions[i],
                        width=unc[i][1] * 4,  # 2 sigma
                        height=unc[i][2] * 4,
                        alpha=0.2,
                        color=colors[idx],
                        linewidth=0
                    )
                    ax.add_patch(ellipse)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def plot_state_evolution(trajectory: List, title: str = "State Evolution") -> Image.Image:
    """
    Plot evolution of key state variables over time.

    Args:
        trajectory: List of BoatState objects
        title: Plot title

    Returns:
        PIL Image
    """
    times = [state.time for state in trajectory]
    velocities = [state.velocity for state in trajectory]
    voltages = [state.battery_voltage for state in trajectory]
    socs = [state.battery_soc * 100 for state in trajectory]  # Convert to percentage
    currents = [state.motor_current for state in trajectory]
    solar_powers = [state.solar_power for state in trajectory]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Velocity
    axes[0, 0].plot(times, velocities, linewidth=2, color='#2196F3')
    axes[0, 0].set_ylabel('Velocity (m/s)', fontsize=11)
    axes[0, 0].set_title('Boat Speed', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].fill_between(times, velocities, alpha=0.3, color='#2196F3')

    # Battery Voltage
    axes[0, 1].plot(times, voltages, linewidth=2, color='#FF9800')
    axes[0, 1].set_ylabel('Voltage (V)', fontsize=11)
    axes[0, 1].set_title('Battery Voltage', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=11.0, color='r', linestyle='--', label='Cutoff (11V)')
    axes[0, 1].legend()

    # State of Charge
    axes[1, 0].plot(times, socs, linewidth=2, color='#4CAF50')
    axes[1, 0].set_ylabel('SOC (%)', fontsize=11)
    axes[1, 0].set_title('Battery State of Charge', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(times, socs, alpha=0.3, color='#4CAF50')
    axes[1, 0].axhline(y=20, color='r', linestyle='--', label='Low (20%)')
    axes[1, 0].legend()

    # Motor Current
    axes[1, 1].plot(times, currents, linewidth=2, color='#F44336')
    axes[1, 1].set_ylabel('Current (A)', fontsize=11)
    axes[1, 1].set_title('Motor Current', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].fill_between(times, currents, alpha=0.3, color='#F44336')

    # Solar Power
    axes[2, 0].plot(times, solar_powers, linewidth=2, color='#FFC107')
    axes[2, 0].set_xlabel('Time (s)', fontsize=11)
    axes[2, 0].set_ylabel('Solar Power (W)', fontsize=11)
    axes[2, 0].set_title('Solar Panel Output', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].fill_between(times, solar_powers, alpha=0.3, color='#FFC107')

    # Energy Balance
    power_consumption = [c * v for c, v in zip(currents, voltages)]
    net_power = [s - p for s, p in zip(solar_powers, power_consumption)]
    axes[2, 1].plot(times, power_consumption, linewidth=2, color='#F44336', label='Consumption')
    axes[2, 1].plot(times, solar_powers, linewidth=2, color='#4CAF50', label='Solar')
    axes[2, 1].plot(times, net_power, linewidth=2, color='#2196F3', linestyle='--', label='Net')
    axes[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2, 1].set_xlabel('Time (s)', fontsize=11)
    axes[2, 1].set_ylabel('Power (W)', fontsize=11)
    axes[2, 1].set_title('Energy Balance', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def plot_strategy_comparison(trajectories_dict: dict,
                            metrics_dict: dict,
                            title: str = "Strategy Comparison") -> Image.Image:
    """
    Compare multiple racing strategies.

    Args:
        trajectories_dict: Dict of {strategy_name: trajectory}
        metrics_dict: Dict of {strategy_name: metrics}
        title: Plot title

    Returns:
        PIL Image
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    strategies = list(trajectories_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    # 1. Velocity comparison
    for idx, strategy in enumerate(strategies):
        traj = trajectories_dict[strategy]
        times = [s.time for s in traj]
        velocities = [s.velocity for s in traj]
        axes[0, 0].plot(times, velocities, label=strategy,
                       linewidth=2, color=colors[idx])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title('Speed Profiles')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Battery SOC comparison
    for idx, strategy in enumerate(strategies):
        traj = trajectories_dict[strategy]
        times = [s.time for s in traj]
        socs = [s.battery_soc * 100 for s in traj]
        axes[0, 1].plot(times, socs, label=strategy,
                       linewidth=2, color=colors[idx])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('SOC (%)')
    axes[0, 1].set_title('Battery Usage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distance vs Time
    for idx, strategy in enumerate(strategies):
        traj = trajectories_dict[strategy]
        times = [s.time for s in traj]
        distances = [np.linalg.norm(s.position) for s in traj]
        axes[1, 0].plot(times, distances, label=strategy,
                       linewidth=2, color=colors[idx])
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Distance (m)')
    axes[1, 0].set_title('Distance Covered')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Performance metrics bar chart
    metric_names = ['avg_velocity', 'efficiency_m_per_wh', 'final_soc']
    metric_labels = ['Avg Speed\n(m/s)', 'Efficiency\n(m/Wh)', 'Final SOC\n(fraction)']

    x = np.arange(len(metric_names))
    width = 0.8 / len(strategies)

    for idx, strategy in enumerate(strategies):
        metrics = metrics_dict[strategy]
        values = [
            metrics.get('avg_velocity', 0),
            metrics.get('efficiency_m_per_wh', 0),
            metrics.get('final_soc', 0)
        ]
        offset = (idx - len(strategies)/2) * width
        axes[1, 1].bar(x + offset, values, width,
                      label=strategy, color=colors[idx])

    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metric_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def plot_uncertainty_bands(trajectory: List, uncertainties: List,
                          title: str = "Predictions with Uncertainty") -> Image.Image:
    """
    Plot trajectory predictions with uncertainty bands.

    Args:
        trajectory: Mean trajectory
        uncertainties: List of uncertainty vectors [velocity_std, x_std, y_std, soc_std]
        title: Plot title

    Returns:
        PIL Image
    """
    times = [s.time for s in trajectory]
    velocities = [s.velocity for s in trajectory]
    socs = [s.battery_soc * 100 for s in trajectory]

    velocity_stds = [u[0] for u in uncertainties]
    soc_stds = [u[3] * 100 for u in uncertainties]  # Convert to percentage

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Velocity with uncertainty
    axes[0].plot(times, velocities, linewidth=2, color='#2196F3', label='Mean')
    axes[0].fill_between(
        times,
        [v - 2*std for v, std in zip(velocities, velocity_stds)],
        [v + 2*std for v, std in zip(velocities, velocity_stds)],
        alpha=0.3, color='#2196F3', label='95% confidence'
    )
    axes[0].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[0].set_title('Speed Prediction', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SOC with uncertainty
    axes[1].plot(times, socs, linewidth=2, color='#4CAF50', label='Mean')
    axes[1].fill_between(
        times,
        [s - 2*std for s, std in zip(socs, soc_stds)],
        [s + 2*std for s, std in zip(socs, soc_stds)],
        alpha=0.3, color='#4CAF50', label='95% confidence'
    )
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('SOC (%)', fontsize=12)
    axes[1].set_title('Battery State of Charge Prediction', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_control_visualization(control_sequence: List[Tuple[float, float]],
                                title: str = "Optimal Control Strategy") -> Image.Image:
    """
    Visualize the control inputs over time.

    Args:
        control_sequence: List of (motor_current, sun_intensity) tuples
        title: Plot title

    Returns:
        PIL Image
    """
    times = list(range(len(control_sequence)))
    currents = [c[0] for c in control_sequence]
    sun_intensities = [c[1] for c in control_sequence]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Motor current
    axes[0].plot(times, currents, linewidth=2, color='#F44336')
    axes[0].fill_between(times, currents, alpha=0.3, color='#F44336')
    axes[0].set_ylabel('Motor Current (A)', fontsize=12)
    axes[0].set_title('Motor Current Command', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Sun intensity
    axes[1].plot(times, sun_intensities, linewidth=2, color='#FFC107')
    axes[1].fill_between(times, sun_intensities, alpha=0.3, color='#FFC107')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Solar Irradiance (W/m²)', fontsize=12)
    axes[1].set_title('Solar Conditions', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


if PLOTLY_AVAILABLE:
    def create_interactive_trajectory_plot(trajectories_dict: dict):
        """Create interactive Plotly trajectory visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('2D Trajectory', 'Speed vs Time',
                          'Battery SOC', 'Power Balance'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for idx, (strategy, traj) in enumerate(trajectories_dict.items()):
            color = colors[idx % len(colors)]

            positions = np.array([s.position for s in traj])
            times = [s.time for s in traj]
            velocities = [s.velocity for s in traj]
            socs = [s.battery_soc * 100 for s in traj]
            power_in = [s.solar_power for s in traj]
            power_out = [s.motor_current * s.battery_voltage for s in traj]

            # 2D trajectory
            fig.add_trace(
                go.Scatter(x=positions[:, 0], y=positions[:, 1],
                          mode='lines+markers', name=strategy,
                          line=dict(color=color)),
                row=1, col=1
            )

            # Speed
            fig.add_trace(
                go.Scatter(x=times, y=velocities,
                          mode='lines', name=strategy,
                          line=dict(color=color), showlegend=False),
                row=1, col=2
            )

            # SOC
            fig.add_trace(
                go.Scatter(x=times, y=socs,
                          mode='lines', name=strategy,
                          line=dict(color=color), showlegend=False),
                row=2, col=1
            )

            # Power balance
            fig.add_trace(
                go.Scatter(x=times, y=power_in,
                          mode='lines', name=f'{strategy} (solar)',
                          line=dict(color=color, dash='dot'), showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=times, y=power_out,
                          mode='lines', name=f'{strategy} (motor)',
                          line=dict(color=color), showlegend=False),
                row=2, col=2
            )

        fig.update_xaxes(title_text="X (m)", row=1, col=1)
        fig.update_yaxes(title_text="Y (m)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="SOC (%)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Power (W)", row=2, col=2)

        fig.update_layout(height=800, title_text="World Model Simulation Results")

        return fig
