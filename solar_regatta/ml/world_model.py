"""
Advanced World Model for Solar Boat Racing.

This module implements a physics-informed world model that:
1. Models boat dynamics (drag, momentum, energy conversion)
2. Predicts future states with uncertainty estimation
3. Plans optimal trajectories and energy strategies
4. Learns from real telemetry data
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


@dataclass
class BoatState:
    """Current state of the solar boat."""
    time: float
    position: np.ndarray  # [x, y] in meters
    velocity: float  # m/s
    heading: float  # radians
    battery_voltage: float  # V
    battery_soc: float  # State of charge (0-1)
    motor_current: float  # A
    solar_power: float  # W


@dataclass
class PhysicsParameters:
    """Physical parameters of the boat and environment."""
    # Boat properties
    mass: float = 50.0  # kg
    hull_drag_coeff: float = 0.5  # dimensionless
    frontal_area: float = 0.3  # m^2

    # Motor/propulsion
    motor_efficiency: float = 0.85  # efficiency
    prop_efficiency: float = 0.65  # propeller efficiency
    max_motor_power: float = 200.0  # W

    # Battery
    battery_capacity: float = 100.0  # Wh
    battery_resistance: float = 0.1  # Ohms
    nominal_voltage: float = 13.0  # V

    # Solar
    solar_panel_area: float = 0.5  # m^2
    solar_efficiency: float = 0.18  # 18% efficiency

    # Environment
    water_density: float = 1000.0  # kg/m^3
    air_density: float = 1.225  # kg/m^3

    # Wave/current (can be updated)
    current_velocity: np.ndarray = None  # [vx, vy] m/s
    wave_resistance: float = 0.0  # N

    def __post_init__(self):
        if self.current_velocity is None:
            self.current_velocity = np.array([0.0, 0.0])


class BoatDynamicsModel:
    """Physics-based model of boat dynamics."""

    def __init__(self, params: PhysicsParameters):
        self.params = params

    def calculate_drag_force(self, velocity: float) -> float:
        """Calculate total drag force (water + air resistance)."""
        if velocity <= 0:
            return 0.0

        # Water drag (dominant at low speeds)
        water_drag = 0.5 * self.params.water_density * \
                     self.params.hull_drag_coeff * \
                     self.params.frontal_area * (velocity ** 2)

        # Air drag
        air_drag = 0.5 * self.params.air_density * \
                   0.3 * self.params.frontal_area * (velocity ** 2)

        # Wave resistance
        wave_drag = self.params.wave_resistance

        return water_drag + air_drag + wave_drag

    def calculate_thrust_force(self, motor_power: float, velocity: float) -> float:
        """Calculate thrust force from motor power."""
        if velocity <= 0.1:
            velocity = 0.1  # Avoid division by zero

        # Power = Force × Velocity
        # Force = (Motor Power × Efficiency) / Velocity
        effective_power = motor_power * self.params.motor_efficiency * \
                         self.params.prop_efficiency
        thrust = effective_power / velocity

        return thrust

    def calculate_acceleration(self, velocity: float, motor_power: float) -> float:
        """Calculate acceleration given current state."""
        thrust = self.calculate_thrust_force(motor_power, velocity)
        drag = self.calculate_drag_force(velocity)

        net_force = thrust - drag
        acceleration = net_force / self.params.mass

        return acceleration

    def calculate_power_consumption(self, motor_current: float,
                                   battery_voltage: float) -> float:
        """Calculate instantaneous power consumption."""
        # Account for battery internal resistance
        voltage_drop = motor_current * self.params.battery_resistance
        effective_voltage = battery_voltage - voltage_drop
        power = effective_voltage * motor_current
        return power

    def calculate_solar_power(self, sun_intensity: float = 1000.0) -> float:
        """Calculate solar power generation (W)."""
        # sun_intensity in W/m^2 (1000 = full sun)
        power = self.params.solar_panel_area * \
                self.params.solar_efficiency * sun_intensity
        return power

    def update_battery_soc(self, current_soc: float, power_net: float,
                           dt: float) -> float:
        """Update battery state of charge."""
        # Energy change in Wh
        energy_change = (power_net * dt) / 3600.0  # Convert W*s to Wh

        # Update SOC
        soc_change = energy_change / self.params.battery_capacity
        new_soc = np.clip(current_soc + soc_change, 0.0, 1.0)

        return new_soc

    def step(self, state: BoatState, motor_current: float,
             sun_intensity: float, dt: float) -> BoatState:
        """
        Single time step simulation of boat dynamics.

        Args:
            state: Current boat state
            motor_current: Commanded motor current (A)
            sun_intensity: Solar irradiance (W/m^2)
            dt: Time step (seconds)

        Returns:
            New boat state after dt seconds
        """
        # Calculate powers
        motor_power = self.calculate_power_consumption(
            motor_current, state.battery_voltage
        )
        solar_power = self.calculate_solar_power(sun_intensity)
        net_power = solar_power - motor_power

        # Update battery
        new_soc = self.update_battery_soc(state.battery_soc, net_power, dt)
        new_voltage = self.params.nominal_voltage * (0.8 + 0.2 * new_soc)

        # Calculate acceleration and update velocity
        acceleration = self.calculate_acceleration(state.velocity, motor_power)
        new_velocity = max(0.0, state.velocity + acceleration * dt)

        # Update position
        displacement = state.velocity * dt
        new_position = state.position + displacement * np.array([
            np.cos(state.heading),
            np.sin(state.heading)
        ])

        return BoatState(
            time=state.time + dt,
            position=new_position,
            velocity=new_velocity,
            heading=state.heading,
            battery_voltage=new_voltage,
            battery_soc=new_soc,
            motor_current=motor_current,
            solar_power=solar_power
        )


class WorldModel:
    """
    Advanced world model for solar boat racing.

    Combines physics-based simulation with learned corrections from data.
    """

    def __init__(self, params: Optional[PhysicsParameters] = None):
        if params is None:
            params = PhysicsParameters()

        self.params = params
        self.dynamics = BoatDynamicsModel(params)

        # Learned correction model (residual learning)
        self.correction_model = None
        self.is_trained = False

    def predict_trajectory(self, initial_state: BoatState,
                          control_sequence: List[Tuple[float, float]],
                          dt: float = 1.0) -> List[BoatState]:
        """
        Predict future trajectory given control inputs.

        Args:
            initial_state: Starting state
            control_sequence: List of (motor_current, sun_intensity) tuples
            dt: Time step for simulation

        Returns:
            List of predicted states
        """
        states = [initial_state]
        current_state = initial_state

        for motor_current, sun_intensity in control_sequence:
            next_state = self.dynamics.step(
                current_state, motor_current, sun_intensity, dt
            )
            states.append(next_state)
            current_state = next_state

        return states

    def predict_with_uncertainty(self, initial_state: BoatState,
                                control_sequence: List[Tuple[float, float]],
                                n_samples: int = 100,
                                dt: float = 1.0) -> Tuple[List[BoatState],
                                                           List[np.ndarray]]:
        """
        Monte Carlo prediction with uncertainty estimation.

        Returns:
            mean_trajectory: Mean predicted states
            uncertainty: Standard deviation at each time step
        """
        trajectories = []

        for _ in range(n_samples):
            # Add noise to parameters for uncertainty
            noisy_params = PhysicsParameters(
                mass=self.params.mass * np.random.normal(1.0, 0.05),
                hull_drag_coeff=self.params.hull_drag_coeff * np.random.normal(1.0, 0.1),
                motor_efficiency=np.clip(
                    self.params.motor_efficiency * np.random.normal(1.0, 0.05),
                    0.6, 0.95
                ),
                prop_efficiency=np.clip(
                    self.params.prop_efficiency * np.random.normal(1.0, 0.1),
                    0.4, 0.8
                )
            )

            temp_dynamics = BoatDynamicsModel(noisy_params)

            # Simulate trajectory
            traj = []
            current_state = initial_state
            for motor_current, sun_intensity in control_sequence:
                next_state = temp_dynamics.step(
                    current_state, motor_current, sun_intensity, dt
                )
                traj.append(next_state)
                current_state = next_state

            trajectories.append(traj)

        # Compute statistics
        mean_trajectory = []
        uncertainties = []

        n_steps = len(trajectories[0])
        for step_idx in range(n_steps):
            # Extract values at this time step
            velocities = [traj[step_idx].velocity for traj in trajectories]
            positions = [traj[step_idx].position for traj in trajectories]
            socs = [traj[step_idx].battery_soc for traj in trajectories]

            # Compute mean state
            mean_state = BoatState(
                time=trajectories[0][step_idx].time,
                position=np.mean(positions, axis=0),
                velocity=np.mean(velocities),
                heading=trajectories[0][step_idx].heading,
                battery_voltage=np.mean([t[step_idx].battery_voltage for t in trajectories]),
                battery_soc=np.mean(socs),
                motor_current=trajectories[0][step_idx].motor_current,
                solar_power=trajectories[0][step_idx].solar_power
            )

            # Compute uncertainty
            uncertainty = np.array([
                np.std(velocities),
                np.std([p[0] for p in positions]),
                np.std([p[1] for p in positions]),
                np.std(socs)
            ])

            mean_trajectory.append(mean_state)
            uncertainties.append(uncertainty)

        return mean_trajectory, uncertainties

    def optimize_control_strategy(self, initial_state: BoatState,
                                  target_distance: float,
                                  max_time: float,
                                  sun_intensity_profile: List[float],
                                  dt: float = 1.0) -> np.ndarray:
        """
        Optimize motor current profile to maximize speed while managing energy.

        Args:
            initial_state: Starting state
            target_distance: Distance to cover (meters)
            max_time: Maximum time allowed (seconds)
            sun_intensity_profile: Solar intensity over time
            dt: Time step

        Returns:
            Optimal motor current sequence
        """
        n_steps = int(max_time / dt)

        def objective(motor_currents):
            """Objective: minimize time while avoiding battery depletion."""
            control_seq = [(mc, sun_intensity_profile[min(i, len(sun_intensity_profile)-1)])
                          for i, mc in enumerate(motor_currents)]

            trajectory = self.predict_trajectory(initial_state, control_seq, dt)

            # Calculate total distance
            total_distance = 0.0
            for i in range(1, len(trajectory)):
                total_distance += np.linalg.norm(
                    trajectory[i].position - trajectory[i-1].position
                )

            # Penalties
            distance_penalty = max(0, target_distance - total_distance) ** 2

            # Battery penalty (avoid full depletion)
            min_soc = min(s.battery_soc for s in trajectory)
            battery_penalty = 1000.0 if min_soc < 0.1 else 0.0

            # Time penalty (want to finish fast)
            time_penalty = trajectory[-1].time

            return distance_penalty * 10.0 + battery_penalty + time_penalty

        # Initial guess: constant moderate current
        x0 = np.ones(n_steps) * 3.0

        # Bounds: 0 to 15A
        bounds = [(0.0, 15.0)] * n_steps

        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        return result.x

    def learn_from_telemetry(self, telemetry_states: List[BoatState],
                            telemetry_controls: List[Tuple[float, float]]):
        """
        Fit correction model to real telemetry data.

        Uses residual learning: learns the difference between physics model
        and real observations.
        """
        if len(telemetry_states) < 2:
            return

        # Simulate what physics model predicts
        predicted_states = self.predict_trajectory(
            telemetry_states[0],
            telemetry_controls,
            dt=1.0
        )

        # Calculate residuals
        residuals = []
        for i in range(1, len(telemetry_states)):
            if i >= len(predicted_states):
                break

            velocity_error = telemetry_states[i].velocity - predicted_states[i].velocity
            soc_error = telemetry_states[i].battery_soc - predicted_states[i].battery_soc

            residuals.append([velocity_error, soc_error])

        # Fit simple correction model (could use ML here)
        if len(residuals) > 0:
            self.correction_model = np.mean(residuals, axis=0)
            self.is_trained = True

    def get_performance_metrics(self, trajectory: List[BoatState]) -> Dict[str, float]:
        """Calculate performance metrics for a trajectory."""
        if len(trajectory) < 2:
            return {}

        velocities = [s.velocity for s in trajectory]
        distances = []
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i].position - trajectory[i-1].position)
            distances.append(dist)

        total_distance = sum(distances)
        total_time = trajectory[-1].time - trajectory[0].time
        energy_used = (trajectory[0].battery_soc - trajectory[-1].battery_soc) * \
                     self.params.battery_capacity

        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'energy_used_wh': energy_used,
            'efficiency_m_per_wh': total_distance / energy_used if energy_used > 0 else 0,
            'final_soc': trajectory[-1].battery_soc,
            'avg_power': np.mean([s.solar_power - s.motor_current * s.battery_voltage
                                  for s in trajectory])
        }


def create_default_world_model() -> WorldModel:
    """Create world model with sensible defaults for solar boat."""
    params = PhysicsParameters(
        mass=50.0,  # 50kg boat
        hull_drag_coeff=0.5,
        frontal_area=0.3,
        motor_efficiency=0.85,
        prop_efficiency=0.65,
        battery_capacity=100.0,  # 100 Wh
        solar_panel_area=0.5,  # 0.5 m^2
        solar_efficiency=0.18
    )

    return WorldModel(params)


def simulate_race(world_model: WorldModel,
                 race_distance: float,
                 sun_profile: List[float],
                 strategy: str = 'optimal') -> Tuple[List[BoatState], Dict]:
    """
    Simulate a complete race.

    Args:
        world_model: The world model to use
        race_distance: Total race distance in meters
        sun_profile: Solar intensity profile over time
        strategy: 'optimal', 'conservative', or 'aggressive'

    Returns:
        trajectory: List of boat states
        metrics: Performance metrics
    """
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

    if strategy == 'optimal':
        # Use optimization
        max_time = 600.0  # 10 minutes
        optimal_currents = world_model.optimize_control_strategy(
            initial_state, race_distance, max_time, sun_profile, dt=1.0
        )
        control_seq = [(curr, sun_profile[min(i, len(sun_profile)-1)])
                      for i, curr in enumerate(optimal_currents)]
    elif strategy == 'aggressive':
        # High constant current
        control_seq = [(8.0, sun_profile[min(i, len(sun_profile)-1)])
                      for i in range(600)]
    else:  # conservative
        # Low constant current
        control_seq = [(3.0, sun_profile[min(i, len(sun_profile)-1)])
                      for i in range(600)]

    trajectory = world_model.predict_trajectory(initial_state, control_seq, dt=1.0)
    metrics = world_model.get_performance_metrics(trajectory)

    return trajectory, metrics
