"""System Identification for Solar Boat Parameters.

This module provides tools to identify/estimate physical parameters of a solar
boat from real telemetry data. These parameters can then be used to calibrate
the world model for accurate predictions.

Key Parameters Estimated:
- Drag coefficient (C_d): From velocity decay during coasting
- Motor efficiency (eta_motor): From power-in vs mechanical power-out
- Propeller efficiency (eta_prop): From thrust vs motor power
- Battery internal resistance (R_int): From voltage sag under load
- Effective mass (m_eff): From acceleration response to thrust
- Solar panel efficiency: From solar power vs irradiance

Theory:
- Drag: F_drag = 0.5 * rho * C_d * A * v^2
- Thrust: F_thrust = (P_motor * eta_motor * eta_prop) / v
- Motion: m * dv/dt = F_thrust - F_drag
- Battery: V = V_oc - I * R_int
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from scipy import optimize
from scipy import signal


@dataclass
class BoatParameters:
    """Physical parameters of a solar boat.

    These parameters define the boat's behavior in the world model.
    All values in SI units unless otherwise noted.
    """
    # Hull parameters
    mass: float = 50.0  # kg (boat + payload)
    hull_length: float = 3.0  # m
    hull_width: float = 0.8  # m
    draft: float = 0.15  # m (depth below waterline)
    wetted_area: float = 1.5  # m^2 (underwater surface area)

    # Drag coefficients
    drag_coefficient: float = 0.3  # Dimensionless
    frontal_area: float = 0.12  # m^2 (cross-section facing motion)

    # Propulsion
    motor_efficiency: float = 0.85  # 0-1
    propeller_efficiency: float = 0.65  # 0-1
    motor_kv: float = 150.0  # RPM per volt (motor constant)

    # Battery
    battery_capacity_ah: float = 20.0  # Ah
    battery_voltage_nominal: float = 36.0  # V
    battery_internal_resistance: float = 0.1  # Ohms

    # Solar
    solar_panel_area: float = 0.5  # m^2
    solar_panel_efficiency: float = 0.18  # 0-1

    # Estimation metadata
    estimated_from_data: bool = False
    estimation_date: Optional[str] = None
    estimation_quality: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoatParameters':
        """Create from dictionary."""
        return cls(**data)

    def save(self, filepath: str | Path):
        """Save parameters to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> 'BoatParameters':
        """Load parameters from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class TelemetrySegment:
    """A segment of telemetry data for analysis."""
    timestamps: np.ndarray  # Unix timestamps
    speeds: np.ndarray  # m/s
    voltages: np.ndarray  # V
    currents: np.ndarray  # A
    rpms: Optional[np.ndarray] = None
    powers: Optional[np.ndarray] = None  # W (calculated if not provided)

    def __post_init__(self):
        """Calculate derived quantities."""
        if self.powers is None:
            self.powers = self.voltages * self.currents

    @property
    def dt(self) -> np.ndarray:
        """Time deltas between samples."""
        return np.diff(self.timestamps)

    @property
    def dv(self) -> np.ndarray:
        """Velocity changes between samples."""
        return np.diff(self.speeds)

    @property
    def accelerations(self) -> np.ndarray:
        """Acceleration at each timestep."""
        return self.dv / self.dt


class SystemIdentifier:
    """Estimate boat parameters from telemetry data.

    Uses various analysis techniques to extract physical parameters
    from recorded telemetry. Results can calibrate the world model.

    Example:
        >>> identifier = SystemIdentifier()
        >>> identifier.load_telemetry(telemetry_points)
        >>> params = identifier.identify_all()
        >>> params.save('my_boat_params.json')
    """

    # Physical constants
    RHO_WATER = 1000.0  # kg/m^3
    RHO_AIR = 1.225  # kg/m^3

    def __init__(self, base_params: Optional[BoatParameters] = None):
        """Initialize system identifier.

        Args:
            base_params: Starting parameter estimates (uses defaults if None)
        """
        self.params = base_params or BoatParameters()
        self.segments: List[TelemetrySegment] = []
        self._coasting_segments: List[TelemetrySegment] = []
        self._powered_segments: List[TelemetrySegment] = []

    def load_telemetry(self, telemetry: List[Any]):
        """Load telemetry data for analysis.

        Args:
            telemetry: List of VESCTelemetryPoint objects
        """
        if not telemetry:
            raise ValueError("Empty telemetry data")

        # Extract arrays from telemetry points
        timestamps = np.array([
            p.timestamp.timestamp() if hasattr(p.timestamp, 'timestamp')
            else p.timestamp
            for p in telemetry
        ])
        speeds = np.array([p.speed_gps for p in telemetry])
        voltages = np.array([p.battery_voltage for p in telemetry])
        currents = np.array([p.motor_current for p in telemetry])

        rpms = None
        if hasattr(telemetry[0], 'motor_rpm') and telemetry[0].motor_rpm is not None:
            rpms = np.array([p.motor_rpm or 0 for p in telemetry])

        # Create main segment
        main_segment = TelemetrySegment(
            timestamps=timestamps,
            speeds=speeds,
            voltages=voltages,
            currents=currents,
            rpms=rpms
        )
        self.segments = [main_segment]

        # Identify coasting and powered segments
        self._identify_segments(main_segment)

    def _identify_segments(self, segment: TelemetrySegment, current_threshold: float = 0.5):
        """Identify coasting and powered segments in the data.

        Args:
            segment: Main telemetry segment
            current_threshold: Current below which is considered coasting (A)
        """
        self._coasting_segments = []
        self._powered_segments = []

        # Find contiguous coasting regions (low current)
        is_coasting = segment.currents < current_threshold
        changes = np.diff(is_coasting.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if is_coasting[0]:
            starts = np.concatenate([[0], starts])
        if is_coasting[-1]:
            ends = np.concatenate([ends, [len(is_coasting)]])

        # Extract coasting segments (minimum 5 points)
        for start, end in zip(starts, ends):
            if end - start >= 5:
                self._coasting_segments.append(TelemetrySegment(
                    timestamps=segment.timestamps[start:end],
                    speeds=segment.speeds[start:end],
                    voltages=segment.voltages[start:end],
                    currents=segment.currents[start:end],
                    rpms=segment.rpms[start:end] if segment.rpms is not None else None,
                ))

        # Find powered segments (high current)
        is_powered = segment.currents > current_threshold * 2
        changes = np.diff(is_powered.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if is_powered[0]:
            starts = np.concatenate([[0], starts])
        if is_powered[-1]:
            ends = np.concatenate([ends, [len(is_powered)]])

        for start, end in zip(starts, ends):
            if end - start >= 5:
                self._powered_segments.append(TelemetrySegment(
                    timestamps=segment.timestamps[start:end],
                    speeds=segment.speeds[start:end],
                    voltages=segment.voltages[start:end],
                    currents=segment.currents[start:end],
                    rpms=segment.rpms[start:end] if segment.rpms is not None else None,
                ))

        print(f"Found {len(self._coasting_segments)} coasting segments, "
              f"{len(self._powered_segments)} powered segments")

    def identify_drag_coefficient(self) -> Tuple[float, float]:
        """Estimate drag coefficient from coasting (motor-off) segments.

        During coasting, the only force is drag:
            m * dv/dt = -F_drag = -0.5 * rho * C_d * A * v^2

        This gives exponential velocity decay. We fit this to find C_d.

        Returns:
            Tuple of (drag_coefficient, r_squared_fit_quality)
        """
        if not self._coasting_segments:
            print("No coasting segments found - cannot estimate drag")
            return self.params.drag_coefficient, 0.0

        # Collect all coasting data
        all_v = []
        all_dv_dt = []

        for seg in self._coasting_segments:
            if len(seg.speeds) < 3:
                continue

            # Calculate dv/dt using central differences
            dt = np.mean(seg.dt)
            dv_dt = np.gradient(seg.speeds, dt)

            # Only use decelerating portions
            mask = (dv_dt < 0) & (seg.speeds > 0.1)
            if np.sum(mask) > 2:
                all_v.extend(seg.speeds[mask])
                all_dv_dt.extend(dv_dt[mask])

        if len(all_v) < 5:
            print("Insufficient coasting data for drag estimation")
            return self.params.drag_coefficient, 0.0

        all_v = np.array(all_v)
        all_dv_dt = np.array(all_dv_dt)

        # Fit: dv/dt = -k * v^2, where k = 0.5 * rho * C_d * A / m
        # Linear regression on log-log scale
        def drag_model(v, k):
            return -k * v**2

        try:
            popt, pcov = optimize.curve_fit(
                drag_model, all_v, all_dv_dt,
                p0=[0.01], bounds=(0, 1)
            )
            k_fit = popt[0]

            # Calculate C_d from k
            # k = 0.5 * rho * C_d * A / m
            # C_d = 2 * k * m / (rho * A)
            c_d = 2 * k_fit * self.params.mass / (
                self.RHO_WATER * self.params.frontal_area
            )

            # Calculate R^2
            predictions = drag_model(all_v, k_fit)
            ss_res = np.sum((all_dv_dt - predictions)**2)
            ss_tot = np.sum((all_dv_dt - np.mean(all_dv_dt))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print(f"Estimated drag coefficient: {c_d:.3f} (R² = {r_squared:.3f})")
            return c_d, r_squared

        except Exception as e:
            print(f"Drag estimation failed: {e}")
            return self.params.drag_coefficient, 0.0

    def identify_battery_resistance(self) -> Tuple[float, float]:
        """Estimate battery internal resistance from voltage sag.

        V = V_oc - I * R_int

        At different currents, voltage changes linearly with current.
        Slope of V vs I gives internal resistance.

        Returns:
            Tuple of (internal_resistance_ohms, r_squared_fit_quality)
        """
        if not self.segments:
            return self.params.battery_internal_resistance, 0.0

        seg = self.segments[0]

        # Need varying current levels
        if np.std(seg.currents) < 1.0:
            print("Insufficient current variation for resistance estimation")
            return self.params.battery_internal_resistance, 0.0

        # Linear fit: V = V_oc - R * I
        # Using numpy polyfit for V = a + b*I, R = -b
        try:
            coeffs = np.polyfit(seg.currents, seg.voltages, 1)
            r_int = -coeffs[0]  # Slope is -R

            # R^2 calculation
            v_pred = np.polyval(coeffs, seg.currents)
            ss_res = np.sum((seg.voltages - v_pred)**2)
            ss_tot = np.sum((seg.voltages - np.mean(seg.voltages))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Sanity check
            if r_int < 0.01 or r_int > 2.0:
                print(f"Unrealistic resistance estimate: {r_int:.3f}Ω")
                return self.params.battery_internal_resistance, 0.0

            print(f"Estimated battery internal resistance: {r_int:.3f}Ω (R² = {r_squared:.3f})")
            return r_int, r_squared

        except Exception as e:
            print(f"Resistance estimation failed: {e}")
            return self.params.battery_internal_resistance, 0.0

    def identify_motor_efficiency(self) -> Tuple[float, float]:
        """Estimate motor+propeller efficiency from power and speed.

        At steady state: P_thrust = F_drag * v = 0.5 * rho * C_d * A * v^3
        Efficiency = P_thrust / P_electrical

        Returns:
            Tuple of (combined_efficiency, r_squared_fit_quality)
        """
        if not self._powered_segments:
            print("No powered segments for efficiency estimation")
            return self.params.motor_efficiency * self.params.propeller_efficiency, 0.0

        # Collect steady-state data points
        efficiencies = []

        for seg in self._powered_segments:
            # Filter for steady-state (low acceleration)
            if len(seg.speeds) < 5:
                continue

            dt = np.mean(seg.dt)
            dv_dt = np.abs(np.gradient(seg.speeds, dt))

            # Steady state = low acceleration and reasonable speed
            mask = (dv_dt < 0.1) & (seg.speeds > 0.3) & (seg.powers > 10)

            for i in np.where(mask)[0]:
                v = seg.speeds[i]
                p_in = seg.powers[i]

                # Thrust power needed at this speed
                # P_thrust = F_drag * v = 0.5 * rho * C_d * A * v^3
                p_thrust = 0.5 * self.RHO_WATER * self.params.drag_coefficient * \
                           self.params.frontal_area * v**3

                if p_in > 0:
                    eta = p_thrust / p_in
                    if 0.1 < eta < 1.0:  # Sanity check
                        efficiencies.append(eta)

        if len(efficiencies) < 3:
            print("Insufficient data for efficiency estimation")
            return self.params.motor_efficiency * self.params.propeller_efficiency, 0.0

        efficiencies = np.array(efficiencies)
        eta_mean = np.mean(efficiencies)
        eta_std = np.std(efficiencies)

        # Quality metric based on consistency
        quality = 1.0 - min(eta_std / eta_mean, 1.0) if eta_mean > 0 else 0

        print(f"Estimated combined efficiency: {eta_mean:.3f} ± {eta_std:.3f} "
              f"(quality = {quality:.2f})")
        return eta_mean, quality

    def identify_effective_mass(self) -> Tuple[float, float]:
        """Estimate effective mass from acceleration response.

        F = m * a
        F_net = F_thrust - F_drag
        m = F_net / a

        Returns:
            Tuple of (effective_mass_kg, estimation_quality)
        """
        if not self._powered_segments:
            print("No powered segments for mass estimation")
            return self.params.mass, 0.0

        mass_estimates = []

        for seg in self._powered_segments:
            if len(seg.speeds) < 5:
                continue

            dt = np.mean(seg.dt)
            accelerations = np.gradient(seg.speeds, dt)

            # Look for acceleration phases (positive acceleration, power applied)
            mask = (accelerations > 0.05) & (seg.powers > 20) & (seg.speeds > 0.1)

            for i in np.where(mask)[0]:
                v = seg.speeds[i]
                p = seg.powers[i]
                a = accelerations[i]

                # Estimate thrust force
                eta = self.params.motor_efficiency * self.params.propeller_efficiency
                f_thrust = eta * p / max(v, 0.1)

                # Drag force
                f_drag = 0.5 * self.RHO_WATER * self.params.drag_coefficient * \
                         self.params.frontal_area * v**2

                # Net force
                f_net = f_thrust - f_drag

                if f_net > 0 and a > 0:
                    m = f_net / a
                    if 10 < m < 500:  # Sanity bounds
                        mass_estimates.append(m)

        if len(mass_estimates) < 3:
            print("Insufficient data for mass estimation")
            return self.params.mass, 0.0

        mass_estimates = np.array(mass_estimates)
        m_mean = np.median(mass_estimates)  # Median more robust
        m_std = np.std(mass_estimates)

        quality = 1.0 - min(m_std / m_mean, 1.0) if m_mean > 0 else 0

        print(f"Estimated effective mass: {m_mean:.1f} ± {m_std:.1f} kg "
              f"(quality = {quality:.2f})")
        return m_mean, quality

    def identify_all(self) -> BoatParameters:
        """Run all identification routines and return updated parameters.

        Returns:
            BoatParameters with estimated values
        """
        params = BoatParameters()

        # Copy base values
        params.hull_length = self.params.hull_length
        params.hull_width = self.params.hull_width
        params.draft = self.params.draft
        params.wetted_area = self.params.wetted_area
        params.frontal_area = self.params.frontal_area
        params.motor_kv = self.params.motor_kv
        params.battery_capacity_ah = self.params.battery_capacity_ah
        params.battery_voltage_nominal = self.params.battery_voltage_nominal
        params.solar_panel_area = self.params.solar_panel_area
        params.solar_panel_efficiency = self.params.solar_panel_efficiency

        # Run identification routines
        print("\n=== System Identification ===\n")

        # 1. Battery resistance (affects voltage readings)
        r_int, r_quality = self.identify_battery_resistance()
        params.battery_internal_resistance = r_int
        params.estimation_quality['battery_resistance'] = r_quality

        # 2. Drag coefficient (from coasting)
        c_d, c_d_quality = self.identify_drag_coefficient()
        params.drag_coefficient = c_d
        params.estimation_quality['drag_coefficient'] = c_d_quality

        # 3. Motor efficiency (needs drag estimate)
        self.params.drag_coefficient = c_d  # Update for efficiency calc
        eta, eta_quality = self.identify_motor_efficiency()

        # Split combined efficiency into motor and propeller
        # Assume 60/40 split if combined is reasonable
        if 0.3 < eta < 0.8:
            params.motor_efficiency = np.sqrt(eta / 0.65) * 0.85  # Normalized
            params.propeller_efficiency = eta / params.motor_efficiency
        else:
            params.motor_efficiency = 0.85
            params.propeller_efficiency = 0.65
        params.estimation_quality['efficiency'] = eta_quality

        # 4. Effective mass
        self.params.motor_efficiency = params.motor_efficiency
        self.params.propeller_efficiency = params.propeller_efficiency
        mass, mass_quality = self.identify_effective_mass()
        params.mass = mass
        params.estimation_quality['mass'] = mass_quality

        # Metadata
        params.estimated_from_data = True
        params.estimation_date = datetime.now().isoformat()

        # Summary
        print("\n=== Identified Parameters ===")
        print(f"  Mass: {params.mass:.1f} kg")
        print(f"  Drag coefficient: {params.drag_coefficient:.3f}")
        print(f"  Motor efficiency: {params.motor_efficiency:.2f}")
        print(f"  Propeller efficiency: {params.propeller_efficiency:.2f}")
        print(f"  Battery resistance: {params.battery_internal_resistance:.3f} Ω")

        overall_quality = np.mean(list(params.estimation_quality.values()))
        print(f"\n  Overall quality: {overall_quality:.2f}")

        return params


def identify_from_session(session_path: str | Path) -> BoatParameters:
    """Convenience function to identify parameters from a saved session.

    Args:
        session_path: Path to session JSON file

    Returns:
        Identified BoatParameters
    """
    from .collector import VESCDataCollector

    collector = VESCDataCollector(simulate=True)
    collector.load_from_file(session_path)

    identifier = SystemIdentifier()
    identifier.load_telemetry(collector.telemetry_data)
    return identifier.identify_all()


def calibrate_world_model(params: BoatParameters) -> Dict[str, Any]:
    """Convert BoatParameters to world model configuration.

    Returns a dictionary that can be passed to create_world_model().

    Args:
        params: Identified boat parameters

    Returns:
        World model configuration dictionary
    """
    return {
        'boat_mass': params.mass,
        'drag_coefficient': params.drag_coefficient,
        'frontal_area': params.frontal_area,
        'motor_efficiency': params.motor_efficiency,
        'propeller_efficiency': params.propeller_efficiency,
        'battery_capacity': params.battery_capacity_ah * params.battery_voltage_nominal,  # Wh
        'battery_voltage': params.battery_voltage_nominal,
        'battery_resistance': params.battery_internal_resistance,
        'solar_panel_area': params.solar_panel_area,
        'solar_efficiency': params.solar_panel_efficiency,
    }
