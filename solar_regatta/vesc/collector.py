"""VESC Tool data collection utilities.

This module provides interfaces for collecting telemetry data from VESC motor
controllers via serial connection. Supports both real hardware and simulated
data for testing.
"""
from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from queue import Queue, Empty

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from .protocol import VESCProtocol, VESCValues, VESCCommand, get_fault_description


@dataclass
class VESCTelemetryPoint:
    """Single telemetry data point from VESC."""

    timestamp: datetime
    gps_position: str  # MGRS format
    speed_gps: float  # m/s from GPS
    battery_voltage: float  # V
    motor_current: float  # A
    input_current: float = 0.0  # A (battery current)
    motor_rpm: Optional[float] = None
    duty_cycle: Optional[float] = None
    amp_hours: Optional[float] = None
    watt_hours: Optional[float] = None
    temp_fet: Optional[float] = None
    temp_motor: Optional[float] = None
    fault_code: int = 0
    power_in: Optional[float] = None  # W
    efficiency: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable timestamp."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_vesc_values(
        cls,
        values: VESCValues,
        gps_position: str = "",
        speed_gps: float = 0.0
    ) -> 'VESCTelemetryPoint':
        """Create telemetry point from VESCValues.

        Args:
            values: Parsed VESC telemetry
            gps_position: GPS position in MGRS format
            speed_gps: GPS-derived speed in m/s

        Returns:
            VESCTelemetryPoint instance
        """
        return cls(
            timestamp=datetime.now(),
            gps_position=gps_position,
            speed_gps=speed_gps,
            battery_voltage=values.input_voltage,
            motor_current=values.avg_motor_current,
            input_current=values.avg_input_current,
            motor_rpm=values.rpm,
            duty_cycle=values.duty_cycle,
            amp_hours=values.amp_hours,
            watt_hours=values.watt_hours,
            temp_fet=values.temp_fet,
            temp_motor=values.temp_motor,
            fault_code=values.fault_code,
            power_in=values.power_in,
            efficiency=values.efficiency,
        )


@dataclass
class RaceSession:
    """Container for a race data collection session."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    location: str = ""
    conditions: str = ""  # Weather, water conditions, etc.
    boat_name: str = ""
    notes: str = ""
    telemetry: List[VESCTelemetryPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'location': self.location,
            'conditions': self.conditions,
            'boat_name': self.boat_name,
            'notes': self.notes,
            'telemetry': [p.to_dict() for p in self.telemetry],
            'summary': self.get_summary(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary statistics."""
        if not self.telemetry:
            return {}

        voltages = [p.battery_voltage for p in self.telemetry]
        currents = [p.motor_current for p in self.telemetry]
        speeds = [p.speed_gps for p in self.telemetry if p.speed_gps > 0]

        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0

        return {
            'duration_seconds': duration,
            'num_points': len(self.telemetry),
            'voltage_min': min(voltages),
            'voltage_max': max(voltages),
            'voltage_avg': sum(voltages) / len(voltages),
            'current_max': max(currents),
            'current_avg': sum(currents) / len(currents),
            'speed_max': max(speeds) if speeds else 0,
            'speed_avg': sum(speeds) / len(speeds) if speeds else 0,
            'total_wh': self.telemetry[-1].watt_hours if self.telemetry[-1].watt_hours else 0,
            'total_ah': self.telemetry[-1].amp_hours if self.telemetry[-1].amp_hours else 0,
        }


class VESCDataCollector:
    """Collect telemetry data from VESC motor controller.

    Supports both synchronous and asynchronous (threaded) data collection.
    Can interface with real hardware or generate simulated data for testing.

    Example:
        >>> collector = VESCDataCollector(port='/dev/ttyUSB0')
        >>> collector.connect()
        >>> collector.start_collection(duration=60, interval=0.5)
        >>> data = collector.get_data()
        >>> collector.save_session('race_001.json')
    """

    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        baudrate: int = 115200,
        simulate: bool = False
    ):
        """Initialize VESC data collector.

        Args:
            port: Serial port where VESC is connected
            baudrate: Serial communication baud rate
            simulate: If True, generate simulated data instead of reading hardware
        """
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate

        self.serial_conn: Optional[serial.Serial] = None
        self.protocol: Optional[VESCProtocol] = None
        self.telemetry_data: List[VESCTelemetryPoint] = []
        self.is_collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        self._data_queue: Queue = Queue()
        self._callbacks: List[Callable[[VESCTelemetryPoint], None]] = []

        # Session management
        self.current_session: Optional[RaceSession] = None

        # Simulation state
        self._sim_voltage = 42.0
        self._sim_current = 0.0
        self._sim_speed = 0.0

        if not simulate and not SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial is required for VESC hardware integration. "
                "Install with: pip install pyserial"
            )

    @staticmethod
    def list_ports() -> List[Dict[str, str]]:
        """List available serial ports.

        Returns:
            List of dicts with port info (device, description, hwid)
        """
        if not SERIAL_AVAILABLE:
            return []

        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
            })
        return ports

    def connect(self) -> bool:
        """Connect to VESC via serial port.

        Returns:
            True if connection successful, False otherwise
        """
        if self.simulate:
            print("Running in simulation mode (no hardware)")
            return True

        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            self.protocol = VESCProtocol(self.serial_conn)
            print(f"Connected to VESC on {self.port}")

            # Try to read initial values to verify connection
            values = self.protocol.get_values()
            if values:
                print(f"  Battery: {values.input_voltage:.1f}V")
                print(f"  FET Temp: {values.temp_fet:.1f}C")
                if values.fault_code:
                    print(f"  WARNING: {get_fault_description(values.fault_code)}")
            return True

        except serial.SerialException as e:
            print(f"Failed to connect to VESC: {e}")
            return False

    def disconnect(self):
        """Disconnect from VESC."""
        self.stop_collection()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from VESC")
        self.serial_conn = None
        self.protocol = None

    def is_connected(self) -> bool:
        """Check if connected to VESC."""
        if self.simulate:
            return True
        return self.serial_conn is not None and self.serial_conn.is_open

    def add_callback(self, callback: Callable[[VESCTelemetryPoint], None]):
        """Add callback for real-time data updates.

        Callback will be called with each new telemetry point during collection.

        Args:
            callback: Function taking VESCTelemetryPoint
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[VESCTelemetryPoint], None]):
        """Remove a previously added callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def read_telemetry_point(self, gps_position: str = "", speed_gps: float = 0.0) -> Optional[VESCTelemetryPoint]:
        """Read a single telemetry point from VESC.

        Args:
            gps_position: GPS position in MGRS format (from external GPS)
            speed_gps: GPS-derived speed in m/s

        Returns:
            VESCTelemetryPoint if successful, None otherwise
        """
        if self.simulate:
            return self._generate_simulated_point(gps_position, speed_gps)

        if not self.protocol:
            return None

        values = self.protocol.get_values()
        if values:
            return VESCTelemetryPoint.from_vesc_values(values, gps_position, speed_gps)
        return None

    def _generate_simulated_point(self, gps_position: str = "", speed_gps: float = 0.0) -> VESCTelemetryPoint:
        """Generate simulated telemetry for testing.

        Simulates realistic boat behavior with varying load.
        """
        import random
        import math

        # Simulate varying motor current (0-15A with some noise)
        t = time.time()
        base_current = 5.0 + 4.0 * math.sin(t / 30)  # Slow variation
        self._sim_current = max(0, base_current + random.gauss(0, 0.5))

        # Voltage drops with current (internal resistance effect)
        self._sim_voltage = 42.0 - 0.1 * self._sim_current + random.gauss(0, 0.1)

        # Speed correlates with current
        self._sim_speed = 0.3 * self._sim_current + random.gauss(0, 0.1)

        # RPM based on speed (assuming direct drive ratio)
        rpm = self._sim_speed * 200  # ~200 RPM per m/s

        return VESCTelemetryPoint(
            timestamp=datetime.now(),
            gps_position=gps_position or "18TXL8910023456",
            speed_gps=speed_gps or self._sim_speed,
            battery_voltage=self._sim_voltage,
            motor_current=self._sim_current,
            input_current=self._sim_current * 0.95,
            motor_rpm=rpm,
            duty_cycle=self._sim_current / 20.0,  # 20A = 100%
            amp_hours=0.0,
            watt_hours=0.0,
            temp_fet=35.0 + self._sim_current * 0.5 + random.gauss(0, 1),
            temp_motor=40.0 + self._sim_current * 0.8 + random.gauss(0, 1),
            fault_code=0,
            power_in=self._sim_voltage * self._sim_current,
            efficiency=0.85 + random.gauss(0, 0.02),
        )

    def start_collection(
        self,
        duration: Optional[int] = None,
        interval: float = 0.5,
        session_id: Optional[str] = None,
        location: str = "",
        conditions: str = "",
        threaded: bool = True
    ):
        """Start collecting telemetry data.

        Args:
            duration: Collection duration in seconds (None for continuous)
            interval: Time between samples in seconds
            session_id: Unique session identifier (auto-generated if None)
            location: Race location description
            conditions: Weather/water conditions
            threaded: If True, collect in background thread
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to VESC. Call connect() first.")

        if self.is_collecting:
            print("Collection already in progress")
            return

        # Create new session
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_session = RaceSession(
            session_id=session_id,
            start_time=datetime.now(),
            location=location,
            conditions=conditions,
        )
        self.telemetry_data.clear()
        self.is_collecting = True

        print(f"Starting data collection (interval={interval}s, session={session_id})")

        if threaded:
            self._collection_thread = threading.Thread(
                target=self._collection_loop,
                args=(duration, interval),
                daemon=True
            )
            self._collection_thread.start()
        else:
            self._collection_loop(duration, interval)

    def _collection_loop(self, duration: Optional[int], interval: float):
        """Internal collection loop."""
        start_time = time.time()

        try:
            while self.is_collecting:
                point = self.read_telemetry_point()
                if point:
                    self.telemetry_data.append(point)
                    self._data_queue.put(point)

                    # Call registered callbacks
                    for callback in self._callbacks:
                        try:
                            callback(point)
                        except Exception as e:
                            print(f"Callback error: {e}")

                if duration and (time.time() - start_time) >= duration:
                    break

                time.sleep(interval)

        except Exception as e:
            print(f"Collection error: {e}")
        finally:
            self.is_collecting = False
            if self.current_session:
                self.current_session.end_time = datetime.now()
                self.current_session.telemetry = self.telemetry_data.copy()
            print(f"Collection complete. Collected {len(self.telemetry_data)} points")

    def stop_collection(self):
        """Stop ongoing data collection."""
        self.is_collecting = False
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=2.0)
        self._collection_thread = None

    def get_data(self) -> List[VESCTelemetryPoint]:
        """Get collected telemetry data.

        Returns:
            List of telemetry points
        """
        return self.telemetry_data

    def get_latest(self) -> Optional[VESCTelemetryPoint]:
        """Get the most recent telemetry point.

        Returns:
            Latest point or None if no data
        """
        try:
            return self._data_queue.get_nowait()
        except Empty:
            return self.telemetry_data[-1] if self.telemetry_data else None

    def save_session(self, filepath: str | Path):
        """Save current session to JSON file.

        Args:
            filepath: Path to output file
        """
        if not self.current_session:
            print("No active session to save")
            return

        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2)

        print(f"Session saved to {filepath}")

    def save_to_file(self, filepath: str | Path, include_metadata: bool = True):
        """Save collected data to JSON file (legacy format).

        Args:
            filepath: Path to output file
            include_metadata: Include collection metadata in output
        """
        filepath = Path(filepath)

        data = {
            'telemetry': [point.to_dict() for point in self.telemetry_data]
        }

        if include_metadata:
            data['metadata'] = {
                'port': self.port,
                'baudrate': self.baudrate,
                'num_points': len(self.telemetry_data),
                'collection_time': datetime.now().isoformat(),
                'simulated': self.simulate,
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Data saved to {filepath}")

    def load_from_file(self, filepath: str | Path) -> List[VESCTelemetryPoint]:
        """Load telemetry data from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            List of loaded telemetry points
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.telemetry_data = []

        # Handle both session format and legacy format
        telemetry_list = data.get('telemetry', [])

        for point_dict in telemetry_list:
            point_dict['timestamp'] = datetime.fromisoformat(point_dict['timestamp'])
            # Handle optional fields
            self.telemetry_data.append(VESCTelemetryPoint(**point_dict))

        print(f"Loaded {len(self.telemetry_data)} points from {filepath}")
        return self.telemetry_data

    def load_session(self, filepath: str | Path) -> RaceSession:
        """Load a saved race session.

        Args:
            filepath: Path to session JSON file

        Returns:
            Loaded RaceSession
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        telemetry = []
        for point_dict in data.get('telemetry', []):
            point_dict['timestamp'] = datetime.fromisoformat(point_dict['timestamp'])
            telemetry.append(VESCTelemetryPoint(**point_dict))

        session = RaceSession(
            session_id=data['session_id'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            location=data.get('location', ''),
            conditions=data.get('conditions', ''),
            boat_name=data.get('boat_name', ''),
            notes=data.get('notes', ''),
            telemetry=telemetry,
        )

        self.current_session = session
        self.telemetry_data = telemetry
        print(f"Loaded session {session.session_id} with {len(telemetry)} points")
        return session


def connect_vesc(
    port: str = '/dev/ttyUSB0',
    baudrate: int = 115200,
    simulate: bool = False
) -> VESCDataCollector:
    """Convenience function to create and connect to VESC.

    Args:
        port: Serial port
        baudrate: Baud rate
        simulate: Use simulated data

    Returns:
        Connected VESCDataCollector instance
    """
    collector = VESCDataCollector(port=port, baudrate=baudrate, simulate=simulate)
    collector.connect()
    return collector


def read_telemetry(
    port: str = '/dev/ttyUSB0',
    duration: int = 60,
    interval: float = 0.5,
    output_file: Optional[str] = None,
    simulate: bool = False
) -> List[VESCTelemetryPoint]:
    """Quick helper to read telemetry data from VESC.

    Args:
        port: Serial port
        duration: Collection duration in seconds
        interval: Sample interval in seconds
        output_file: Optional file to save data
        simulate: Use simulated data

    Returns:
        List of telemetry points
    """
    collector = connect_vesc(port=port, simulate=simulate)
    try:
        collector.start_collection(duration=duration, interval=interval, threaded=False)
        data = collector.get_data()

        if output_file:
            collector.save_to_file(output_file)

        return data
    finally:
        collector.disconnect()
