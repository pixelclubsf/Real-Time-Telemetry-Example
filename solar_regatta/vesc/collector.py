"""VESC Tool data collection utilities.

This module provides interfaces for collecting telemetry data from VESC motor
controllers via serial connection or CAN bus.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


@dataclass
class VESCTelemetryPoint:
    """Single telemetry data point from VESC."""

    timestamp: datetime
    gps_position: str  # MGRS format
    speed_gps: float  # m/s from GPS
    battery_voltage: float  # V
    motor_current: float  # A
    motor_rpm: Optional[float] = None
    duty_cycle: Optional[float] = None
    amp_hours: Optional[float] = None
    watt_hours: Optional[float] = None
    temp_fet: Optional[float] = None
    temp_motor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable timestamp."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class VESCDataCollector:
    """Collect telemetry data from VESC motor controller.

    This class provides methods to connect to VESC Tool and collect real-time
    telemetry data for analysis and annotation.

    Example:
        >>> collector = VESCDataCollector(port='/dev/ttyUSB0', baudrate=115200)
        >>> collector.connect()
        >>> collector.start_collection(duration=60, interval=1.0)
        >>> data = collector.get_data()
        >>> collector.save_to_file('race_data.json')
    """

    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        """Initialize VESC data collector.

        Args:
            port: Serial port where VESC is connected
            baudrate: Serial communication baud rate
        """
        if not SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial is required for VESC integration. "
                "Install it with: pip install pyserial"
            )

        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.telemetry_data: List[VESCTelemetryPoint] = []
        self.is_collecting = False

    def connect(self) -> bool:
        """Connect to VESC via serial port.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            print(f"Connected to VESC on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to VESC: {e}")
            return False

    def disconnect(self):
        """Disconnect from VESC."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from VESC")

    def read_telemetry_point(self) -> Optional[VESCTelemetryPoint]:
        """Read a single telemetry point from VESC.

        Returns:
            VESCTelemetryPoint if successful, None otherwise

        Note:
            This is a placeholder. Actual implementation depends on VESC Tool's
            communication protocol. You may need to use PyVESC library or implement
            the VESC protocol directly.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return None

        # TODO: Implement actual VESC protocol reading
        # This is a placeholder that shows the expected data structure
        raise NotImplementedError(
            "Direct VESC reading requires PyVESC library or manual protocol implementation. "
            "See https://github.com/LiamBindle/PyVESC for reference implementation."
        )

    def start_collection(
        self,
        duration: Optional[int] = None,
        interval: float = 1.0,
        annotation: str = ""
    ):
        """Start collecting telemetry data.

        Args:
            duration: Collection duration in seconds (None for continuous)
            interval: Time between samples in seconds
            annotation: Optional annotation/label for this data collection session
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            raise RuntimeError("Not connected to VESC. Call connect() first.")

        self.is_collecting = True
        self.telemetry_data.clear()
        start_time = time.time()

        print(f"Starting data collection (interval={interval}s)")
        if annotation:
            print(f"Annotation: {annotation}")

        try:
            while self.is_collecting:
                point = self.read_telemetry_point()
                if point:
                    self.telemetry_data.append(point)
                    print(f"Collected point {len(self.telemetry_data)}: "
                          f"V={point.battery_voltage:.2f}V, I={point.motor_current:.2f}A")

                if duration and (time.time() - start_time) >= duration:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nCollection stopped by user")
        finally:
            self.is_collecting = False
            print(f"Collection complete. Collected {len(self.telemetry_data)} points")

    def stop_collection(self):
        """Stop ongoing data collection."""
        self.is_collecting = False

    def get_data(self) -> List[VESCTelemetryPoint]:
        """Get collected telemetry data.

        Returns:
            List of telemetry points
        """
        return self.telemetry_data

    def save_to_file(self, filepath: str | Path, include_metadata: bool = True):
        """Save collected data to JSON file.

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
                'collection_time': datetime.now().isoformat()
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
        for point_dict in data['telemetry']:
            point_dict['timestamp'] = datetime.fromisoformat(point_dict['timestamp'])
            self.telemetry_data.append(VESCTelemetryPoint(**point_dict))

        print(f"Loaded {len(self.telemetry_data)} points from {filepath}")
        return self.telemetry_data


def connect_vesc(port: str = '/dev/ttyUSB0', baudrate: int = 115200) -> VESCDataCollector:
    """Convenience function to create and connect to VESC.

    Args:
        port: Serial port
        baudrate: Baud rate

    Returns:
        Connected VESCDataCollector instance
    """
    collector = VESCDataCollector(port=port, baudrate=baudrate)
    collector.connect()
    return collector


def read_telemetry(
    port: str = '/dev/ttyUSB0',
    duration: int = 60,
    interval: float = 1.0,
    output_file: Optional[str] = None
) -> List[VESCTelemetryPoint]:
    """Quick helper to read telemetry data from VESC.

    Args:
        port: Serial port
        duration: Collection duration in seconds
        interval: Sample interval in seconds
        output_file: Optional file to save data

    Returns:
        List of telemetry points
    """
    collector = connect_vesc(port=port)
    try:
        collector.start_collection(duration=duration, interval=interval)
        data = collector.get_data()

        if output_file:
            collector.save_to_file(output_file)

        return data
    finally:
        collector.disconnect()
