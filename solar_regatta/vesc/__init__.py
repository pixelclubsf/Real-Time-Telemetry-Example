"""VESC Tool integration for real-time telemetry collection."""

from .collector import (
    VESCDataCollector,
    VESCTelemetryPoint,
    RaceSession,
    connect_vesc,
    read_telemetry,
)
from .protocol import (
    VESCProtocol,
    VESCValues,
    VESCCommand,
    get_fault_description,
)
from .gps import (
    GPSReader,
    GPSFix,
    CombinedTelemetrySource,
)

__all__ = [
    # Collector
    "VESCDataCollector",
    "VESCTelemetryPoint",
    "RaceSession",
    "connect_vesc",
    "read_telemetry",
    # Protocol
    "VESCProtocol",
    "VESCValues",
    "VESCCommand",
    "get_fault_description",
    # GPS
    "GPSReader",
    "GPSFix",
    "CombinedTelemetrySource",
]
