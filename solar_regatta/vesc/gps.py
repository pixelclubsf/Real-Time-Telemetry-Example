"""GPS Integration for Solar Regatta.

This module provides GPS position and speed data from various sources:
- Serial GPS modules (NMEA 0183 protocol)
- USB GPS dongles
- Simulated GPS for testing

Supports standard NMEA sentences:
- $GPGGA: Position fix data
- $GPRMC: Recommended minimum data (position + speed + course)
- $GPVTG: Course over ground and speed
"""
from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Callable, Tuple
from queue import Queue, Empty
import math

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import mgrs
    MGRS_AVAILABLE = True
except ImportError:
    MGRS_AVAILABLE = False


@dataclass
class GPSFix:
    """GPS position and velocity data."""

    timestamp: datetime
    latitude: float  # Decimal degrees (positive = N)
    longitude: float  # Decimal degrees (positive = E)
    altitude: float  # Meters above sea level
    speed: float  # m/s (ground speed)
    course: float  # Degrees from true north (0-360)
    fix_quality: int  # 0=invalid, 1=GPS, 2=DGPS, 4=RTK
    satellites: int  # Number of satellites in use
    hdop: float  # Horizontal dilution of precision

    @property
    def speed_knots(self) -> float:
        """Speed in knots."""
        return self.speed * 1.94384

    @property
    def speed_kph(self) -> float:
        """Speed in km/h."""
        return self.speed * 3.6

    @property
    def mgrs_position(self) -> str:
        """Position in MGRS format."""
        if not MGRS_AVAILABLE:
            return f"{self.latitude:.6f},{self.longitude:.6f}"
        try:
            m = mgrs.MGRS()
            return m.toMGRS(self.latitude, self.longitude, MGRSPrecision=4)
        except Exception:
            return f"{self.latitude:.6f},{self.longitude:.6f}"

    def distance_to(self, other: 'GPSFix') -> float:
        """Calculate distance to another point in meters (Haversine).

        Args:
            other: Another GPSFix

        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters

        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlat = math.radians(other.latitude - self.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c


def parse_nmea_checksum(sentence: str) -> bool:
    """Validate NMEA sentence checksum.

    Args:
        sentence: NMEA sentence including $ and *XX checksum

    Returns:
        True if checksum is valid
    """
    if '*' not in sentence:
        return False

    try:
        data, checksum_hex = sentence.rsplit('*', 1)
        data = data.lstrip('$')
        checksum_hex = checksum_hex.strip()

        calculated = 0
        for char in data:
            calculated ^= ord(char)

        return calculated == int(checksum_hex, 16)
    except (ValueError, IndexError):
        return False


def parse_gprmc(sentence: str) -> Optional[GPSFix]:
    """Parse GPRMC (Recommended Minimum) sentence.

    Format: $GPRMC,hhmmss.ss,A,llll.ll,N,yyyyy.yy,E,knots,course,ddmmyy,mv,mvE*cs

    Args:
        sentence: NMEA GPRMC sentence

    Returns:
        GPSFix if valid, None otherwise
    """
    if not sentence.startswith('$GPRMC') and not sentence.startswith('$GNRMC'):
        return None

    if not parse_nmea_checksum(sentence):
        return None

    try:
        parts = sentence.split('*')[0].split(',')
        if len(parts) < 10:
            return None

        # Status check (A=valid, V=invalid)
        if parts[2] != 'A':
            return None

        # Time
        time_str = parts[1]
        date_str = parts[9]
        if len(time_str) >= 6 and len(date_str) >= 6:
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(float(time_str[4:]))
            day = int(date_str[0:2])
            month = int(date_str[2:4])
            year = 2000 + int(date_str[4:6])
            timestamp = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Latitude: ddmm.mmmm
        lat_raw = float(parts[3])
        lat_deg = int(lat_raw / 100)
        lat_min = lat_raw - (lat_deg * 100)
        latitude = lat_deg + lat_min / 60
        if parts[4] == 'S':
            latitude = -latitude

        # Longitude: dddmm.mmmm
        lon_raw = float(parts[5])
        lon_deg = int(lon_raw / 100)
        lon_min = lon_raw - (lon_deg * 100)
        longitude = lon_deg + lon_min / 60
        if parts[6] == 'W':
            longitude = -longitude

        # Speed in knots -> m/s
        speed_knots = float(parts[7]) if parts[7] else 0.0
        speed = speed_knots * 0.514444

        # Course
        course = float(parts[8]) if parts[8] else 0.0

        return GPSFix(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude=0.0,  # Not in GPRMC
            speed=speed,
            course=course,
            fix_quality=1,  # Assume GPS fix if valid
            satellites=0,  # Not in GPRMC
            hdop=0.0,  # Not in GPRMC
        )

    except (ValueError, IndexError):
        return None


def parse_gpgga(sentence: str) -> Optional[GPSFix]:
    """Parse GPGGA (Global Positioning System Fix Data) sentence.

    Format: $GPGGA,hhmmss.ss,llll.ll,N,yyyyy.yy,E,q,nn,hdop,alt,M,height,M,age,id*cs

    Args:
        sentence: NMEA GPGGA sentence

    Returns:
        GPSFix if valid, None otherwise
    """
    if not sentence.startswith('$GPGGA') and not sentence.startswith('$GNGGA'):
        return None

    if not parse_nmea_checksum(sentence):
        return None

    try:
        parts = sentence.split('*')[0].split(',')
        if len(parts) < 12:
            return None

        # Fix quality (0=invalid, 1=GPS, 2=DGPS, etc.)
        fix_quality = int(parts[6]) if parts[6] else 0
        if fix_quality == 0:
            return None

        # Time (no date in GGA)
        time_str = parts[1]
        now = datetime.now(timezone.utc)
        if len(time_str) >= 6:
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(float(time_str[4:]))
            timestamp = now.replace(hour=hour, minute=minute, second=second)
        else:
            timestamp = now

        # Latitude
        lat_raw = float(parts[2])
        lat_deg = int(lat_raw / 100)
        lat_min = lat_raw - (lat_deg * 100)
        latitude = lat_deg + lat_min / 60
        if parts[3] == 'S':
            latitude = -latitude

        # Longitude
        lon_raw = float(parts[4])
        lon_deg = int(lon_raw / 100)
        lon_min = lon_raw - (lon_deg * 100)
        longitude = lon_deg + lon_min / 60
        if parts[5] == 'W':
            longitude = -longitude

        # Satellites and HDOP
        satellites = int(parts[7]) if parts[7] else 0
        hdop = float(parts[8]) if parts[8] else 99.9

        # Altitude
        altitude = float(parts[9]) if parts[9] else 0.0

        return GPSFix(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            speed=0.0,  # Not in GGA
            course=0.0,  # Not in GGA
            fix_quality=fix_quality,
            satellites=satellites,
            hdop=hdop,
        )

    except (ValueError, IndexError):
        return None


def parse_gpvtg(sentence: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse GPVTG (Course Over Ground and Ground Speed) sentence.

    Format: $GPVTG,course,T,course,M,speed,N,speed,K,mode*cs

    Args:
        sentence: NMEA GPVTG sentence

    Returns:
        Tuple of (course_degrees, speed_ms) or (None, None)
    """
    if not sentence.startswith('$GPVTG') and not sentence.startswith('$GNVTG'):
        return None, None

    if not parse_nmea_checksum(sentence):
        return None, None

    try:
        parts = sentence.split('*')[0].split(',')
        if len(parts) < 8:
            return None, None

        # Course (true)
        course = float(parts[1]) if parts[1] else None

        # Speed in km/h -> m/s
        speed_kph = float(parts[7]) if parts[7] else None
        speed = speed_kph / 3.6 if speed_kph is not None else None

        return course, speed

    except (ValueError, IndexError):
        return None, None


class GPSReader:
    """Read GPS data from serial port.

    Parses NMEA sentences and provides current position/velocity.

    Example:
        >>> gps = GPSReader(port='/dev/ttyUSB1')
        >>> gps.start()
        >>> fix = gps.get_fix()
        >>> print(f"Position: {fix.latitude}, {fix.longitude}")
        >>> print(f"Speed: {fix.speed} m/s")
        >>> gps.stop()
    """

    def __init__(
        self,
        port: str = '/dev/ttyUSB1',
        baudrate: int = 9600,
        simulate: bool = False
    ):
        """Initialize GPS reader.

        Args:
            port: Serial port for GPS module
            baudrate: Baud rate (typically 4800, 9600, or 115200)
            simulate: Generate simulated GPS data
        """
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate

        self.serial_conn: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fix_queue: Queue = Queue(maxsize=100)
        self._current_fix: Optional[GPSFix] = None
        self._callbacks: List[Callable[[GPSFix], None]] = []

        # For merging GGA and RMC data
        self._last_gga: Optional[GPSFix] = None
        self._last_rmc: Optional[GPSFix] = None

        # Simulation state
        self._sim_lat = 37.7749  # San Francisco
        self._sim_lon = -122.4194
        self._sim_heading = 45.0

        if not simulate and not SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial is required for GPS integration. "
                "Install with: pip install pyserial"
            )

    @staticmethod
    def list_ports() -> List[str]:
        """List available serial ports that might be GPS devices."""
        if not SERIAL_AVAILABLE:
            return []

        gps_ports = []
        for port in serial.tools.list_ports.comports():
            # Common GPS device identifiers
            desc_lower = port.description.lower()
            if any(x in desc_lower for x in ['gps', 'gnss', 'u-blox', 'nmea', 'usb-serial']):
                gps_ports.append(port.device)
            else:
                gps_ports.append(port.device)  # Include all for now
        return gps_ports

    def connect(self) -> bool:
        """Connect to GPS device.

        Returns:
            True if connection successful
        """
        if self.simulate:
            print("GPS running in simulation mode")
            return True

        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            print(f"Connected to GPS on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to GPS: {e}")
            return False

    def disconnect(self):
        """Disconnect from GPS."""
        self.stop()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.serial_conn = None

    def add_callback(self, callback: Callable[[GPSFix], None]):
        """Add callback for new GPS fixes."""
        self._callbacks.append(callback)

    def start(self):
        """Start reading GPS data in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reading GPS data."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _read_loop(self):
        """Background thread for reading GPS data."""
        while self._running:
            try:
                if self.simulate:
                    fix = self._generate_simulated_fix()
                    time.sleep(1.0)  # Simulate 1Hz GPS
                else:
                    fix = self._read_nmea_fix()

                if fix:
                    self._current_fix = fix
                    try:
                        self._fix_queue.put_nowait(fix)
                    except Exception:
                        pass  # Queue full, drop oldest

                    for callback in self._callbacks:
                        try:
                            callback(fix)
                        except Exception:
                            pass

            except Exception as e:
                print(f"GPS read error: {e}")
                time.sleep(0.1)

    def _read_nmea_fix(self) -> Optional[GPSFix]:
        """Read and parse NMEA sentences from serial port."""
        if not self.serial_conn:
            return None

        try:
            line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
            if not line:
                return None

            # Try to parse different sentence types
            if line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                fix = parse_gprmc(line)
                if fix:
                    self._last_rmc = fix
                    # Merge with GGA data if available
                    if self._last_gga:
                        fix.altitude = self._last_gga.altitude
                        fix.fix_quality = self._last_gga.fix_quality
                        fix.satellites = self._last_gga.satellites
                        fix.hdop = self._last_gga.hdop
                    return fix

            elif line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                fix = parse_gpgga(line)
                if fix:
                    self._last_gga = fix
                    # Merge with RMC data if available
                    if self._last_rmc:
                        fix.speed = self._last_rmc.speed
                        fix.course = self._last_rmc.course
                    return fix

            elif line.startswith('$GPVTG') or line.startswith('$GNVTG'):
                course, speed = parse_gpvtg(line)
                if self._current_fix and (course is not None or speed is not None):
                    if course is not None:
                        self._current_fix.course = course
                    if speed is not None:
                        self._current_fix.speed = speed

        except Exception:
            pass

        return None

    def _generate_simulated_fix(self) -> GPSFix:
        """Generate simulated GPS fix for testing."""
        import random

        # Simulate boat moving
        speed = 1.5 + random.gauss(0, 0.2)  # ~1.5 m/s with noise
        self._sim_heading += random.gauss(0, 2)  # Slight heading changes

        # Update position
        distance = speed * 1.0  # 1 second interval
        self._sim_lat += distance * math.cos(math.radians(self._sim_heading)) / 111000
        self._sim_lon += distance * math.sin(math.radians(self._sim_heading)) / (111000 * math.cos(math.radians(self._sim_lat)))

        return GPSFix(
            timestamp=datetime.now(timezone.utc),
            latitude=self._sim_lat,
            longitude=self._sim_lon,
            altitude=0.0,
            speed=speed,
            course=self._sim_heading % 360,
            fix_quality=1,
            satellites=8 + random.randint(-2, 2),
            hdop=1.2 + random.gauss(0, 0.2),
        )

    def get_fix(self, timeout: float = 1.0) -> Optional[GPSFix]:
        """Get latest GPS fix.

        Args:
            timeout: Max time to wait for fix

        Returns:
            Latest GPSFix or None
        """
        try:
            return self._fix_queue.get(timeout=timeout)
        except Empty:
            return self._current_fix

    def get_current_fix(self) -> Optional[GPSFix]:
        """Get current fix without waiting."""
        return self._current_fix


class CombinedTelemetrySource:
    """Combine VESC and GPS data into synchronized telemetry.

    Merges data from VESCDataCollector and GPSReader into
    unified telemetry points with both electrical and position data.
    """

    def __init__(self, vesc_collector, gps_reader):
        """Initialize combined source.

        Args:
            vesc_collector: VESCDataCollector instance
            gps_reader: GPSReader instance
        """
        self.vesc = vesc_collector
        self.gps = gps_reader
        self._last_gps_fix: Optional[GPSFix] = None

        # Register GPS callback
        self.gps.add_callback(self._on_gps_update)

    def _on_gps_update(self, fix: GPSFix):
        """Store latest GPS fix."""
        self._last_gps_fix = fix

    def read_combined(self):
        """Read combined VESC + GPS telemetry point.

        Returns:
            VESCTelemetryPoint with GPS data merged in
        """
        # Get GPS data
        gps_position = ""
        speed_gps = 0.0

        if self._last_gps_fix:
            gps_position = self._last_gps_fix.mgrs_position
            speed_gps = self._last_gps_fix.speed

        # Read VESC with GPS data
        return self.vesc.read_telemetry_point(
            gps_position=gps_position,
            speed_gps=speed_gps
        )

    def start(self):
        """Start both data sources."""
        self.gps.start()

    def stop(self):
        """Stop both data sources."""
        self.gps.stop()
