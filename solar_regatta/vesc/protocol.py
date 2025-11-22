"""VESC serial protocol implementation.

This module implements the VESC communication protocol for reading telemetry
data from VESC motor controllers. Based on the VESC firmware protocol
specification.

Protocol Overview:
- Packets are framed with start byte (0x02 for short, 0x03 for long)
- CRC16 checksum for data integrity
- Command IDs identify the request/response type
- Little-endian byte ordering for multi-byte values
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union


class VESCCommand(IntEnum):
    """VESC command IDs for communication protocol."""
    COMM_FW_VERSION = 0
    COMM_JUMP_TO_BOOTLOADER = 1
    COMM_ERASE_NEW_APP = 2
    COMM_WRITE_NEW_APP_DATA = 3
    COMM_GET_VALUES = 4  # Main telemetry command
    COMM_SET_DUTY = 5
    COMM_SET_CURRENT = 6
    COMM_SET_CURRENT_BRAKE = 7
    COMM_SET_RPM = 8
    COMM_SET_POS = 9
    COMM_SET_HANDBRAKE = 10
    COMM_SET_DETECT = 11
    COMM_SET_SERVO_POS = 12
    COMM_SET_MCCONF = 13
    COMM_GET_MCCONF = 14
    COMM_GET_MCCONF_DEFAULT = 15
    COMM_SET_APPCONF = 16
    COMM_GET_APPCONF = 17
    COMM_GET_APPCONF_DEFAULT = 18
    COMM_SAMPLE_PRINT = 19
    COMM_TERMINAL_CMD = 20
    COMM_PRINT = 21
    COMM_ROTOR_POSITION = 22
    COMM_EXPERIMENT_SAMPLE = 23
    COMM_DETECT_MOTOR_PARAM = 24
    COMM_DETECT_MOTOR_R_L = 25
    COMM_DETECT_MOTOR_FLUX_LINKAGE = 26
    COMM_DETECT_ENCODER = 27
    COMM_DETECT_HALL_FOC = 28
    COMM_REBOOT = 29
    COMM_ALIVE = 30
    COMM_GET_DECODED_PPM = 31
    COMM_GET_DECODED_ADC = 32
    COMM_GET_DECODED_CHUK = 33
    COMM_FORWARD_CAN = 34
    COMM_SET_CHUCK_DATA = 35
    COMM_CUSTOM_APP_DATA = 36
    COMM_NRF_START_PAIRING = 37
    COMM_GET_VALUES_SETUP = 50  # Extended telemetry
    COMM_SET_MCCONF_TEMP = 51
    COMM_SET_MCCONF_TEMP_SETUP = 52
    COMM_GET_VALUES_SELECTIVE = 53
    COMM_GET_VALUES_SETUP_SELECTIVE = 54


@dataclass
class VESCValues:
    """Parsed telemetry values from VESC COMM_GET_VALUES response.

    All electrical values are in SI units:
    - Voltages in V
    - Currents in A
    - Temperatures in Celsius
    - RPM as mechanical RPM
    - Power calculated as V * I
    """
    temp_fet: float  # MOSFET temperature (C)
    temp_motor: float  # Motor temperature (C)
    avg_motor_current: float  # Motor phase current (A)
    avg_input_current: float  # Battery current (A)
    avg_id: float  # D-axis current (A)
    avg_iq: float  # Q-axis current (A)
    duty_cycle: float  # Duty cycle (0.0 - 1.0)
    rpm: float  # Electrical RPM
    input_voltage: float  # Battery voltage (V)
    amp_hours: float  # Ah consumed
    amp_hours_charged: float  # Ah charged (regen)
    watt_hours: float  # Wh consumed
    watt_hours_charged: float  # Wh charged (regen)
    tachometer: int  # Tachometer counts
    tachometer_abs: int  # Absolute tachometer
    fault_code: int  # Fault code (0 = no fault)

    @property
    def power_in(self) -> float:
        """Input power from battery (W)."""
        return self.input_voltage * self.avg_input_current

    @property
    def power_motor(self) -> float:
        """Motor power output estimate (W)."""
        return self.input_voltage * self.avg_motor_current * self.duty_cycle

    @property
    def efficiency(self) -> float:
        """Motor efficiency estimate (0.0 - 1.0)."""
        if self.power_in > 0:
            return min(self.power_motor / self.power_in, 1.0)
        return 0.0


@dataclass
class VESCValuesSetup:
    """Extended telemetry from COMM_GET_VALUES_SETUP.

    Includes additional fields for setup and diagnostics.
    """
    temp_fet: float
    temp_motor: float
    avg_motor_current: float
    avg_input_current: float
    duty_cycle: float
    rpm: float
    input_voltage: float
    amp_hours: float
    amp_hours_charged: float
    watt_hours: float
    watt_hours_charged: float
    tachometer: int
    tachometer_abs: int
    fault_code: int
    pid_pos: float
    vesc_id: int
    temp_mos1: float
    temp_mos2: float
    temp_mos3: float
    vd: float
    vq: float


# CRC16 lookup table for VESC protocol (CCITT polynomial 0x1021)
CRC16_TABLE = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0,
]


def crc16(data: bytes) -> int:
    """Calculate CRC16 checksum for VESC protocol.

    Uses CCITT polynomial (0x1021) with lookup table for efficiency.

    Args:
        data: Bytes to checksum

    Returns:
        16-bit CRC value
    """
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFF00) ^ CRC16_TABLE[(crc >> 8) ^ byte]
    return crc & 0xFFFF


def build_packet(command: VESCCommand, payload: bytes = b'') -> bytes:
    """Build a VESC protocol packet.

    Packet structure:
    - Short packet (payload <= 256 bytes):
      [0x02] [len] [payload...] [crc_hi] [crc_lo] [0x03]
    - Long packet (payload > 256 bytes):
      [0x03] [len_hi] [len_lo] [payload...] [crc_hi] [crc_lo] [0x03]

    Args:
        command: VESC command ID
        payload: Additional payload data (default empty)

    Returns:
        Complete packet bytes ready to send
    """
    # Prepend command ID to payload
    data = bytes([command]) + payload
    data_len = len(data)

    # Calculate CRC over the data (command + payload)
    crc = crc16(data)

    if data_len <= 256:
        # Short packet format
        packet = bytes([0x02, data_len]) + data + struct.pack('>H', crc) + bytes([0x03])
    else:
        # Long packet format
        packet = bytes([0x03]) + struct.pack('>H', data_len) + data + struct.pack('>H', crc) + bytes([0x03])

    return packet


def parse_packet(data: bytes) -> Tuple[Optional[int], Optional[bytes]]:
    """Parse a VESC protocol packet.

    Args:
        data: Raw bytes received from VESC

    Returns:
        Tuple of (command_id, payload) if valid, (None, None) if invalid
    """
    if len(data) < 5:
        return None, None

    # Check start byte
    start_byte = data[0]

    if start_byte == 0x02:
        # Short packet
        payload_len = data[1]
        if len(data) < payload_len + 5:
            return None, None

        payload = data[2:2 + payload_len]
        crc_received = struct.unpack('>H', data[2 + payload_len:4 + payload_len])[0]
        end_byte = data[4 + payload_len]

    elif start_byte == 0x03:
        # Long packet
        if len(data) < 3:
            return None, None
        payload_len = struct.unpack('>H', data[1:3])[0]
        if len(data) < payload_len + 6:
            return None, None

        payload = data[3:3 + payload_len]
        crc_received = struct.unpack('>H', data[3 + payload_len:5 + payload_len])[0]
        end_byte = data[5 + payload_len]
    else:
        return None, None

    # Verify end byte
    if end_byte != 0x03:
        return None, None

    # Verify CRC
    crc_calculated = crc16(payload)
    if crc_calculated != crc_received:
        return None, None

    # Extract command ID and remaining payload
    if len(payload) < 1:
        return None, None

    command_id = payload[0]
    return command_id, payload[1:]


def parse_get_values(payload: bytes) -> Optional[VESCValues]:
    """Parse COMM_GET_VALUES response payload.

    The payload contains telemetry data in a fixed format.
    Values are scaled integers that need conversion to real units.

    Args:
        payload: Response payload (after command ID)

    Returns:
        VESCValues dataclass if parsing successful, None otherwise
    """
    if len(payload) < 56:
        return None

    try:
        # Unpack all values according to VESC protocol
        # Format: big-endian, various sizes
        offset = 0

        # Temperature values (scaled by 10)
        temp_fet = struct.unpack_from('>h', payload, offset)[0] / 10.0
        offset += 2
        temp_motor = struct.unpack_from('>h', payload, offset)[0] / 10.0
        offset += 2

        # Current values (scaled by 100)
        avg_motor_current = struct.unpack_from('>i', payload, offset)[0] / 100.0
        offset += 4
        avg_input_current = struct.unpack_from('>i', payload, offset)[0] / 100.0
        offset += 4
        avg_id = struct.unpack_from('>i', payload, offset)[0] / 100.0
        offset += 4
        avg_iq = struct.unpack_from('>i', payload, offset)[0] / 100.0
        offset += 4

        # Duty cycle (scaled by 1000)
        duty_cycle = struct.unpack_from('>h', payload, offset)[0] / 1000.0
        offset += 2

        # RPM (raw value)
        rpm = float(struct.unpack_from('>i', payload, offset)[0])
        offset += 4

        # Voltage (scaled by 10)
        input_voltage = struct.unpack_from('>h', payload, offset)[0] / 10.0
        offset += 2

        # Energy counters (scaled by 10000)
        amp_hours = struct.unpack_from('>i', payload, offset)[0] / 10000.0
        offset += 4
        amp_hours_charged = struct.unpack_from('>i', payload, offset)[0] / 10000.0
        offset += 4
        watt_hours = struct.unpack_from('>i', payload, offset)[0] / 10000.0
        offset += 4
        watt_hours_charged = struct.unpack_from('>i', payload, offset)[0] / 10000.0
        offset += 4

        # Tachometer values (raw)
        tachometer = struct.unpack_from('>i', payload, offset)[0]
        offset += 4
        tachometer_abs = struct.unpack_from('>i', payload, offset)[0]
        offset += 4

        # Fault code
        fault_code = payload[offset]

        return VESCValues(
            temp_fet=temp_fet,
            temp_motor=temp_motor,
            avg_motor_current=avg_motor_current,
            avg_input_current=avg_input_current,
            avg_id=avg_id,
            avg_iq=avg_iq,
            duty_cycle=duty_cycle,
            rpm=rpm,
            input_voltage=input_voltage,
            amp_hours=amp_hours,
            amp_hours_charged=amp_hours_charged,
            watt_hours=watt_hours,
            watt_hours_charged=watt_hours_charged,
            tachometer=tachometer,
            tachometer_abs=tachometer_abs,
            fault_code=fault_code,
        )

    except (struct.error, IndexError):
        return None


class VESCProtocol:
    """High-level interface for VESC serial communication.

    Handles packet building, sending, receiving, and parsing.

    Example:
        >>> import serial
        >>> ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
        >>> protocol = VESCProtocol(ser)
        >>> values = protocol.get_values()
        >>> print(f"Battery: {values.input_voltage}V, Current: {values.avg_input_current}A")
    """

    def __init__(self, serial_port):
        """Initialize protocol handler.

        Args:
            serial_port: Open serial.Serial instance
        """
        self.serial = serial_port
        self._buffer = bytearray()

    def send_command(self, command: VESCCommand, payload: bytes = b'') -> bool:
        """Send a command packet to VESC.

        Args:
            command: Command ID to send
            payload: Optional payload data

        Returns:
            True if send successful, False otherwise
        """
        packet = build_packet(command, payload)
        try:
            self.serial.write(packet)
            return True
        except Exception:
            return False

    def receive_packet(self, timeout: float = 0.1) -> Tuple[Optional[int], Optional[bytes]]:
        """Receive and parse a response packet.

        Reads from serial until a complete valid packet is found or timeout.

        Args:
            timeout: Read timeout in seconds

        Returns:
            Tuple of (command_id, payload) if valid packet received
        """
        old_timeout = self.serial.timeout
        self.serial.timeout = timeout

        try:
            # Read available data
            data = self.serial.read(256)
            if not data:
                return None, None

            self._buffer.extend(data)

            # Try to find and parse a packet
            for i in range(len(self._buffer)):
                if self._buffer[i] in (0x02, 0x03):
                    # Try to parse from this position
                    result = parse_packet(bytes(self._buffer[i:]))
                    if result[0] is not None:
                        # Found valid packet, remove processed bytes
                        # Calculate packet length to remove
                        if self._buffer[i] == 0x02:
                            packet_len = self._buffer[i + 1] + 5
                        else:
                            packet_len = struct.unpack('>H', bytes(self._buffer[i+1:i+3]))[0] + 6
                        self._buffer = self._buffer[i + packet_len:]
                        return result

            # No valid packet found, trim buffer if too large
            if len(self._buffer) > 1024:
                self._buffer = self._buffer[-256:]

            return None, None

        finally:
            self.serial.timeout = old_timeout

    def get_values(self, timeout: float = 0.1) -> Optional[VESCValues]:
        """Request and receive telemetry values from VESC.

        Sends COMM_GET_VALUES command and parses response.

        Args:
            timeout: Response timeout in seconds

        Returns:
            VESCValues if successful, None otherwise
        """
        # Clear any pending data
        self._buffer.clear()
        self.serial.reset_input_buffer()

        # Send request
        if not self.send_command(VESCCommand.COMM_GET_VALUES):
            return None

        # Wait for response
        cmd_id, payload = self.receive_packet(timeout)

        if cmd_id != VESCCommand.COMM_GET_VALUES or payload is None:
            return None

        return parse_get_values(payload)

    def set_current(self, current_amps: float) -> bool:
        """Set motor current setpoint.

        Args:
            current_amps: Target current in Amps

        Returns:
            True if command sent successfully
        """
        # Current is sent as milliamps (int32)
        current_ma = int(current_amps * 1000)
        payload = struct.pack('>i', current_ma)
        return self.send_command(VESCCommand.COMM_SET_CURRENT, payload)

    def set_duty(self, duty: float) -> bool:
        """Set motor duty cycle.

        Args:
            duty: Duty cycle (-1.0 to 1.0)

        Returns:
            True if command sent successfully
        """
        # Duty is sent as scaled int32 (x100000)
        duty_scaled = int(duty * 100000)
        payload = struct.pack('>i', duty_scaled)
        return self.send_command(VESCCommand.COMM_SET_DUTY, payload)

    def set_rpm(self, rpm: float) -> bool:
        """Set motor RPM setpoint.

        Args:
            rpm: Target electrical RPM

        Returns:
            True if command sent successfully
        """
        payload = struct.pack('>i', int(rpm))
        return self.send_command(VESCCommand.COMM_SET_RPM, payload)


# Fault code descriptions
FAULT_CODES = {
    0: "FAULT_CODE_NONE",
    1: "FAULT_CODE_OVER_VOLTAGE",
    2: "FAULT_CODE_UNDER_VOLTAGE",
    3: "FAULT_CODE_DRV",
    4: "FAULT_CODE_ABS_OVER_CURRENT",
    5: "FAULT_CODE_OVER_TEMP_FET",
    6: "FAULT_CODE_OVER_TEMP_MOTOR",
    7: "FAULT_CODE_GATE_DRIVER_OVER_VOLTAGE",
    8: "FAULT_CODE_GATE_DRIVER_UNDER_VOLTAGE",
    9: "FAULT_CODE_MCU_UNDER_VOLTAGE",
    10: "FAULT_CODE_BOOTING_FROM_WATCHDOG_RESET",
    11: "FAULT_CODE_ENCODER_SPI",
    12: "FAULT_CODE_ENCODER_SINCOS_BELOW_MIN_AMPLITUDE",
    13: "FAULT_CODE_ENCODER_SINCOS_ABOVE_MAX_AMPLITUDE",
    14: "FAULT_CODE_FLASH_CORRUPTION",
    15: "FAULT_CODE_HIGH_OFFSET_CURRENT_SENSOR_1",
    16: "FAULT_CODE_HIGH_OFFSET_CURRENT_SENSOR_2",
    17: "FAULT_CODE_HIGH_OFFSET_CURRENT_SENSOR_3",
    18: "FAULT_CODE_UNBALANCED_CURRENTS",
}


def get_fault_description(fault_code: int) -> str:
    """Get human-readable description of fault code.

    Args:
        fault_code: Numeric fault code from VESCValues

    Returns:
        Fault description string
    """
    return FAULT_CODES.get(fault_code, f"UNKNOWN_FAULT_{fault_code}")
