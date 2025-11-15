"""
Example: Collect real-time telemetry data from VESC Tool

This script demonstrates how to connect to a VESC motor controller and collect
telemetry data for analysis and annotation.

Requirements:
    - VESC motor controller connected via USB/Serial
    - pyserial library: pip install pyserial

Usage:
    python examples/collect_vesc_data.py --port /dev/ttyUSB0 --duration 60
"""

import argparse
from solar_regatta.vesc import VESCDataCollector


def main():
    parser = argparse.ArgumentParser(
        description="Collect telemetry data from VESC motor controller"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port where VESC is connected (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baud rate (default: 115200)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Collection duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vesc_telemetry.json",
        help="Output file path (default: vesc_telemetry.json)"
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="",
        help="Annotation/label for this collection session"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("VESC Telemetry Data Collector")
    print("=" * 70)
    print(f"Port: {args.port}")
    print(f"Baudrate: {args.baudrate}")
    print(f"Duration: {args.duration}s")
    print(f"Interval: {args.interval}s")
    print(f"Output: {args.output}")
    if args.annotation:
        print(f"Annotation: {args.annotation}")
    print("=" * 70)

    # Create collector
    collector = VESCDataCollector(port=args.port, baudrate=args.baudrate)

    # Connect to VESC
    if not collector.connect():
        print("Failed to connect to VESC. Please check:")
        print("  - VESC is connected to the specified port")
        print("  - Port permissions are correct (try: sudo chmod 666 /dev/ttyUSB0)")
        print("  - No other program is using the port")
        return 1

    try:
        # Collect data
        collector.start_collection(
            duration=args.duration,
            interval=args.interval,
            annotation=args.annotation
        )

        # Save to file
        collector.save_to_file(args.output)

        # Print summary
        data = collector.get_data()
        print("\n" + "=" * 70)
        print("COLLECTION SUMMARY")
        print("=" * 70)
        print(f"Total points collected: {len(data)}")
        if data:
            voltages = [p.battery_voltage for p in data]
            currents = [p.motor_current for p in data]
            print(f"Voltage range: {min(voltages):.2f}V - {max(voltages):.2f}V")
            print(f"Current range: {min(currents):.2f}A - {max(currents):.2f}A")
        print("=" * 70)

    except Exception as e:
        print(f"Error during collection: {e}")
        return 1
    finally:
        collector.disconnect()

    return 0


if __name__ == "__main__":
    exit(main())
