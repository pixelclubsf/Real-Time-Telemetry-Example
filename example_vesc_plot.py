"""
Example: How to plot speed vs time with GPS coordinates from VESC data

This script demonstrates:
- Calculating speeds from GPS coordinates
- Generating performance metrics
- Creating visualizations
- Saving results to files

Run with: python example_vesc_plot.py
"""

import argparse
from datetime import datetime
from pathlib import Path

from solar_regatta import (
    calculate_speeds,
    plot_speed_vs_time,
    plot_with_coordinates,
    analyze_performance,
    generate_sample_vesc_data,
)


def run_simple_example():
    """Run a simple example with minimal data."""
    print("=" * 70)
    print("SIMPLE EXAMPLE: Basic Speed Calculation")
    print("=" * 70)

    # Example: Using MGRS coordinates with timestamps
    gps_points = [
        "10SEG1234567890",  # MGRS format examples
        "10SEG1234567891",
        "10SEG1234567892",
        "10SEG1234567893",
    ]

    timestamps = [
        datetime(2025, 10, 30, 10, 0, 0),
        datetime(2025, 10, 30, 10, 0, 10),
        datetime(2025, 10, 30, 10, 0, 20),
        datetime(2025, 10, 30, 10, 0, 30),
    ]

    # Calculate speeds
    speeds = calculate_speeds(gps_points, timestamps)

    # Print results
    print(f"\nData points: {len(gps_points)}")
    print(f"Time span: {(timestamps[-1] - timestamps[0]).total_seconds():.0f} seconds")
    print(f"\nCalculated speeds:")
    for i, speed in enumerate(speeds):
        print(f"  Point {i+1}: {speed:.2f} m/s")

    if speeds:
        print(f"\nMax speed: {max(speeds):.2f} m/s")
        print(f"Average speed: {sum(speeds)/len(speeds):.2f} m/s")

    print("\nImages saved to current directory")
    print("=" * 70)

    return speeds, timestamps, gps_points


def run_full_example(duration=60, interval=5, save_plots=True):
    """Run a full example with generated sample data."""
    print("=" * 70)
    print("FULL EXAMPLE: Complete Telemetry Analysis")
    print("=" * 70)
    print(f"Duration: {duration}s | Interval: {interval}s")
    print("=" * 70)

    # Generate sample VESC data
    print("\n[1/5] Generating sample VESC telemetry data...")
    gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
        generate_sample_vesc_data(duration_seconds=duration, interval=interval)

    print(f"  Generated {len(gps_points)} data points")

    # Calculate speeds from GPS
    print("\n[2/5] Calculating speeds from GPS coordinates...")
    speeds = calculate_speeds(gps_points, timestamps)
    print(f"  Calculated {len(speeds)} speed measurements")

    # Analyze performance
    print("\n[3/5] Analyzing performance metrics...")
    metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Duration:         {metrics['duration']:.1f} seconds ({metrics['duration']/60:.1f} minutes)")
    print(f"Distance:         {metrics['distance']:.1f} meters")
    print(f"\nSpeed:")
    print(f"  Maximum:        {metrics['max_speed']:.2f} m/s")
    print(f"  Minimum:        {metrics['min_speed']:.2f} m/s")
    print(f"  Average:        {metrics['avg_speed']:.2f} m/s")
    print(f"\nBattery Voltage:")
    print(f"  Maximum:        {metrics['max_voltage']:.2f} V")
    print(f"  Minimum:        {metrics['min_voltage']:.2f} V")
    print(f"\nMotor Current:")
    print(f"  Maximum:        {metrics['max_current']:.2f} A")
    print(f"  Average:        {metrics['avg_current']:.2f} A")
    print("=" * 70)

    if save_plots:
        print("\n[4/5] Generating visualizations...")

        # Save speed vs time plot
        fig1 = plot_speed_vs_time(speeds, timestamps, title="Solar Boat Speed Analysis")
        output1 = Path('solar_boat_speed.png')
        fig1.savefig(output1, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output1}")

        # Save speed with GPS coordinates
        fig2, ax = plot_with_coordinates(
            speeds,
            timestamps,
            gps_points,
            title="Speed vs Time with GPS Data"
        )
        output2 = Path('solar_boat_speed_gps.png')
        fig2.savefig(output2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output2}")

        print("\n[5/5] Complete!")
        print("\nOutput files:")
        print(f"  - {output1.absolute()}")
        print(f"  - {output2.absolute()}")
    else:
        print("\n[4/5] Skipping plot generation (use --save-plots to enable)")
        print("[5/5] Complete!")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Example script for VESC telemetry analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example_vesc_plot.py --simple
  python example_vesc_plot.py --full --duration 300
  python example_vesc_plot.py --full --no-save-plots
        """
    )

    parser.add_argument(
        '--simple',
        action='store_true',
        help='Run simple example with minimal data'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full example with generated sample data'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration for full example in seconds (default: 60)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Sample interval in seconds (default: 5)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save plot images (default: True)'
    )
    parser.add_argument(
        '--no-save-plots',
        action='store_false',
        dest='save_plots',
        help='Do not save plot images'
    )

    args = parser.parse_args()

    # If no mode specified, run both
    if not args.simple and not args.full:
        args.simple = True
        args.full = True

    try:
        if args.simple:
            run_simple_example()
            if args.full:
                print("\n")

        if args.full:
            run_full_example(
                duration=args.duration,
                interval=args.interval,
                save_plots=args.save_plots
            )

        print("\n✓ Example completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
