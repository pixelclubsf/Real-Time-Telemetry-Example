"""
Example: Annotate telemetry data for machine learning

This script demonstrates how to load, visualize, and annotate telemetry data
for training machine learning models or analyzing performance.

Usage:
    python examples/annotate_telemetry.py --input vesc_telemetry.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from solar_regatta import calculate_speeds, analyze_performance, plot_all_metrics


def load_telemetry(filepath: str) -> dict:
    """Load telemetry data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def annotate_session(data: dict) -> dict:
    """Interactive annotation of telemetry session."""
    print("\n" + "=" * 70)
    print("TELEMETRY ANNOTATION")
    print("=" * 70)

    annotations = {}

    # Session-level annotations
    annotations['session_type'] = input(
        "Session type (race/practice/test): "
    ).strip() or "practice"

    annotations['conditions'] = input(
        "Conditions (sunny/cloudy/windy/calm): "
    ).strip() or "unknown"

    annotations['boat_config'] = input(
        "Boat configuration notes: "
    ).strip() or ""

    annotations['notes'] = input(
        "Additional notes: "
    ).strip() or ""

    # Performance labels
    print("\nPerformance Labels:")
    annotations['max_speed_achieved'] = input(
        "Max speed achieved (m/s) [optional]: "
    ).strip()

    annotations['battery_issues'] = input(
        "Any battery issues? (yes/no): "
    ).strip().lower() == 'yes'

    annotations['motor_issues'] = input(
        "Any motor issues? (yes/no): "
    ).strip().lower() == 'yes'

    # Event markers
    print("\nEvent Markers:")
    annotations['events'] = []
    while True:
        event_time = input(
            "Event timestamp (HH:MM:SS) or 'done': "
        ).strip()
        if event_time.lower() == 'done':
            break
        event_desc = input("Event description: ").strip()
        if event_time and event_desc:
            annotations['events'].append({
                'time': event_time,
                'description': event_desc
            })

    annotations['annotated_at'] = datetime.now().isoformat()
    annotations['annotated_by'] = input(
        "\nYour name/ID: "
    ).strip() or "unknown"

    return annotations


def save_annotated_data(data: dict, annotations: dict, output_path: str):
    """Save data with annotations."""
    data['annotations'] = annotations

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nAnnotated data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate telemetry data for ML training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input telemetry JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: input_annotated.json)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization plots"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading telemetry from: {args.input}")
    data = load_telemetry(args.input)

    # Show summary
    print("\n" + "=" * 70)
    print("TELEMETRY SUMMARY")
    print("=" * 70)
    print(f"Total data points: {len(data.get('telemetry', []))}")

    if 'metadata' in data:
        print("\nMetadata:")
        for key, value in data['metadata'].items():
            print(f"  {key}: {value}")

    # Extract data for analysis
    telemetry = data.get('telemetry', [])
    if telemetry:
        gps_points = [p['gps_position'] for p in telemetry]
        timestamps = [datetime.fromisoformat(p['timestamp']) for p in telemetry]
        voltages = [p['battery_voltage'] for p in telemetry]
        currents = [p['motor_current'] for p in telemetry]

        # Calculate speeds
        speeds = calculate_speeds(gps_points, timestamps)

        # Analyze performance
        metrics = analyze_performance(speeds, voltages, currents, timestamps)

        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")

        # Visualize if requested
        if args.visualize:
            print("\nGenerating visualizations...")
            plot_all_metrics(speeds, voltages, currents, timestamps, gps_points)

    # Annotate
    print("\n" + "=" * 70)
    print("Begin annotation (press Enter to skip any field)")
    print("=" * 70)

    annotations = annotate_session(data)

    # Save annotated data
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.with_name(
            input_path.stem + "_annotated" + input_path.suffix
        )

    save_annotated_data(data, annotations, str(output_path))

    print("\n" + "=" * 70)
    print("Annotation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
