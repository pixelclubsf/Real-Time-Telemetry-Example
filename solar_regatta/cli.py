"""Command line helpers for Solar Regatta telemetry analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.analysis import (
    analyze_performance,
    calculate_speeds,
    generate_sample_vesc_data,
)
from .ml import evaluate_model, prepare_training_data, train_speed_model


def _to_serializable_array(values):
    return [float(v) for v in values]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate telemetry, train a predictive model, and report metrics.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Race duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Sampling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        help="Optional path to store the learned coefficients as JSON",
    )
    parser.add_argument(
        "--export-predictions",
        type=Path,
        help="Path to save predicted speed curve as JSON",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    gps_points, timestamps, speeds_raw, battery_voltage, motor_current = generate_sample_vesc_data(
        duration_seconds=args.duration,
        interval=args.interval,
    )

    speeds = calculate_speeds(gps_points, timestamps)
    metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)
    print("Performance summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    model = train_speed_model(speeds, battery_voltage, motor_current, timestamps)
    feature_matrix, targets, _ = prepare_training_data(
        speeds, battery_voltage, motor_current, timestamps
    )
    regression_metrics = evaluate_model(model, feature_matrix, targets)
    print("\nLinear regression diagnostics:")
    for key, value in regression_metrics.items():
        print(f"  {key}: {value:.5f}")

    if args.save_model:
        payload = {
            "feature_names": model.feature_names,
            "coefficients": _to_serializable_array(model.coefficients),
            "intercept": float(model.intercept),
        }
        args.save_model.write_text(json.dumps(payload, indent=2))
        print(f"\nModel saved to {args.save_model}")

    if args.export_predictions:
        predictions = model.predict(feature_matrix).tolist()
        args.export_predictions.write_text(json.dumps({"predicted_speeds": predictions}, indent=2))
        print(f"Predictions exported to {args.export_predictions}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
