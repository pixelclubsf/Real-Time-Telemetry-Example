#!/usr/bin/env python3
"""
Solar Regatta - Comprehensive Dashboard
Multi-tab Gradio interface for VESC data collection, model training, inference, and analysis.
Deployed on Hugging Face Spaces.

This is the main entry point for the Hugging Face Space.
"""

import sys
from pathlib import Path

# Add current directory to Python path to find local solar_regatta module
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import gradio as gr
import numpy as np
import json
import io
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# Import Solar Regatta modules - handle import errors gracefully
try:
    from solar_regatta import (
        generate_sample_vesc_data,
        calculate_speeds,
        analyze_performance,
    )
    from solar_regatta.ml.models import PerformanceModel
except Exception as e:
    print(f"Warning: Core Solar Regatta modules failed to import: {e}")

try:
    from solar_regatta.ml import (
        train_speed_model,
        prepare_training_data,
        count_model_parameters,
        FeatureEngineer,
    )
except Exception as e:
    print(f"Warning: ML modules failed to import: {e}")

try:
    from solar_regatta.viz.plotly_charts import (
        create_speed_plot,
        create_voltage_plot,
        create_current_plot,
        create_efficiency_plot,
        create_gps_path_plot,
    )
except Exception as e:
    print(f"Warning: Plotly modules failed to import: {e}")

# Try to import tree models
try:
    from solar_regatta.ml import RandomForestSpeedModel
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from solar_regatta.ml import XGBoostSpeedModel
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from solar_regatta.vesc import VESCDataCollector, GPSReader, VESCTelemetryPoint
    VESC_AVAILABLE = True
except Exception:
    VESC_AVAILABLE = False

try:
    from solar_regatta.ml.system_id import SystemIdentifier, BoatParameters, calibrate_world_model
    SYSTEM_ID_AVAILABLE = True
except Exception:
    SYSTEM_ID_AVAILABLE = False

try:
    from solar_regatta.ml.world_model import (
        create_default_world_model,
        simulate_race,
        BoatState,
        PhysicsParameters
    )
    from solar_regatta.viz.world_model_viz import (
        plot_trajectory_2d,
        plot_state_evolution,
        plot_strategy_comparison,
        plot_uncertainty_bands,
        create_control_visualization
    )
    WORLD_MODEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: World model not available: {e}")
    WORLD_MODEL_AVAILABLE = False


# ============================================================================
# GLOBAL STATE FOR LIVE TELEMETRY
# ============================================================================

# Global collector instance for live telemetry
live_collector = None
live_gps = None
live_data_buffer = []
MAX_BUFFER_SIZE = 500

def get_available_ports():
    """Get list of available serial ports."""
    if VESC_AVAILABLE:
        ports = VESCDataCollector.list_ports()
        return [p['device'] for p in ports] if ports else ["No ports found"]
    return ["VESC module not available"]

def get_gps_ports():
    """Get list of available GPS ports."""
    if VESC_AVAILABLE:
        ports = GPSReader.list_ports()
        return ports if ports else ["No GPS ports found"]
    return ["GPS module not available"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_parameter_table(model, X=None):
    """Format parameter information as HTML table."""
    info = count_model_parameters(model)

    html = f"""
    <div style="font-family: monospace; padding: 20px;">
        <h2>Model Parameter Summary</h2>
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #f0f0f0;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Model Type</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{info['model_type']}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Total Parameters</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{info['total_parameters']:,}</td>
            </tr>
    """

    if X is not None:
        html += f"""
            <tr style="background-color: #f0f0f0;">
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Training Samples</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{X.shape[0]:,}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Features</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd;">{X.shape[1]:,}</td>
            </tr>
        """

    html += "</table>"

    # Coefficients table for linear models
    if isinstance(model, PerformanceModel):
        html += """
        <h3>Coefficients</h3>
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #4CAF50; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Feature</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Coefficient</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Abs Value</th>
            </tr>
        """

        for name, coef in zip(model.feature_names, model.coefficients):
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{name}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{coef:.6f}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{abs(coef):.6f}</td>
            </tr>
            """

        html += f"""
            <tr style="background-color: #f0f0f0;">
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>Intercept</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><strong>{model.intercept:.6f}</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><strong>{abs(model.intercept):.6f}</strong></td>
            </tr>
        </table>
        """

        # Feature importance
        importance_order = np.argsort(np.abs(model.coefficients))[::-1]
        html += """
        <h3>Feature Importance</h3>
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #2196F3; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Rank</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Feature</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Importance</th>
            </tr>
        """

        for rank, idx in enumerate(importance_order, 1):
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{rank}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{model.feature_names[idx]}</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{abs(model.coefficients[idx]):.6f}</td>
            </tr>
            """

        html += "</table>"

    html += "</div>"
    return html


def create_feature_importance_plot(model):
    """Create feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(model, PerformanceModel):
        importance = np.abs(model.coefficients)
        features = model.feature_names

        # Sort by importance
        sorted_idx = np.argsort(importance)
        pos = np.arange(len(sorted_idx))

        ax.barh(pos, importance[sorted_idx], color='#4CAF50')
        ax.set_yticks(pos)
        ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title('Feature Importance (Linear Model)')
        ax.grid(axis='x', alpha=0.3)

    elif hasattr(model, 'get_feature_importance'):
        try:
            importance = model.get_feature_importance()
            features = [f"Feature {i}" for i in range(len(importance))]

            # Sort by importance
            sorted_idx = np.argsort(importance)[-10:]  # Top 10
            pos = np.arange(len(sorted_idx))

            ax.barh(pos, importance[sorted_idx], color='#2196F3')
            ax.set_yticks(pos)
            ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=9)
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Feature Importances')
            ax.grid(axis='x', alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Feature importance not available',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


# ============================================================================
# TAB 1: VESC DATA COLLECTION
# ============================================================================

def collect_vesc_data_simulation(duration, interval):
    """Simulate VESC data collection (since real hardware may not be available)."""
    try:
        gps, timestamps, speeds_raw, voltage, current = \
            generate_sample_vesc_data(duration_seconds=duration, interval=interval)

        # Format data for display
        data = []
        for i, (gps_pt, ts, speed, volt, curr) in enumerate(
            zip(gps, timestamps, speeds_raw, voltage, current)
        ):
            data.append({
                "Sample": i + 1,
                "Timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "GPS (MGRS)": gps_pt,
                "Speed (m/s)": f"{speed:.2f}",
                "Voltage (V)": f"{volt:.2f}",
                "Current (A)": f"{curr:.2f}",
            })

        json_data = json.dumps(data, indent=2)

        # Create summary
        summary = f"""
        ### Collection Summary
        - **Duration:** {duration} seconds
        - **Samples Collected:** {len(gps)}
        - **Sampling Interval:** {interval} seconds
        - **Speed Range:** {min(speeds_raw):.2f} - {max(speeds_raw):.2f} m/s
        - **Voltage Range:** {min(voltage):.2f} - {max(voltage):.2f} V
        - **Current Range:** {min(current):.2f} - {max(current):.2f} A
        """

        return json_data, summary
    except Exception as e:
        return f"Error: {str(e)}", f"Error collecting data: {str(e)}"


def export_collection_csv(json_str):
    """Export collection data as CSV."""
    try:
        data = json.loads(json_str)
        csv_lines = ["Sample,Timestamp,GPS (MGRS),Speed (m/s),Voltage (V),Current (A)"]
        for row in data:
            csv_lines.append(
                f"{row['Sample']},{row['Timestamp']},{row['GPS (MGRS)']},{row['Speed (m/s)']},{row['Voltage (V)']},{row['Current (A)']}"
            )
        csv_content = "\n".join(csv_lines)
        return csv_content
    except Exception as e:
        return f"Error exporting: {str(e)}"


# ============================================================================
# TAB 2: MODEL TRAINING
# ============================================================================

def train_and_analyze(model_type, duration, n_samples, use_feature_engineering):
    """Train model and return parameter analysis."""
    try:
        # Generate data
        interval = max(1, duration // n_samples)
        gps, timestamps, speeds_raw, voltage, current = \
            generate_sample_vesc_data(duration_seconds=duration, interval=interval)

        speeds = calculate_speeds(gps, timestamps)

        # Prepare training data
        if use_feature_engineering:
            engineer = FeatureEngineer(
                rolling_windows=[3, 5],
                lag_features=2,
                include_derivatives=True,
                include_physics=True
            )
            X = engineer.fit_transform(speeds, voltage, current)
            y = speeds[engineer.feature_delay:]

            # Train linear model with engineered features
            X_aug = np.column_stack([X, np.ones(len(X))])
            solution, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            coefficients = solution[:-1]
            intercept = solution[-1]

            model = PerformanceModel(
                coefficients=coefficients,
                intercept=intercept,
                feature_names=engineer.get_feature_names()
            )

            # Calculate R² on training data
            predictions = X @ coefficients + intercept
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            X, y, _ = prepare_training_data(speeds, voltage, current, timestamps)

            if model_type == "Linear Regression":
                model = train_speed_model(speeds, voltage, current, timestamps)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            elif model_type == "Random Forest" and SKLEARN_AVAILABLE:
                model = RandomForestSpeedModel(n_estimators=50, max_depth=10)
                model.fit(X, y)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                model = XGBoostSpeedModel(n_estimators=50, max_depth=6)
                model.fit(X, y, verbose=False)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            else:
                model = train_speed_model(speeds, voltage, current, timestamps)
                predictions = model.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Generate outputs
        param_table = format_parameter_table(model, X)
        feature_plot = create_feature_importance_plot(model)

        # Summary stats
        info = count_model_parameters(model)
        mae = np.mean(np.abs(y - predictions))
        rmse = np.sqrt(np.mean((y - predictions) ** 2))

        summary = f"""
        ### Model Summary
        - **Type:** {info['model_type']}
        - **Total Parameters:** {info['total_parameters']:,}
        - **Training Samples:** {X.shape[0]:,}
        - **Features:** {X.shape[1]:,}
        - **R² Score:** {r2_score:.4f}
        - **MAE:** {mae:.4f} m/s
        - **RMSE:** {rmse:.4f} m/s
        """

        return param_table, feature_plot, summary
    except Exception as e:
        error_msg = f"Error training model: {str(e)}"
        return error_msg, None, error_msg


# ============================================================================
# TAB 3: SPEED PREDICTION
# ============================================================================

# Global state for saved models
saved_models = {}

def train_inference_model(model_type, duration, n_samples, use_feature_engineering):
    """Train and save model for inference."""
    try:
        interval = max(1, duration // n_samples)
        gps, timestamps, speeds_raw, voltage, current = \
            generate_sample_vesc_data(duration_seconds=duration, interval=interval)

        speeds = calculate_speeds(gps, timestamps)

        if use_feature_engineering:
            engineer = FeatureEngineer(
                rolling_windows=[3, 5],
                lag_features=2,
                include_derivatives=True,
                include_physics=True
            )
            X = engineer.fit_transform(speeds, voltage, current)
            y = speeds[engineer.feature_delay:]
            X_aug = np.column_stack([X, np.ones(len(X))])
            solution, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            coefficients = solution[:-1]
            intercept = solution[-1]
            model = PerformanceModel(
                coefficients=coefficients,
                intercept=intercept,
                feature_names=engineer.get_feature_names()
            )
        else:
            X, y, _ = prepare_training_data(speeds, voltage, current, timestamps)
            if model_type == "Linear Regression":
                model = train_speed_model(speeds, voltage, current, timestamps)
            elif model_type == "Random Forest" and SKLEARN_AVAILABLE:
                model = RandomForestSpeedModel(n_estimators=50, max_depth=10)
                model.fit(X, y)
            elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                model = XGBoostSpeedModel(n_estimators=50, max_depth=6)
                model.fit(X, y, verbose=False)
            else:
                model = train_speed_model(speeds, voltage, current, timestamps)

        saved_models['current'] = {
            'model': model,
            'type': model_type,
            'engineered': use_feature_engineering
        }

        return "✓ Model trained and ready for predictions!", f"Model Type: {model_type}"
    except Exception as e:
        return f"Error: {str(e)}", "Failed to train model"


def predict_speed(voltage, current, timestamp_str, prev_speed=None):
    """Make speed prediction using saved model."""
    try:
        if 'current' not in saved_models:
            return "No model trained yet! Train a model first.", "N/A"

        model_info = saved_models['current']
        model = model_info['model']

        # Create feature vector
        if prev_speed is None:
            prev_speed = 0.0

        features = np.array([[prev_speed, voltage, current, 1.0]])

        prediction = model.predict(features)[0]

        return f"**Predicted Speed:** {prediction:.3f} m/s", f"Input - Voltage: {voltage}V, Current: {current}A"
    except Exception as e:
        return f"Error: {str(e)}", "Prediction failed"


# ============================================================================
# TAB 4: DATA ANALYSIS DASHBOARD
# ============================================================================

def analyze_uploaded_data(file):
    """Analyze uploaded telemetry data and create visualizations."""
    try:
        if file is None:
            return None, None, None, None, None, "Please upload a telemetry file"

        # Load JSON data
        file_content = file.read().decode('utf-8')
        data = json.loads(file_content)

        # Parse telemetry data
        gps_points = [d.get("gps_position", f"10SEG{i:010d}") for i, d in enumerate(data)]
        speeds = [float(d.get("speed_gps", 0)) for d in data]
        voltages = [float(d.get("battery_voltage", 13)) for d in data]
        currents = [float(d.get("motor_current", 5)) for d in data]

        # Create visualizations
        fig_speed = create_speed_plot(speeds, list(range(len(speeds))))
        fig_voltage = create_voltage_plot(voltages, list(range(len(voltages))))
        fig_current = create_current_plot(currents, list(range(len(currents))))
        fig_efficiency = create_efficiency_plot(speeds, currents)
        fig_gps = create_gps_path_plot(gps_points)

        # Analyze performance
        metrics = analyze_performance(speeds, voltages, currents, list(range(len(speeds))))

        summary = f"""
        ### Telemetry Analysis Summary
        - **Total Samples:** {len(speeds)}
        - **Max Speed:** {metrics['max_speed']:.2f} m/s
        - **Avg Speed:** {metrics['avg_speed']:.2f} m/s
        - **Min Speed:** {metrics['min_speed']:.2f} m/s
        - **Distance:** {metrics['distance']:.1f} m
        - **Duration:** {metrics['duration']:.0f} seconds
        - **Max Voltage:** {metrics['max_voltage']:.2f} V
        - **Min Voltage:** {metrics['min_voltage']:.2f} V
        - **Max Current:** {metrics['max_current']:.2f} A
        - **Avg Current:** {metrics['avg_current']:.2f} A
        """

        return fig_speed, fig_voltage, fig_current, fig_efficiency, fig_gps, summary
    except Exception as e:
        return None, None, None, None, None, f"Error analyzing data: {str(e)}"


# ============================================================================
# TAB: LIVE TELEMETRY
# ============================================================================

def connect_hardware(vesc_port, gps_port, use_simulation):
    """Connect to VESC and GPS hardware."""
    global live_collector, live_gps, live_data_buffer

    try:
        live_data_buffer = []

        # Create VESC collector
        live_collector = VESCDataCollector(
            port=vesc_port,
            simulate=use_simulation
        )

        if not live_collector.connect():
            return "Failed to connect to VESC", "Disconnected"

        # Create GPS reader if port specified
        if gps_port and gps_port != "None" and not use_simulation:
            live_gps = GPSReader(port=gps_port, simulate=False)
            live_gps.connect()
            live_gps.start()
            gps_status = f"GPS connected on {gps_port}"
        elif use_simulation:
            live_gps = GPSReader(simulate=True)
            live_gps.connect()
            live_gps.start()
            gps_status = "GPS simulation active"
        else:
            live_gps = None
            gps_status = "No GPS"

        mode = "SIMULATION" if use_simulation else "HARDWARE"
        return f"Connected ({mode}) - VESC: {vesc_port}, {gps_status}", "Connected"

    except Exception as e:
        return f"Connection error: {str(e)}", "Error"


def disconnect_hardware():
    """Disconnect from hardware."""
    global live_collector, live_gps

    try:
        if live_collector:
            live_collector.disconnect()
            live_collector = None

        if live_gps:
            live_gps.stop()
            live_gps.disconnect()
            live_gps = None

        return "Disconnected", "Disconnected"
    except Exception as e:
        return f"Disconnect error: {str(e)}", "Error"


def start_recording(session_name, location, conditions):
    """Start recording telemetry data."""
    global live_collector, live_data_buffer

    if not live_collector or not live_collector.is_connected():
        return "Not connected to hardware!", "Stopped"

    try:
        live_data_buffer = []
        session_id = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        live_collector.start_collection(
            duration=None,  # Continuous
            interval=0.5,
            session_id=session_id,
            location=location,
            conditions=conditions,
            threaded=True
        )

        return f"Recording started: {session_id}", "Recording"
    except Exception as e:
        return f"Start error: {str(e)}", "Error"


def stop_recording():
    """Stop recording and return summary."""
    global live_collector

    if not live_collector:
        return "No active recording", "Stopped", ""

    try:
        live_collector.stop_collection()
        data = live_collector.get_data()

        if data:
            summary = f"""### Recording Summary
- **Points Collected:** {len(data)}
- **Duration:** {(data[-1].timestamp - data[0].timestamp).total_seconds():.1f}s
- **Avg Voltage:** {sum(p.battery_voltage for p in data) / len(data):.2f}V
- **Max Current:** {max(p.motor_current for p in data):.2f}A
- **Avg Speed:** {sum(p.speed_gps for p in data) / len(data):.2f} m/s
"""
        else:
            summary = "No data collected"

        return "Recording stopped", "Stopped", summary
    except Exception as e:
        return f"Stop error: {str(e)}", "Error", ""


def save_session(filename):
    """Save current session to file."""
    global live_collector

    if not live_collector or not live_collector.current_session:
        return "No session to save"

    try:
        filepath = filename or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        live_collector.save_session(filepath)
        return f"Session saved to {filepath}"
    except Exception as e:
        return f"Save error: {str(e)}"


def get_live_telemetry():
    """Get current telemetry reading for live display."""
    global live_collector, live_gps, live_data_buffer

    if not live_collector or not live_collector.is_connected():
        return "---", "---", "---", "---", "---", "---", None, None

    try:
        # Get GPS data if available
        gps_position = ""
        speed_gps = 0.0
        if live_gps:
            fix = live_gps.get_current_fix()
            if fix:
                gps_position = fix.mgrs_position
                speed_gps = fix.speed

        # Read telemetry
        point = live_collector.read_telemetry_point(gps_position, speed_gps)

        if not point:
            return "---", "---", "---", "---", "---", "---", None, None

        # Add to buffer for charts
        live_data_buffer.append(point)
        if len(live_data_buffer) > MAX_BUFFER_SIZE:
            live_data_buffer = live_data_buffer[-MAX_BUFFER_SIZE:]

        # Create live charts
        if len(live_data_buffer) > 1:
            times = [(p.timestamp - live_data_buffer[0].timestamp).total_seconds()
                    for p in live_data_buffer]
            voltages = [p.battery_voltage for p in live_data_buffer]
            currents = [p.motor_current for p in live_data_buffer]
            speeds = [p.speed_gps for p in live_data_buffer]
            powers = [p.power_in or (p.battery_voltage * p.motor_current)
                     for p in live_data_buffer]

            # Voltage/Current chart
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(
                go.Scatter(x=times, y=voltages, name="Voltage", line=dict(color="blue")),
                secondary_y=False
            )
            fig1.add_trace(
                go.Scatter(x=times, y=currents, name="Current", line=dict(color="red")),
                secondary_y=True
            )
            fig1.update_layout(
                title="Voltage & Current",
                xaxis_title="Time (s)",
                height=300,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            fig1.update_yaxes(title_text="Voltage (V)", secondary_y=False)
            fig1.update_yaxes(title_text="Current (A)", secondary_y=True)

            # Speed/Power chart
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(
                go.Scatter(x=times, y=speeds, name="Speed", line=dict(color="green")),
                secondary_y=False
            )
            fig2.add_trace(
                go.Scatter(x=times, y=powers, name="Power", line=dict(color="orange")),
                secondary_y=True
            )
            fig2.update_layout(
                title="Speed & Power",
                xaxis_title="Time (s)",
                height=300,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            fig2.update_yaxes(title_text="Speed (m/s)", secondary_y=False)
            fig2.update_yaxes(title_text="Power (W)", secondary_y=True)
        else:
            fig1 = None
            fig2 = None

        # Format display values
        voltage_str = f"{point.battery_voltage:.2f} V"
        current_str = f"{point.motor_current:.2f} A"
        speed_str = f"{point.speed_gps:.2f} m/s"
        power_str = f"{point.power_in:.1f} W" if point.power_in else "---"
        temp_str = f"{point.temp_fet:.1f}C" if point.temp_fet else "---"
        gps_str = point.gps_position[:15] if point.gps_position else "---"

        return voltage_str, current_str, speed_str, power_str, temp_str, gps_str, fig1, fig2

    except Exception as e:
        return f"Error: {e}", "---", "---", "---", "---", "---", None, None


def run_system_identification():
    """Run system identification on collected data."""
    global live_collector

    if not live_collector or not live_collector.telemetry_data:
        return "No telemetry data available. Record some data first!", None

    if not SYSTEM_ID_AVAILABLE:
        return "System identification module not available", None

    try:
        identifier = SystemIdentifier()
        identifier.load_telemetry(live_collector.telemetry_data)
        params = identifier.identify_all()

        # Format results
        html = f"""
        <div style="font-family: monospace; padding: 20px;">
            <h2>Identified Boat Parameters</h2>
            <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                <tr style="background-color: #4CAF50; color: white;">
                    <th style="padding: 10px; border: 1px solid #ddd;">Parameter</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Value</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Quality</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Effective Mass</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.mass:.1f} kg</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.estimation_quality.get('mass', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Drag Coefficient</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.drag_coefficient:.3f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.estimation_quality.get('drag_coefficient', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Motor Efficiency</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.motor_efficiency:.1%}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.estimation_quality.get('efficiency', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Propeller Efficiency</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.propeller_efficiency:.1%}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">-</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Battery Internal Resistance</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.battery_internal_resistance:.3f} Ohm</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{params.estimation_quality.get('battery_resistance', 0):.2f}</td>
                </tr>
            </table>
            <p><em>Quality scores range from 0 (poor) to 1 (excellent)</em></p>
            <p><strong>Estimated:</strong> {params.estimation_date}</p>
        </div>
        """

        # Save params
        params.save("identified_params.json")

        return html, "Parameters saved to identified_params.json"

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", None


# ============================================================================
# BUILD GRADIO INTERFACE
# ============================================================================

with gr.Blocks(
    title="Solar Regatta - VESC Dashboard",
    theme=gr.themes.Soft(),
    css="""
    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; color: white; border-radius: 10px; }
    .tab-title { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    """
) as demo:

    gr.Markdown("""
    # 🚤 Solar Regatta - VESC Dashboard

    Comprehensive suite for solar boat telemetry analysis:
    - Collect real-time VESC motor controller data
    - Train machine learning models on telemetry
    - Make real-time speed predictions
    - Analyze and visualize race data

    **Current Status:** Ready for Hugging Face Spaces deployment
    """)

    with gr.Tabs():

        # ====================================================================
        # TAB 0: LIVE TELEMETRY (Hardware Integration)
        # ====================================================================
        with gr.Tab("🔴 Live Telemetry"):
            gr.Markdown("## Real-Time Hardware Integration")
            gr.Markdown("""
            Connect to VESC motor controller and GPS for live telemetry.
            Use **Simulation Mode** for testing without hardware.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Connection Setup")

                    use_sim = gr.Checkbox(
                        label="Simulation Mode (no hardware required)",
                        value=True
                    )

                    vesc_port = gr.Dropdown(
                        choices=get_available_ports(),
                        value=get_available_ports()[0] if get_available_ports() else None,
                        label="VESC Serial Port",
                        allow_custom_value=True
                    )

                    gps_port = gr.Dropdown(
                        choices=["None"] + get_gps_ports(),
                        value="None",
                        label="GPS Serial Port (optional)",
                        allow_custom_value=True
                    )

                    with gr.Row():
                        connect_btn = gr.Button("Connect", variant="primary")
                        disconnect_btn = gr.Button("Disconnect", variant="secondary")

                    connection_status = gr.Textbox(
                        label="Connection Status",
                        value="Disconnected",
                        interactive=False
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Recording Controls")

                    session_name = gr.Textbox(
                        label="Session Name (optional)",
                        placeholder="race_001"
                    )

                    location = gr.Textbox(
                        label="Location",
                        placeholder="Lake Merced, SF"
                    )

                    conditions = gr.Textbox(
                        label="Conditions",
                        placeholder="Sunny, light wind, calm water"
                    )

                    with gr.Row():
                        start_rec_btn = gr.Button("Start Recording", variant="primary")
                        stop_rec_btn = gr.Button("Stop Recording", variant="stop")

                    recording_status = gr.Textbox(
                        label="Recording Status",
                        value="Stopped",
                        interactive=False
                    )

            gr.Markdown("### Live Readings")

            with gr.Row():
                voltage_display = gr.Textbox(label="Voltage", value="---", interactive=False)
                current_display = gr.Textbox(label="Current", value="---", interactive=False)
                speed_display = gr.Textbox(label="Speed", value="---", interactive=False)
                power_display = gr.Textbox(label="Power", value="---", interactive=False)
                temp_display = gr.Textbox(label="FET Temp", value="---", interactive=False)
                gps_display = gr.Textbox(label="GPS", value="---", interactive=False)

            refresh_btn = gr.Button("Refresh Reading", variant="secondary")

            with gr.Row():
                live_chart_1 = gr.Plot(label="Voltage & Current")
                live_chart_2 = gr.Plot(label="Speed & Power")

            gr.Markdown("### Session Management")

            with gr.Row():
                with gr.Column():
                    save_filename = gr.Textbox(
                        label="Save Filename",
                        placeholder="session_001.json"
                    )
                    save_btn = gr.Button("Save Session")
                    save_status = gr.Textbox(label="Save Status", interactive=False)

                with gr.Column():
                    recording_summary = gr.Markdown()

            gr.Markdown("### System Identification")
            gr.Markdown("""
            After collecting telemetry data (especially with varying motor loads and coasting periods),
            run System ID to estimate your boat's physical parameters.
            """)

            run_sysid_btn = gr.Button("Run System Identification", variant="primary")
            sysid_output = gr.HTML()
            sysid_status = gr.Textbox(label="Status", interactive=False)

            # Connect button handlers
            connect_btn.click(
                fn=connect_hardware,
                inputs=[vesc_port, gps_port, use_sim],
                outputs=[connection_status, connection_status]
            )

            disconnect_btn.click(
                fn=disconnect_hardware,
                outputs=[connection_status, connection_status]
            )

            start_rec_btn.click(
                fn=start_recording,
                inputs=[session_name, location, conditions],
                outputs=[recording_status, recording_status]
            )

            stop_rec_btn.click(
                fn=stop_recording,
                outputs=[recording_status, recording_status, recording_summary]
            )

            refresh_btn.click(
                fn=get_live_telemetry,
                outputs=[voltage_display, current_display, speed_display,
                        power_display, temp_display, gps_display,
                        live_chart_1, live_chart_2]
            )

            save_btn.click(
                fn=save_session,
                inputs=[save_filename],
                outputs=[save_status]
            )

            run_sysid_btn.click(
                fn=run_system_identification,
                outputs=[sysid_output, sysid_status]
            )

        # ====================================================================
        # TAB 1: VESC DATA COLLECTION
        # ====================================================================
        with gr.Tab("📊 Data Collection"):
            gr.Markdown("## VESC Data Collection & Export")
            gr.Markdown("""
            Simulate VESC motor controller telemetry data collection.
            In production, this connects to real VESC hardware via serial port.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Configuration")
                    duration_collection = gr.Slider(
                        minimum=30,
                        maximum=600,
                        value=120,
                        step=30,
                        label="Collection Duration (seconds)"
                    )
                    interval_collection = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="Sampling Interval (seconds)"
                    )
                    collect_btn = gr.Button("🔴 Start Collection", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Live Data Stream")
                    collection_summary = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Raw Data (JSON)")
                    collection_data = gr.Textbox(
                        label="Telemetry Data",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
                    export_csv_btn = gr.Button("📥 Export as CSV")
                    csv_output = gr.Textbox(
                        label="CSV Export",
                        lines=5,
                        interactive=False
                    )

            # Connect collection
            collect_btn.click(
                fn=collect_vesc_data_simulation,
                inputs=[duration_collection, interval_collection],
                outputs=[collection_data, collection_summary]
            )

            export_csv_btn.click(
                fn=export_collection_csv,
                inputs=[collection_data],
                outputs=[csv_output]
            )

        # ====================================================================
        # TAB 2: MODEL TRAINING
        # ====================================================================
        with gr.Tab("🤖 Model Training"):
            gr.Markdown("## Train ML Models on VESC Data")
            gr.Markdown("""
            Train various machine learning models to predict solar boat speed
            from battery voltage and motor current.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Training Configuration")

                    model_type_train = gr.Dropdown(
                        choices=["Linear Regression", "Random Forest", "XGBoost"],
                        value="Linear Regression",
                        label="Model Type"
                    )

                    duration_train = gr.Slider(
                        minimum=60,
                        maximum=600,
                        value=300,
                        step=60,
                        label="Simulation Duration (seconds)"
                    )

                    n_samples_train = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=60,
                        step=10,
                        label="Number of Samples"
                    )

                    use_feature_eng_train = gr.Checkbox(
                        label="Use Feature Engineering (20+ features)",
                        value=False
                    )

                    train_model_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### Training Summary")
                    summary_output = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Parameter Table")
                    param_table_output = gr.HTML()

                with gr.Column():
                    gr.Markdown("### Feature Importance")
                    feature_plot_output = gr.Image(label="Feature Importance Plot")

            # Connect training
            train_model_btn.click(
                fn=train_and_analyze,
                inputs=[model_type_train, duration_train, n_samples_train, use_feature_eng_train],
                outputs=[param_table_output, feature_plot_output, summary_output]
            )

        # ====================================================================
        # TAB 3: SPEED PREDICTION
        # ====================================================================
        with gr.Tab("⚡ Speed Predictions"):
            gr.Markdown("## Real-Time Speed Prediction")
            gr.Markdown("""
            Train a model and use it to predict boat speed from electrical measurements.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Train Model")

                    model_type_pred = gr.Dropdown(
                        choices=["Linear Regression", "Random Forest", "XGBoost"],
                        value="Linear Regression",
                        label="Model Type"
                    )

                    duration_pred = gr.Slider(
                        minimum=60,
                        maximum=600,
                        value=300,
                        step=60,
                        label="Training Duration (seconds)"
                    )

                    n_samples_pred = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=60,
                        step=10,
                        label="Training Samples"
                    )

                    use_feature_eng_pred = gr.Checkbox(
                        label="Use Feature Engineering",
                        value=False
                    )

                    train_pred_btn = gr.Button("🔧 Prepare Model", variant="primary")
                    train_status = gr.Textbox(label="Training Status", interactive=False)
                    train_info = gr.Textbox(label="Model Info", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Step 2: Make Predictions")

                    voltage_input = gr.Number(
                        value=13.0,
                        label="Battery Voltage (V)"
                    )

                    current_input = gr.Number(
                        value=5.0,
                        label="Motor Current (A)"
                    )

                    prev_speed_input = gr.Number(
                        value=2.0,
                        label="Previous Speed (m/s)"
                    )

                    predict_btn = gr.Button("🎯 Predict Speed", variant="primary")

                    prediction_output = gr.Textbox(label="Prediction Result", interactive=False)
                    prediction_input_info = gr.Textbox(label="Input Summary", interactive=False)

            # Connect prediction
            train_pred_btn.click(
                fn=train_inference_model,
                inputs=[model_type_pred, duration_pred, n_samples_pred, use_feature_eng_pred],
                outputs=[train_status, train_info]
            )

            predict_btn.click(
                fn=predict_speed,
                inputs=[voltage_input, current_input, gr.Textbox(value="2024-01-01", visible=False), prev_speed_input],
                outputs=[prediction_output, prediction_input_info]
            )

        # ====================================================================
        # TAB 4: DATA ANALYSIS
        # ====================================================================
        with gr.Tab("📈 Analysis Dashboard"):
            gr.Markdown("## Telemetry Data Analysis & Visualization")
            gr.Markdown("""
            Upload your race telemetry data and explore interactive visualizations
            of speed, voltage, current, and efficiency metrics.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Data")
                    data_file = gr.File(
                        label="Upload JSON Telemetry",
                        file_types=[".json"]
                    )
                    analyze_btn = gr.Button("📊 Analyze Data", variant="primary", size="lg")
                    analysis_summary = gr.Markdown()

                with gr.Column(scale=2):
                    gr.Markdown("### Quick Stats")

            gr.Markdown("### Visualizations")

            with gr.Row():
                with gr.Column():
                    speed_plot = gr.Plot(label="Speed vs Time")
                with gr.Column():
                    voltage_plot = gr.Plot(label="Voltage vs Time")

            with gr.Row():
                with gr.Column():
                    current_plot = gr.Plot(label="Current vs Time")
                with gr.Column():
                    efficiency_plot = gr.Plot(label="Speed vs Current")

            with gr.Row():
                gps_plot = gr.Plot(label="GPS Track")

            # Connect analysis
            analyze_btn.click(
                fn=analyze_uploaded_data,
                inputs=[data_file],
                outputs=[speed_plot, voltage_plot, current_plot, efficiency_plot, gps_plot, analysis_summary]
            )

        # ====================================================================
        # TAB 5: WORLD MODEL SIMULATOR
        # ====================================================================
        if WORLD_MODEL_AVAILABLE:
            with gr.Tab("🌍 World Model"):
                gr.Markdown("## Physics-Based World Model & Race Simulation")
                gr.Markdown("""
                Advanced predictive model that simulates boat physics, energy dynamics,
                and optimal racing strategies. Uses real physical models of drag, propulsion,
                battery dynamics, and solar power generation.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Race Configuration")

                        race_distance = gr.Slider(
                            minimum=100,
                            maximum=2000,
                            value=500,
                            step=50,
                            label="Race Distance (m)"
                        )

                        sun_condition = gr.Dropdown(
                            choices=["Full Sun", "Partly Cloudy", "Variable", "Cloudy"],
                            value="Full Sun",
                            label="Sun Conditions"
                        )

                        strategies_to_compare = gr.CheckboxGroup(
                            choices=["optimal", "aggressive", "conservative"],
                            value=["optimal", "aggressive"],
                            label="Strategies to Compare"
                        )

                        simulate_btn = gr.Button("🚀 Run Simulation", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### Simulation Settings")

                        boat_mass = gr.Slider(
                            minimum=30,
                            maximum=80,
                            value=50,
                            step=5,
                            label="Boat Mass (kg)"
                        )

                        battery_capacity = gr.Slider(
                            minimum=50,
                            maximum=200,
                            value=100,
                            step=10,
                            label="Battery Capacity (Wh)"
                        )

                        solar_area = gr.Slider(
                            minimum=0.3,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Solar Panel Area (m²)"
                        )

                        enable_uncertainty = gr.Checkbox(
                            label="Show Uncertainty Estimates (slower)",
                            value=False
                        )

                gr.Markdown("### Simulation Results")

                with gr.Row():
                    trajectory_plot = gr.Image(label="Race Trajectories")
                    state_evolution_plot = gr.Image(label="State Evolution")

                with gr.Row():
                    strategy_comparison_plot = gr.Image(label="Strategy Comparison")
                    uncertainty_plot = gr.Image(label="Prediction Uncertainty")

                with gr.Row():
                    metrics_output = gr.Markdown()

                # Connect world model simulation
                def run_world_model_simulation(distance, sun_cond, strategies,
                                             mass, capacity, area, uncertainty):
                    try:
                        # Create sun profile based on conditions
                        duration = 600
                        if sun_cond == "Full Sun":
                            sun_profile = [1000.0] * duration
                        elif sun_cond == "Cloudy":
                            sun_profile = [300.0] * duration
                        elif sun_cond == "Partly Cloudy":
                            sun_profile = [600.0] * duration
                        else:  # Variable
                            sun_profile = [1000.0 - 400 * np.sin(t * 0.05) for t in range(duration)]

                        # Create custom world model
                        params = PhysicsParameters(
                            mass=mass,
                            battery_capacity=capacity,
                            solar_panel_area=area
                        )
                        world_model = create_default_world_model()
                        world_model.params = params
                        world_model.dynamics.params = params

                        # Run simulations for each strategy
                        trajectories_dict = {}
                        metrics_dict = {}

                        for strategy in strategies:
                            traj, metrics = simulate_race(
                                world_model, distance, sun_profile, strategy
                            )
                            trajectories_dict[strategy] = traj
                            metrics_dict[strategy] = metrics

                        # Create visualizations
                        traj_plot = plot_trajectory_2d(
                            list(trajectories_dict.values()),
                            labels=list(trajectories_dict.keys()),
                            title=f"{distance}m Race - Trajectory Comparison"
                        )

                        # Pick first strategy for detailed evolution
                        first_strategy = strategies[0]
                        evolution_plot = plot_state_evolution(
                            trajectories_dict[first_strategy],
                            title=f"State Evolution - {first_strategy.title()} Strategy"
                        )

                        comparison_plot = plot_strategy_comparison(
                            trajectories_dict,
                            metrics_dict,
                            title="Strategy Performance Comparison"
                        )

                        # Uncertainty analysis if enabled
                        if uncertainty and len(strategies) > 0:
                            # Run uncertainty prediction for first strategy
                            initial_state = BoatState(
                                time=0.0,
                                position=np.array([0.0, 0.0]),
                                velocity=0.0,
                                heading=0.0,
                                battery_voltage=13.0,
                                battery_soc=1.0,
                                motor_current=0.0,
                                solar_power=0.0
                            )

                            # Create simple control sequence
                            control_seq = [(5.0, sun_profile[min(i, len(sun_profile)-1)])
                                         for i in range(min(300, len(sun_profile)))]

                            mean_traj, uncertainties = world_model.predict_with_uncertainty(
                                initial_state, control_seq, n_samples=50, dt=1.0
                            )

                            unc_plot = plot_uncertainty_bands(
                                mean_traj, uncertainties,
                                title="Prediction Uncertainty (Monte Carlo)"
                            )
                        else:
                            unc_plot = None

                        # Format metrics
                        metrics_md = f"""
                        ### Performance Summary

                        | Strategy | Distance (m) | Time (s) | Avg Speed (m/s) | Energy Used (Wh) | Efficiency (m/Wh) | Final SOC (%) |
                        |----------|-------------|----------|----------------|-----------------|------------------|---------------|
                        """

                        for strategy in strategies:
                            m = metrics_dict[strategy]
                            metrics_md += f"| {strategy.title()} | {m['total_distance']:.1f} | {m['total_time']:.1f} | {m['avg_velocity']:.2f} | {m['energy_used_wh']:.1f} | {m['efficiency_m_per_wh']:.2f} | {m['final_soc']*100:.1f} |\n"

                        metrics_md += f"""

                        **Race Configuration:**
                        - Distance: {distance}m
                        - Sun Conditions: {sun_cond}
                        - Boat Mass: {mass}kg
                        - Battery: {capacity}Wh
                        - Solar Panel: {area}m²
                        """

                        return traj_plot, evolution_plot, comparison_plot, unc_plot, metrics_md

                    except Exception as e:
                        import traceback
                        error_msg = f"Error running simulation: {str(e)}\n{traceback.format_exc()}"
                        return None, None, None, None, error_msg

                simulate_btn.click(
                    fn=run_world_model_simulation,
                    inputs=[race_distance, sun_condition, strategies_to_compare,
                           boat_mass, battery_capacity, solar_area, enable_uncertainty],
                    outputs=[trajectory_plot, state_evolution_plot,
                            strategy_comparison_plot, uncertainty_plot, metrics_output]
                )

        # ====================================================================
        # TAB 6: ABOUT
        # ====================================================================
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            # About Solar Regatta

            **Solar Regatta** is a comprehensive Python toolkit for analyzing telemetry data
            from solar-powered boats competing in autonomous races.

            ## Features

            - **Real-time VESC Integration** - Connect directly to VESC motor controllers
            - **GPS Tracking** - MGRS format GPS position logging and path analysis
            - **ML Models** - Train linear regression, random forest, and XGBoost models
            - **Feature Engineering** - 40+ derived features from raw telemetry
            - **Interactive Visualizations** - Plotly-based charts in notebooks and web UIs
            - **Performance Analytics** - Speed, efficiency, battery, and motor metrics

            ## Quick Links

            - **GitHub:** [charlieijk/SolarRegetta](https://github.com/charlieijk/SolarRegetta)
            - **Hugging Face Space:** [charlieijk/solar-regatta](https://huggingface.co/spaces/charlieijk/solar-regatta)
            - **Documentation:** [VESC Integration Guide](https://github.com/charlieijk/SolarRegetta/blob/main/VESC_INTEGRATION.md)

            ## Technologies

            - Python 3.8+
            - NumPy, SciPy, Scikit-learn
            - Plotly, Matplotlib
            - Gradio for web UI
            - PySerial for VESC communication

            ## Author

            Charlie Cullen (@charlieijk)

            Built with ❤️ for the Pixel Club SF solar boat racing initiative.
            """)

    gr.Markdown("""
    ---

    **Solar Regatta v0.1.0** | Deployed on Hugging Face Spaces

    [GitHub](https://github.com/charlieijk/SolarRegetta) |
    [Issues](https://github.com/charlieijk/SolarRegetta/issues) |
    [Discussion](https://huggingface.co/spaces/charlieijk/solar-regatta/discussions)
    """)


if __name__ == "__main__":
    demo.launch(share=False)
