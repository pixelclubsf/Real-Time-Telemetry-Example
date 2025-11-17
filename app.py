#!/usr/bin/env python3
"""
Hugging Face Space - Solar Regatta Model Parameter Inspector
Interactive UI for exploring ML models and their parameters.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

from solar_regatta import generate_sample_vesc_data, calculate_speeds
from solar_regatta.ml import (
    train_speed_model,
    prepare_training_data,
    count_model_parameters,
    FeatureEngineer,
)
from solar_regatta.ml.models import PerformanceModel

# Try to import tree models
try:
    from solar_regatta.ml import RandomForestSpeedModel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from solar_regatta.ml import XGBoostSpeedModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def format_parameter_table(model, X=None):
    """Format parameter information as HTML table."""
    from solar_regatta.ml.models import PerformanceModel

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

    # Tree model info
    elif "n_trees" in info["details"]:
        details = info["details"]
        html += "<h3>Model Configuration</h3><ul>"
        if "n_trees" in details:
            html += f"<li><strong>Number of Trees:</strong> {details['n_trees']:,}</li>"
        if "total_nodes" in details:
            html += f"<li><strong>Total Nodes:</strong> {details['total_nodes']:,}</li>"
            html += f"<li><strong>Avg Nodes per Tree:</strong> {details['avg_nodes_per_tree']:.1f}</li>"
        html += "</ul>"

    html += "</div>"
    return html


def create_feature_importance_plot(model):
    """Create feature importance bar chart."""
    from solar_regatta.ml.models import PerformanceModel

    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(model, PerformanceModel):
        importance = np.abs(model.coefficients)
        features = model.feature_names

        # Sort by importance
        sorted_idx = np.argsort(importance)
        pos = np.arange(len(sorted_idx))

        ax.barh(pos, importance[sorted_idx], color='#4CAF50')
        ax.set_yticks(pos)
        ax.set_yticklabels([features[i] for i in sorted_idx])
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
            ax.set_yticklabels([features[i] for i in sorted_idx])
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


def train_and_analyze(model_type, duration, n_samples, use_feature_engineering):
    """Train model and return parameter analysis."""

    # Generate data
    interval = duration // n_samples
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

    # Generate outputs
    param_table = format_parameter_table(model, X)
    feature_plot = create_feature_importance_plot(model)

    # Summary stats
    info = count_model_parameters(model)
    summary = f"""
    ### Model Summary
    - **Type:** {info['model_type']}
    - **Total Parameters:** {info['total_parameters']:,}
    - **Training Samples:** {X.shape[0]:,}
    - **Features:** {X.shape[1]:,}
    """

    return param_table, feature_plot, summary


# Create Gradio interface
with gr.Blocks(title="Solar Regatta - Model Parameter Inspector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚤 Solar Regatta - Model Parameter Inspector

    Explore ML model parameters for solar boat telemetry analysis. Train different models and inspect their parameters,
    coefficients, and feature importance in real-time.

    **Features:**
    - Linear Regression with coefficient analysis
    - Random Forest with tree statistics
    - XGBoost gradient boosting
    - Feature engineering pipeline
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Configuration")

            model_type = gr.Dropdown(
                choices=["Linear Regression", "Random Forest", "XGBoost"],
                value="Linear Regression",
                label="Model Type"
            )

            duration = gr.Slider(
                minimum=60,
                maximum=600,
                value=300,
                step=60,
                label="Simulation Duration (seconds)"
            )

            n_samples = gr.Slider(
                minimum=20,
                maximum=100,
                value=60,
                step=10,
                label="Number of Samples"
            )

            use_feature_engineering = gr.Checkbox(
                label="Use Feature Engineering (creates 20+ features)",
                value=False
            )

            train_btn = gr.Button("Train Model & Show Parameters", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Model Summary")
            summary_output = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Parameter Table")
            param_table_output = gr.HTML()

        with gr.Column():
            gr.Markdown("### Feature Importance")
            feature_plot_output = gr.Image(label="Feature Importance Plot")

    # Connect button
    train_btn.click(
        fn=train_and_analyze,
        inputs=[model_type, duration, n_samples, use_feature_engineering],
        outputs=[param_table_output, feature_plot_output, summary_output]
    )

    gr.Markdown("""
    ---
    ### About

    This Space demonstrates ML model parameter inspection for the **Solar Regatta** project -
    a comprehensive toolkit for analyzing telemetry data from solar-powered boats.

    **GitHub:** [Solar Regatta](https://github.com/charlieijk/SolarRegetta)

    Built with ❤️ using Gradio and Hugging Face Spaces
    """)

if __name__ == "__main__":
    demo.launch()
