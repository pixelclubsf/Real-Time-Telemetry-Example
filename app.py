"""
Solar Boat Race Analysis - Flask Web Interface
Interactive dashboard for visualizing VESC telemetry data
"""

from flask import Flask, render_template, request, jsonify
import plotly.graph_objects as go
import plotly.utils
import json
from datetime import datetime, timedelta
import random

from solar import (
    calculate_speeds,
    generate_sample_vesc_data,
    analyze_performance
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload

# Global variables to store current data
current_data = {
    'gps_points': [],
    'timestamps': [],
    'speeds': [],
    'battery_voltage': [],
    'motor_current': [],
    'metrics': {}
}


def create_speed_plot(speeds, timestamps):
    """Create interactive Plotly speed vs time graph"""
    time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_values[:-1],
        y=speeds,
        mode='lines+markers',
        name='Speed',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Speed:</b> %{y:.2f} m/s<extra></extra>'
    ))

    fig.update_layout(
        title='Speed vs Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Speed (m/s)',
        hovermode='x unified',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Arial, sans-serif', size=12)
    )

    return fig


def create_voltage_plot(battery_voltage, timestamps):
    """Create interactive battery voltage graph"""
    time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_values,
        y=battery_voltage,
        mode='lines+markers',
        name='Voltage',
        line=dict(color='#A23B72', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(162, 59, 114, 0.2)',
        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Voltage:</b> %{y:.2f}V<extra></extra>'
    ))

    # Add warning line
    fig.add_hline(
        y=11.0,
        line_dash='dash',
        line_color='red',
        annotation_text='Low Voltage Cutoff',
        annotation_position='right'
    )

    fig.update_layout(
        title='Battery Voltage Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Voltage (V)',
        hovermode='x unified',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Arial, sans-serif', size=12)
    )

    return fig


def create_current_plot(motor_current, timestamps):
    """Create interactive motor current graph"""
    time_values = [(t - timestamps[0]).total_seconds() for t in timestamps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_values,
        y=motor_current,
        mode='lines+markers',
        name='Current',
        line=dict(color='#F18F01', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(241, 143, 1, 0.2)',
        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Current:</b> %{y:.2f}A<extra></extra>'
    ))

    fig.update_layout(
        title='Motor Current Draw',
        xaxis_title='Time (seconds)',
        yaxis_title='Current (A)',
        hovermode='x unified',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Arial, sans-serif', size=12)
    )

    return fig


def create_efficiency_plot(speeds, motor_current):
    """Create interactive speed vs current efficiency plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=speeds,
        y=motor_current[:-1],
        mode='markers',
        name='Efficiency',
        marker=dict(
            size=8,
            color=speeds,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Speed (m/s)')
        ),
        hovertemplate='<b>Speed:</b> %{x:.2f} m/s<br><b>Current:</b> %{y:.2f}A<extra></extra>'
    ))

    fig.update_layout(
        title='Speed vs Motor Current (Efficiency)',
        xaxis_title='Speed (m/s)',
        yaxis_title='Current (A)',
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Arial, sans-serif', size=12)
    )

    return fig


def create_gps_path_plot(gps_points):
    """Create interactive GPS path visualization"""
    # For now, create a simple sequential path visualization
    distances = list(range(len(gps_points)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distances,
        y=list(range(len(gps_points))),
        mode='lines+markers',
        name='GPS Path',
        line=dict(color='#06A77D', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Point:</b> %{text}<br><b>Sequence:</b> %{x}<extra></extra>',
        text=gps_points
    ))

    fig.update_layout(
        title='GPS Track Points',
        xaxis_title='Track Sequence',
        yaxis_title='Position Index',
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Arial, sans-serif', size=12)
    )

    return fig


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/load-sample-data', methods=['POST'])
def load_sample_data():
    """Load sample VESC data"""
    try:
        duration = request.json.get('duration', 300)
        interval = request.json.get('interval', 5)

        # Generate sample data
        gps_points, timestamps, speeds_raw, battery_voltage, motor_current = \
            generate_sample_vesc_data(duration_seconds=duration, interval=interval)

        # Calculate speeds from GPS
        speeds = calculate_speeds(gps_points, timestamps)

        # Analyze performance
        metrics = analyze_performance(speeds, battery_voltage, motor_current, timestamps)

        # Store in global current_data
        current_data['gps_points'] = gps_points
        current_data['timestamps'] = timestamps
        current_data['speeds'] = speeds
        current_data['battery_voltage'] = battery_voltage
        current_data['motor_current'] = motor_current
        current_data['metrics'] = metrics

        return jsonify({
            'status': 'success',
            'message': f'Loaded sample data with {len(gps_points)} points'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/charts')
def get_charts():
    """Get all chart data as JSON"""
    try:
        if not current_data['speeds']:
            return jsonify({'status': 'error', 'message': 'No data loaded'}), 400

        # Create all plots
        speed_plot = create_speed_plot(current_data['speeds'], current_data['timestamps'])
        voltage_plot = create_voltage_plot(current_data['battery_voltage'], current_data['timestamps'])
        current_plot = create_current_plot(current_data['motor_current'], current_data['timestamps'])
        efficiency_plot = create_efficiency_plot(current_data['speeds'], current_data['motor_current'])
        gps_plot = create_gps_path_plot(current_data['gps_points'])

        return jsonify({
            'status': 'success',
            'speed_chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(speed_plot)),
            'voltage_chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(voltage_plot)),
            'current_chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(current_plot)),
            'efficiency_chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(efficiency_plot)),
            'gps_chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(gps_plot)),
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/metrics')
def get_metrics():
    """Get performance metrics"""
    try:
        if not current_data['metrics']:
            return jsonify({'status': 'error', 'message': 'No data loaded'}), 400

        metrics = current_data['metrics']

        return jsonify({
            'status': 'success',
            'max_speed': f"{metrics['max_speed']:.2f} m/s",
            'min_speed': f"{metrics['min_speed']:.2f} m/s",
            'avg_speed': f"{metrics['avg_speed']:.2f} m/s",
            'max_voltage': f"{metrics['max_voltage']:.2f}V",
            'min_voltage': f"{metrics['min_voltage']:.2f}V",
            'max_current': f"{metrics['max_current']:.2f}A",
            'avg_current': f"{metrics['avg_current']:.2f}A",
            'distance': f"{metrics['distance']:.1f}m",
            'duration': f"{metrics['duration']:.0f}s ({metrics['duration']/60:.1f} min)",
            'start_position': current_data['gps_points'][0] if current_data['gps_points'] else 'N/A',
            'end_position': current_data['gps_points'][-1] if current_data['gps_points'] else 'N/A',
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export')
def export_data():
    """Export data as JSON"""
    try:
        if not current_data['speeds']:
            return jsonify({'status': 'error', 'message': 'No data to export'}), 400

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': current_data['metrics'],
            'data_points': len(current_data['speeds']),
            'speeds': current_data['speeds'],
            'battery_voltage': current_data['battery_voltage'],
            'motor_current': current_data['motor_current'],
            'gps_points': current_data['gps_points'],
        }

        return jsonify(export_data)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
