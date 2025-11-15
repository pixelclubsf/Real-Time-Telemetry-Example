"""Plotly visualization helpers used by notebooks and scripts."""

from datetime import datetime
from typing import Sequence

import plotly.graph_objects as go


def _time_values(timestamps: Sequence[datetime]) -> Sequence[float]:
    if not timestamps:
        return []
    start = timestamps[0]
    if hasattr(start, "timestamp"):
        return [(t - start).total_seconds() for t in timestamps]
    return timestamps


def create_speed_plot(speeds, timestamps):
    time_values = _time_values(timestamps)
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
    time_values = _time_values(timestamps)
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
    time_values = _time_values(timestamps)
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
