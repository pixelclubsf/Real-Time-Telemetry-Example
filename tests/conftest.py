"""
Pytest configuration and shared fixtures for Solar Regatta tests.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_boat_state():
    """Create a sample boat state for testing."""
    from solar_regatta.ml.world_model import BoatState

    return BoatState(
        time=0.0,
        position=np.array([0.0, 0.0]),
        velocity=2.0,
        heading=0.0,
        battery_voltage=13.0,
        battery_soc=0.8,
        motor_current=5.0,
        solar_power=100.0
    )


@pytest.fixture
def physics_params():
    """Create default physics parameters."""
    from solar_regatta.ml.world_model import PhysicsParameters

    return PhysicsParameters(
        mass=50.0,
        hull_drag_coeff=0.5,
        frontal_area=0.3,
        motor_efficiency=0.85,
        prop_efficiency=0.65,
        battery_capacity=100.0,
        solar_panel_area=0.5,
        solar_efficiency=0.18
    )


@pytest.fixture
def world_model(physics_params):
    """Create world model with default parameters."""
    from solar_regatta.ml.world_model import WorldModel

    return WorldModel(physics_params)


@pytest.fixture
def sample_trajectory(sample_boat_state, world_model):
    """Generate a sample trajectory for testing."""
    control_sequence = [(5.0, 800.0) for _ in range(100)]
    return world_model.predict_trajectory(sample_boat_state, control_sequence)


@pytest.fixture
def environment_simulator():
    """Create environment simulator for testing."""
    from solar_regatta.environment import create_default_environment

    return create_default_environment()


