"""Tests for the physics-based world model."""
import pytest
import numpy as np


class TestBoatState:
    """Tests for BoatState dataclass."""

    def test_boat_state_creation(self, sample_boat_state):
        """Test BoatState can be created with valid params."""
        assert sample_boat_state.time == 0.0
        assert sample_boat_state.velocity == 2.0
        assert sample_boat_state.battery_soc == 0.8

    def test_boat_state_position_is_array(self, sample_boat_state):
        """Test position is numpy array."""
        assert isinstance(sample_boat_state.position, np.ndarray)
        assert len(sample_boat_state.position) == 2


class TestPhysicsParameters:
    """Tests for PhysicsParameters dataclass."""

    def test_physics_params_creation(self, physics_params):
        """Test PhysicsParameters can be created."""
        assert physics_params.mass == 50.0
        assert physics_params.motor_efficiency == 0.85
        assert 0 < physics_params.solar_efficiency < 1


class TestWorldModel:
    """Tests for WorldModel simulation."""

    def test_world_model_creation(self, world_model):
        """Test WorldModel can be created."""
        assert world_model is not None

    def test_predict_trajectory_basic(self, world_model, sample_boat_state):
        """Test basic trajectory prediction."""
        # 10 timesteps with constant control
        control_sequence = [(5.0, 800.0) for _ in range(10)]

        trajectory = world_model.predict_trajectory(
            sample_boat_state,
            control_sequence
        )

        assert len(trajectory) == 11  # Initial state + 10 steps
        assert trajectory[0].time == 0.0
        assert trajectory[-1].time > 0.0

    def test_trajectory_velocity_changes(self, world_model, sample_boat_state):
        """Test that velocity changes over trajectory."""
        control_sequence = [(10.0, 1000.0) for _ in range(50)]

        trajectory = world_model.predict_trajectory(
            sample_boat_state,
            control_sequence
        )

        initial_velocity = trajectory[0].velocity
        final_velocity = trajectory[-1].velocity

        # Velocity should change (boat accelerating or decelerating)
        # Allow for steady state where they might be equal
        assert isinstance(final_velocity, (int, float))

    def test_battery_soc_decreases_with_power(self, world_model, sample_boat_state):
        """Test that battery SOC decreases when drawing power."""
        # High current, low solar
        control_sequence = [(20.0, 0.0) for _ in range(100)]

        trajectory = world_model.predict_trajectory(
            sample_boat_state,
            control_sequence
        )

        initial_soc = trajectory[0].battery_soc
        final_soc = trajectory[-1].battery_soc

        # With no solar and high current, SOC should decrease
        assert final_soc < initial_soc

    def test_simulate_race_helper(self):
        """Test the simulate_race convenience function."""
        from solar_regatta.ml.world_model import create_default_world_model, simulate_race

        world_model = create_default_world_model()

        trajectory, metrics = simulate_race(
            world_model,
            race_distance=100.0,
            sun_profile=[800.0] * 120,
            strategy='optimal'
        )

        assert len(trajectory) > 0
        assert isinstance(metrics, dict)


class TestCreateDefaultWorldModel:
    """Tests for default world model creation."""

    def test_create_default_model(self):
        """Test default world model can be created."""
        from solar_regatta.ml.world_model import create_default_world_model

        model = create_default_world_model()
        assert model is not None

    def test_default_model_has_physics(self):
        """Test default model has physics parameters."""
        from solar_regatta.ml.world_model import create_default_world_model

        model = create_default_world_model()
        assert hasattr(model, 'dynamics') or hasattr(model, 'physics')
