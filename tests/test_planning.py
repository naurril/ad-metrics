"""Tests for end-to-end planning and driving metrics."""

import numpy as np
import pytest
from admetrics.planning import (
    l2_distance,
    collision_rate,
    progress_score,
    route_completion,
    average_displacement_error_planning,
    lateral_deviation,
    heading_error,
    velocity_error,
    comfort_metrics,
    driving_score,
    planning_kl_divergence,
)


class TestL2Distance:
    """Test L2 distance calculation."""
    
    def test_identical_trajectories(self):
        """Test L2 distance with identical trajectories."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        dist = l2_distance(traj, traj)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_parallel_offset(self):
        """Test L2 distance with parallel trajectories."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        expert = np.array([[0, 1], [1, 1], [2, 1]])
        dist = l2_distance(pred, expert)
        assert dist == pytest.approx(1.0, abs=1e-6)
    
    def test_weighted_distance(self):
        """Test weighted L2 distance."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        expert = np.array([[0, 1], [1, 1], [2, 1]])
        weights = np.array([1.0, 1.0, 2.0])  # Weight last point more
        dist = l2_distance(pred, expert, weights=weights)
        # (1*1 + 1*1 + 2*1) / (1+1+2) = 4/4 = 1.0
        assert dist == pytest.approx(1.0, abs=1e-6)
    
    def test_3d_trajectory(self):
        """Test with 3D trajectories."""
        pred = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        expert = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]])
        dist = l2_distance(pred, expert)
        assert dist == pytest.approx(1.0, abs=1e-6)
    
    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        pred = np.array([[0, 0], [1, 0]])
        expert = np.array([[0, 0], [1, 0], [2, 0]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            l2_distance(pred, expert)


class TestCollisionRate:
    """Test collision rate calculation."""
    
    def test_no_collision(self):
        """Test with no collisions."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [np.array([[5, 5]])]  # Far away
        result = collision_rate(traj, obstacles)
        assert result['collision_rate'] == 0.0
        assert result['num_collisions'] == 0
        assert result['first_collision'] is None
    
    def test_single_collision(self):
        """Test with single collision."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [np.array([[1, 0]])]  # Collides at t=1
        result = collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0))
        assert result['collision_rate'] > 0.0
        assert result['num_collisions'] >= 1
    
    def test_dynamic_obstacle(self):
        """Test with dynamic obstacle."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        # Obstacle moving toward ego
        obstacles = [np.array([[0, 5], [1, 2], [2, 0]])]
        result = collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0))
        assert result['num_collisions'] >= 1
    
    def test_safety_margin(self):
        """Test safety margin."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        obstacles = [np.array([[1, 2.5]])]
        
        # Without margin, no collision
        result1 = collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0), safety_margin=0.0)
        
        # With large margin, collision
        result2 = collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0), safety_margin=2.0)
        
        assert result2['num_collisions'] >= result1['num_collisions']
    
    def test_multiple_obstacles(self):
        """Test with multiple obstacles."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        obstacles = [
            np.array([[1, 0]]),
            np.array([[3, 0]])
        ]
        result = collision_rate(traj, obstacles, vehicle_size=(2.0, 1.0))
        assert result['num_collisions'] >= 1


class TestProgressScore:
    """Test progress score calculation."""
    
    def test_complete_path(self):
        """Test with complete path traversal."""
        traj = np.array([[0, 0], [5, 0], [10, 0]])
        ref = np.array([[0, 0], [5, 0], [10, 0]])
        result = progress_score(traj, ref)
        assert result['progress_ratio'] == pytest.approx(1.0, abs=0.1)
        assert result['goal_reached'] is True
    
    def test_partial_progress(self):
        """Test with partial progress."""
        traj = np.array([[0, 0], [2, 0], [4, 0]])
        ref = np.array([[0, 0], [5, 0], [10, 0]])
        result = progress_score(traj, ref)
        assert 0.0 < result['progress_ratio'] < 1.0
    
    def test_no_progress(self):
        """Test with no progress."""
        traj = np.array([[0, 0], [0, 0.1], [0, 0]])
        ref = np.array([[0, 0], [5, 0], [10, 0]])
        result = progress_score(traj, ref)
        assert result['progress_ratio'] == pytest.approx(0.0, abs=0.1)
    
    def test_custom_goal(self):
        """Test with custom goal position."""
        traj = np.array([[0, 0], [5, 0], [10, 0]])
        ref = np.array([[0, 0], [10, 0], [20, 0]])
        goal = np.array([10, 0])
        result = progress_score(traj, ref, goal_position=goal)
        assert result['goal_reached'] is True


class TestRouteCompletion:
    """Test route completion metrics."""
    
    def test_all_waypoints_reached(self):
        """Test with all waypoints reached."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        waypoints = np.array([[1, 0], [2, 0]])
        result = route_completion(traj, waypoints, completion_radius=0.5)
        assert result['completion_rate'] == 1.0
        assert result['num_waypoints_reached'] == 2
        assert all(result['waypoint_status'])
    
    def test_partial_completion(self):
        """Test with partial waypoint completion."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        waypoints = np.array([[1, 0], [5, 0], [10, 0]])
        result = route_completion(traj, waypoints, completion_radius=0.5)
        assert result['completion_rate'] == pytest.approx(1/3, abs=0.01)
        assert result['num_waypoints_reached'] == 1
    
    def test_no_completion(self):
        """Test with no waypoints reached."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        waypoints = np.array([[10, 10], [20, 20]])
        result = route_completion(traj, waypoints, completion_radius=0.5)
        assert result['completion_rate'] == 0.0
        assert result['num_waypoints_reached'] == 0
    
    def test_completion_radius(self):
        """Test different completion radii."""
        traj = np.array([[0, 0], [1, 0], [2, 0]])
        waypoints = np.array([[1.3, 0]])
        
        result_small = route_completion(traj, waypoints, completion_radius=0.2)
        result_large = route_completion(traj, waypoints, completion_radius=0.5)
        
        assert result_large['num_waypoints_reached'] >= result_small['num_waypoints_reached']


class TestAverageDisplacementErrorPlanning:
    """Test planning ADE calculation."""
    
    def test_perfect_match(self):
        """Test with perfect trajectory match."""
        traj = np.random.rand(10, 2)
        result = average_displacement_error_planning(traj, traj)
        assert result['ADE'] == pytest.approx(0.0, abs=1e-6)
        assert result['FDE'] == pytest.approx(0.0, abs=1e-6)
    
    def test_constant_error(self):
        """Test with constant displacement."""
        pred = np.array([[0, 0], [1, 0], [2, 0]])
        expert = np.array([[0, 1], [1, 1], [2, 1]])
        result = average_displacement_error_planning(pred, expert)
        assert result['ADE'] == pytest.approx(1.0, abs=1e-6)
        assert result['FDE'] == pytest.approx(1.0, abs=1e-6)
    
    def test_temporal_weighting(self):
        """Test temporal weighting schemes."""
        pred = np.zeros((10, 2))
        expert = np.ones((10, 2))
        
        result_uniform = average_displacement_error_planning(pred, expert, timestep_weights='uniform')
        result_linear = average_displacement_error_planning(pred, expert, timestep_weights='linear')
        result_exp = average_displacement_error_planning(pred, expert, timestep_weights='exponential')
        
        # All should have similar ADE for constant error
        assert result_uniform['ADE'] > 0
        assert result_linear['weighted_ADE'] > 0
        assert result_exp['weighted_ADE'] > 0


class TestLateralDeviation:
    """Test lateral deviation metrics."""
    
    def test_perfect_following(self):
        """Test with perfect path following."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        result = lateral_deviation(traj, ref)
        assert result['mean_lateral_error'] == pytest.approx(0.0, abs=1e-6)
        assert result['lane_keeping_rate'] == 1.0
    
    def test_constant_offset(self):
        """Test with constant lateral offset."""
        traj = np.array([[0, 0.5], [1, 0.5], [2, 0.5]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        result = lateral_deviation(traj, ref)
        assert result['mean_lateral_error'] == pytest.approx(0.5, abs=0.1)
    
    def test_lane_departure(self):
        """Test lane departure detection."""
        # Half inside lane, half outside
        traj = np.array([[0, 0.5], [1, 0.5], [2, 2.5], [3, 2.5]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        result = lateral_deviation(traj, ref)
        assert 0.0 < result['lane_keeping_rate'] < 1.0


class TestHeadingError:
    """Test heading error calculation."""
    
    def test_perfect_heading(self):
        """Test with perfect heading match."""
        headings = np.array([0.0, 0.1, 0.2, 0.3])
        result = heading_error(headings, headings)
        assert result['mean_heading_error'] == pytest.approx(0.0, abs=1e-6)
    
    def test_constant_offset(self):
        """Test with constant heading offset."""
        pred = np.array([0.0, 0.1, 0.2])
        expert = np.array([0.1, 0.2, 0.3])
        result = heading_error(pred, expert)
        assert result['mean_heading_error'] == pytest.approx(0.1, abs=1e-6)
        assert result['mean_heading_error_deg'] == pytest.approx(np.degrees(0.1), abs=0.1)
    
    def test_wraparound(self):
        """Test heading wraparound handling."""
        pred = np.array([0.0, np.pi - 0.1])
        expert = np.array([0.0, -np.pi + 0.1])
        result = heading_error(pred, expert)
        # Should handle wraparound correctly
        assert result['mean_heading_error'] < np.pi


class TestVelocityError:
    """Test velocity error calculation."""
    
    def test_perfect_velocity(self):
        """Test with perfect velocity match."""
        vel = np.array([10.0, 12.0, 15.0])
        result = velocity_error(vel, vel)
        assert result['mean_velocity_error'] == pytest.approx(0.0, abs=1e-6)
        assert result['rmse_velocity'] == pytest.approx(0.0, abs=1e-6)
    
    def test_constant_error(self):
        """Test with constant velocity error."""
        pred = np.array([10.0, 12.0, 15.0])
        expert = np.array([11.0, 13.0, 16.0])
        result = velocity_error(pred, expert)
        assert result['mean_velocity_error'] == pytest.approx(1.0, abs=1e-6)
        assert result['max_velocity_error'] == pytest.approx(1.0, abs=1e-6)


class TestComfortMetrics:
    """Test comfort metrics calculation."""
    
    def test_smooth_trajectory(self):
        """Test with smooth constant velocity."""
        # Constant velocity trajectory
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        result = comfort_metrics(traj, t)
        assert result['mean_acceleration'] == pytest.approx(0.0, abs=0.1)
        assert result['comfort_rate'] >= 0.9
    
    def test_harsh_braking(self):
        """Test with harsh acceleration change."""
        # Start slow, then fast acceleration
        traj = np.array([[0, 0], [0.5, 0], [2, 0], [5, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        result = comfort_metrics(traj, t, max_acceleration=1.0)
        assert result['max_acceleration'] > 0.5
    
    def test_short_trajectory(self):
        """Test with very short trajectory."""
        traj = np.array([[0, 0], [1, 0]])
        t = np.array([0.0, 1.0])
        result = comfort_metrics(traj, t)
        assert result['mean_jerk'] == 0.0


class TestDrivingScore:
    """Test comprehensive driving score."""
    
    def test_perfect_driving(self):
        """Test with perfect driving."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        obstacles = []
        
        result = driving_score(traj, traj, obstacles, ref, t)
        assert result['driving_score'] >= 90.0
        assert result['safety_score'] == 100.0
    
    def test_collision_penalty(self):
        """Test that collisions reduce score."""
        traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        t = np.array([0.0, 1.0, 2.0, 3.0])
        obstacles = [np.array([[1, 0]])]
        
        result = driving_score(traj, traj, obstacles, ref, t)
        assert result['safety_score'] < 100.0
        assert result['driving_score'] < 100.0
    
    def test_custom_weights(self):
        """Test custom component weights."""
        traj = np.random.rand(10, 2)
        ref = np.random.rand(15, 2)
        t = np.linspace(0, 5, 10)
        obstacles = []
        
        weights = {'planning': 0.5, 'safety': 0.3, 'progress': 0.1, 'comfort': 0.1}
        result = driving_score(traj, traj, obstacles, ref, t, weights=weights)
        assert 0.0 <= result['driving_score'] <= 100.0


class TestPlanningKLDivergence:
    """Test KL divergence calculation."""
    
    def test_identical_distributions(self):
        """Test with identical distributions."""
        dist = np.array([0.2, 0.5, 0.3])
        kl = planning_kl_divergence(dist, dist)
        assert kl == pytest.approx(0.0, abs=1e-6)
    
    def test_different_distributions(self):
        """Test with different distributions."""
        pred = np.array([0.3, 0.4, 0.3])
        expert = np.array([0.1, 0.6, 0.3])
        kl = planning_kl_divergence(pred, expert)
        assert kl > 0.0
    
    def test_normalization(self):
        """Test that unnormalized inputs are handled."""
        pred = np.array([1, 2, 1])  # Not normalized
        expert = np.array([1, 3, 2])  # Not normalized
        kl = planning_kl_divergence(pred, expert)
        assert kl >= 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_trajectory(self):
        """Test with empty/minimal trajectories."""
        traj = np.array([[0, 0]])
        t = np.array([0.0])
        result = comfort_metrics(traj, t)
        assert result['comfort_rate'] == 1.0
    
    def test_single_point_trajectory(self):
        """Test with single point."""
        traj = np.array([[0, 0]])
        ref = np.array([[0, 0], [1, 0]])
        result = progress_score(traj, ref)
        assert 0.0 <= result['progress_ratio'] <= 1.0
    
    def test_backwards_trajectory(self):
        """Test with backwards motion."""
        pred = np.array([[3, 0], [2, 0], [1, 0], [0, 0]])
        expert = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        dist = l2_distance(pred, expert)
        assert dist > 0.0
    
    def test_stationary_vehicle(self):
        """Test with stationary vehicle."""
        traj = np.array([[0, 0], [0, 0], [0, 0]])
        ref = np.array([[0, 0], [1, 0], [2, 0]])
        result = progress_score(traj, ref)
        assert result['progress_ratio'] == pytest.approx(0.0, abs=0.1)
