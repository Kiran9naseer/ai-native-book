#!/usr/bin/env python3
"""
Navigation Module for Isaac AI Brain

This module provides navigation utilities integrating perception data with Nav2
for autonomous humanoid robot navigation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable
from enum import Enum
import threading
import time
import math


class NavigationState(Enum):
    """Navigation state enumeration."""
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class Pose2D:
    """2D pose representation."""
    x: float
    y: float
    theta: float  # Orientation in radians
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Pose2D':
        return cls(x=arr[0], y=arr[1], theta=arr[2])
    
    def distance_to(self, other: 'Pose2D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_to(self, other: 'Pose2D') -> float:
        return math.atan2(other.y - self.y, other.x - self.x)


@dataclass
class NavigationGoal:
    """Navigation goal container."""
    pose: Pose2D
    goal_id: str = ""
    tolerance_xy: float = 0.25
    tolerance_yaw: float = 0.25
    timeout: float = 300.0  # 5 minutes default
    created_at: float = field(default_factory=time.time)


@dataclass
class NavigationFeedback:
    """Navigation progress feedback."""
    current_pose: Pose2D
    goal_pose: Pose2D
    distance_remaining: float
    estimated_time_remaining: float
    recovery_count: int
    state: NavigationState


@dataclass
class BipedalConstraints:
    """Bipedal locomotion constraints for path validation."""
    max_step_length: float = 0.30
    max_step_width: float = 0.20
    max_velocity: float = 0.4
    max_angular_velocity: float = 0.5
    min_stability_margin: float = 0.05
    max_slope: float = 0.26  # ~15 degrees


class PathValidator:
    """
    Validates navigation paths against bipedal locomotion constraints.
    """
    
    def __init__(self, constraints: Optional[BipedalConstraints] = None):
        self.constraints = constraints or BipedalConstraints()
    
    def validate_path(self, path: List[Pose2D]) -> Tuple[bool, List[str]]:
        """
        Validate a path against bipedal constraints.
        
        Args:
            path: List of poses forming the path
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if len(path) < 2:
            return True, []
        
        for i in range(1, len(path)):
            prev = path[i - 1]
            curr = path[i]
            
            # Check step length
            step_length = prev.distance_to(curr)
            if step_length > self.constraints.max_step_length:
                issues.append(
                    f"Step {i}: length {step_length:.2f}m exceeds max {self.constraints.max_step_length:.2f}m"
                )
            
            # Check lateral movement
            heading = prev.angle_to(curr)
            lateral = abs(math.sin(heading - prev.theta) * step_length)
            if lateral > self.constraints.max_step_width:
                issues.append(
                    f"Step {i}: lateral {lateral:.2f}m exceeds max {self.constraints.max_step_width:.2f}m"
                )
            
            # Check angular change
            angular_change = abs(self._normalize_angle(curr.theta - prev.theta))
            if angular_change > math.pi / 4:  # 45 degrees per step
                issues.append(
                    f"Step {i}: angular change {math.degrees(angular_change):.1f}° is large"
                )
        
        return len(issues) == 0, issues
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class TrajectoryGenerator:
    """
    Generates smooth trajectories for humanoid navigation.
    """
    
    def __init__(self, constraints: Optional[BipedalConstraints] = None):
        self.constraints = constraints or BipedalConstraints()
    
    def generate_trajectory(
        self,
        start: Pose2D,
        goal: Pose2D,
        num_points: int = 50
    ) -> List[Pose2D]:
        """
        Generate a smooth trajectory from start to goal.
        
        Args:
            start: Starting pose
            goal: Goal pose
            num_points: Number of intermediate points
            
        Returns:
            List of poses forming the trajectory
        """
        trajectory = []
        
        # Calculate total distance and direction
        distance = start.distance_to(goal)
        direction = start.angle_to(goal)
        
        # Generate intermediate poses
        for i in range(num_points + 1):
            t = i / num_points
            
            # Smooth interpolation using cubic easing
            t_smooth = self._ease_in_out_cubic(t)
            
            # Position interpolation
            x = start.x + t_smooth * (goal.x - start.x)
            y = start.y + t_smooth * (goal.y - start.y)
            
            # Orientation interpolation with smooth transition
            if t < 0.3:
                # Initial phase: turn towards goal
                theta_target = direction
                theta = self._interpolate_angle(start.theta, theta_target, t / 0.3)
            elif t > 0.7:
                # Final phase: align with goal orientation
                theta = self._interpolate_angle(direction, goal.theta, (t - 0.7) / 0.3)
            else:
                # Middle phase: face direction of travel
                theta = direction
            
            trajectory.append(Pose2D(x=x, y=y, theta=theta))
        
        return trajectory
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing for smooth acceleration/deceleration."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def _interpolate_angle(self, a1: float, a2: float, t: float) -> float:
        """Interpolate between two angles."""
        diff = self._normalize_angle(a2 - a1)
        return a1 + t * diff
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class ObstacleDetector:
    """
    Integrates perception data for obstacle detection and avoidance.
    """
    
    def __init__(self):
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.lock = threading.Lock()
    
    def update_obstacles(self, obstacle_list: List[Tuple[float, float, float]]):
        """Update the obstacle list from perception data."""
        with self.lock:
            self.obstacles = obstacle_list.copy()
    
    def check_collision(
        self,
        pose: Pose2D,
        robot_radius: float = 0.3
    ) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        """
        Check if a pose would result in collision.
        
        Args:
            pose: The pose to check
            robot_radius: Robot's approximate radius
            
        Returns:
            Tuple of (is_collision, closest_obstacle)
        """
        with self.lock:
            closest_obstacle = None
            min_distance = float('inf')
            
            for obs in self.obstacles:
                ox, oy, r = obs
                distance = math.sqrt((pose.x - ox)**2 + (pose.y - oy)**2)
                clearance = distance - r - robot_radius
                
                if clearance < min_distance:
                    min_distance = clearance
                    closest_obstacle = obs
                
                if clearance < 0:
                    return True, obs
            
            return False, closest_obstacle
    
    def get_safe_direction(
        self,
        pose: Pose2D,
        goal: Pose2D,
        robot_radius: float = 0.3
    ) -> float:
        """
        Get a safe direction to move that avoids obstacles.
        
        Args:
            pose: Current pose
            goal: Goal pose
            robot_radius: Robot's radius
            
        Returns:
            Safe heading angle
        """
        goal_direction = pose.angle_to(goal)
        
        with self.lock:
            if not self.obstacles:
                return goal_direction
            
            # Calculate repulsion from obstacles
            repulsion_x = 0.0
            repulsion_y = 0.0
            
            for ox, oy, r in self.obstacles:
                dx = pose.x - ox
                dy = pose.y - oy
                distance = max(math.sqrt(dx*dx + dy*dy), 0.01)
                
                # Repulsion strength inversely proportional to distance
                influence_radius = r + robot_radius + 1.0
                if distance < influence_radius:
                    strength = (influence_radius - distance) / influence_radius
                    repulsion_x += strength * dx / distance
                    repulsion_y += strength * dy / distance
            
            # Combine goal attraction with obstacle repulsion
            goal_x = math.cos(goal_direction)
            goal_y = math.sin(goal_direction)
            
            combined_x = goal_x + 0.5 * repulsion_x
            combined_y = goal_y + 0.5 * repulsion_y
            
            return math.atan2(combined_y, combined_x)


class NavigationManager:
    """
    High-level navigation manager integrating perception and Nav2.
    """
    
    def __init__(
        self,
        constraints: Optional[BipedalConstraints] = None
    ):
        self.constraints = constraints or BipedalConstraints()
        self.path_validator = PathValidator(self.constraints)
        self.trajectory_generator = TrajectoryGenerator(self.constraints)
        self.obstacle_detector = ObstacleDetector()
        
        # State
        self.current_pose: Optional[Pose2D] = None
        self.current_goal: Optional[NavigationGoal] = None
        self.current_path: List[Pose2D] = []
        self.state = NavigationState.IDLE
        self.recovery_count = 0
        
        # Callbacks
        self.on_state_change: Optional[Callable[[NavigationState], None]] = None
        self.on_feedback: Optional[Callable[[NavigationFeedback], None]] = None
        
        self.lock = threading.Lock()
    
    def set_goal(self, goal: NavigationGoal) -> bool:
        """
        Set a new navigation goal.
        
        Args:
            goal: The navigation goal
            
        Returns:
            True if goal was accepted
        """
        with self.lock:
            if self.current_pose is None:
                return False
            
            self.current_goal = goal
            self.recovery_count = 0
            self._set_state(NavigationState.PLANNING)
            
            # Generate initial trajectory
            trajectory = self.trajectory_generator.generate_trajectory(
                self.current_pose,
                goal.pose
            )
            
            # Validate trajectory
            is_valid, issues = self.path_validator.validate_path(trajectory)
            
            if not is_valid:
                # Try to replan with smaller steps
                trajectory = self._replan_with_constraints(
                    self.current_pose,
                    goal.pose
                )
            
            self.current_path = trajectory
            self._set_state(NavigationState.NAVIGATING)
            
            return True
    
    def cancel_goal(self):
        """Cancel the current navigation goal."""
        with self.lock:
            self.current_goal = None
            self.current_path = []
            self._set_state(NavigationState.CANCELED)
    
    def update_pose(self, pose: Pose2D):
        """Update the current robot pose from localization."""
        with self.lock:
            self.current_pose = pose
            
            if self.state == NavigationState.NAVIGATING:
                self._check_progress()
    
    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Update obstacle information from perception."""
        self.obstacle_detector.update_obstacles(obstacles)
        
        # Check if current path is still valid
        with self.lock:
            if self.state == NavigationState.NAVIGATING and self.current_path:
                for pose in self.current_path:
                    is_collision, _ = self.obstacle_detector.check_collision(pose)
                    if is_collision:
                        self._trigger_replan()
                        break
    
    def get_velocity_command(self) -> Tuple[float, float]:
        """
        Get velocity command for the robot.
        
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        with self.lock:
            if self.state != NavigationState.NAVIGATING:
                return 0.0, 0.0
            
            if self.current_pose is None or not self.current_path:
                return 0.0, 0.0
            
            # Find closest path point ahead
            target_pose = self._get_lookahead_point()
            
            if target_pose is None:
                return 0.0, 0.0
            
            # Compute velocities towards target
            distance = self.current_pose.distance_to(target_pose)
            angle_to_target = self.current_pose.angle_to(target_pose)
            angle_error = self._normalize_angle(angle_to_target - self.current_pose.theta)
            
            # Simple proportional control
            linear_vel = min(
                0.5 * distance,
                self.constraints.max_velocity
            )
            angular_vel = np.clip(
                1.0 * angle_error,
                -self.constraints.max_angular_velocity,
                self.constraints.max_angular_velocity
            )
            
            # Reduce linear velocity when rotating
            if abs(angle_error) > 0.3:
                linear_vel *= 0.5
            
            return linear_vel, angular_vel
    
    def get_feedback(self) -> Optional[NavigationFeedback]:
        """Get current navigation feedback."""
        with self.lock:
            if self.current_pose is None or self.current_goal is None:
                return None
            
            distance = self.current_pose.distance_to(self.current_goal.pose)
            estimated_time = distance / max(self.constraints.max_velocity, 0.1)
            
            return NavigationFeedback(
                current_pose=self.current_pose,
                goal_pose=self.current_goal.pose,
                distance_remaining=distance,
                estimated_time_remaining=estimated_time,
                recovery_count=self.recovery_count,
                state=self.state
            )
    
    def _check_progress(self):
        """Check navigation progress and state."""
        if self.current_goal is None or self.current_pose is None:
            return
        
        distance = self.current_pose.distance_to(self.current_goal.pose)
        angle_error = abs(self._normalize_angle(
            self.current_pose.theta - self.current_goal.pose.theta
        ))
        
        # Check if goal reached
        if (distance < self.current_goal.tolerance_xy and
                angle_error < self.current_goal.tolerance_yaw):
            self._set_state(NavigationState.SUCCEEDED)
            return
        
        # Check timeout
        elapsed = time.time() - self.current_goal.created_at
        if elapsed > self.current_goal.timeout:
            self._set_state(NavigationState.FAILED)
    
    def _trigger_replan(self):
        """Trigger path replanning due to obstacles."""
        if self.current_goal is None or self.current_pose is None:
            return
        
        self._set_state(NavigationState.RECOVERING)
        self.recovery_count += 1
        
        # Generate new path avoiding obstacles
        trajectory = self._replan_with_constraints(
            self.current_pose,
            self.current_goal.pose
        )
        
        self.current_path = trajectory
        self._set_state(NavigationState.NAVIGATING)
    
    def _replan_with_constraints(
        self,
        start: Pose2D,
        goal: Pose2D
    ) -> List[Pose2D]:
        """Replan path with stricter constraints."""
        # Generate with more points for smoother path
        return self.trajectory_generator.generate_trajectory(
            start, goal, num_points=100
        )
    
    def _get_lookahead_point(self) -> Optional[Pose2D]:
        """Get the lookahead point on the path."""
        if not self.current_path or self.current_pose is None:
            return None
        
        lookahead_distance = 0.5
        
        # Find point at lookahead distance
        for pose in self.current_path:
            if self.current_pose.distance_to(pose) >= lookahead_distance:
                return pose
        
        # Return final pose if path is short
        return self.current_path[-1]
    
    def _set_state(self, new_state: NavigationState):
        """Set navigation state and trigger callback."""
        old_state = self.state
        self.state = new_state
        
        if self.on_state_change and old_state != new_state:
            self.on_state_change(new_state)
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# ROS 2 integration
def create_navigation_node():
    """Create ROS 2 navigation node."""
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Path, Odometry
        from std_msgs.msg import String
        
        class NavigationNode(Node):
            def __init__(self):
                super().__init__('navigation_manager_node')
                
                self.nav_manager = NavigationManager()
                
                # Subscribers
                self.odom_sub = self.create_subscription(
                    Odometry, '/odom', self.odom_callback, 10
                )
                self.goal_sub = self.create_subscription(
                    PoseStamped, '/goal_pose', self.goal_callback, 10
                )
                
                # Publishers
                self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
                self.path_pub = self.create_publisher(Path, '/planned_path', 10)
                
                # Timer for velocity commands
                self.timer = self.create_timer(0.05, self.publish_velocity)
                
                self.get_logger().info('Navigation Manager Node initialized')
            
            def odom_callback(self, msg):
                pose = Pose2D(
                    x=msg.pose.pose.position.x,
                    y=msg.pose.pose.position.y,
                    theta=self._get_yaw(msg.pose.pose.orientation)
                )
                self.nav_manager.update_pose(pose)
            
            def goal_callback(self, msg):
                goal = NavigationGoal(
                    pose=Pose2D(
                        x=msg.pose.position.x,
                        y=msg.pose.position.y,
                        theta=self._get_yaw(msg.pose.orientation)
                    )
                )
                self.nav_manager.set_goal(goal)
            
            def publish_velocity(self):
                linear, angular = self.nav_manager.get_velocity_command()
                
                twist = Twist()
                twist.linear.x = linear
                twist.angular.z = angular
                self.cmd_vel_pub.publish(twist)
            
            def _get_yaw(self, q):
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                return math.atan2(siny_cosp, cosy_cosp)
        
        return NavigationNode
    
    except ImportError:
        return None


def main():
    """Main entry point."""
    try:
        import rclpy
        rclpy.init()
        
        NavigationNode = create_navigation_node()
        if NavigationNode:
            node = NavigationNode()
            rclpy.spin(node)
            node.destroy_node()
        
        rclpy.shutdown()
    
    except ImportError:
        print("ROS 2 not available. Running standalone demo.")
        
        # Standalone demo
        nav_manager = NavigationManager()
        
        # Set initial pose
        nav_manager.update_pose(Pose2D(x=0.0, y=0.0, theta=0.0))
        
        # Set goal
        goal = NavigationGoal(pose=Pose2D(x=5.0, y=3.0, theta=1.57))
        nav_manager.set_goal(goal)
        
        # Simulate navigation
        for i in range(100):
            # Get velocity command
            linear, angular = nav_manager.get_velocity_command()
            
            # Simulate robot motion
            if nav_manager.current_pose:
                dt = 0.1
                new_theta = nav_manager.current_pose.theta + angular * dt
                new_x = nav_manager.current_pose.x + linear * math.cos(new_theta) * dt
                new_y = nav_manager.current_pose.y + linear * math.sin(new_theta) * dt
                nav_manager.update_pose(Pose2D(x=new_x, y=new_y, theta=new_theta))
            
            feedback = nav_manager.get_feedback()
            if feedback:
                print(f"Step {i}: distance={feedback.distance_remaining:.2f}m, state={feedback.state.value}")
            
            if nav_manager.state in [NavigationState.SUCCEEDED, NavigationState.FAILED]:
                break
            
            time.sleep(0.1)
        
        print(f"Final state: {nav_manager.state.value}")


if __name__ == '__main__':
    main()
