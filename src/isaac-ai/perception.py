#!/usr/bin/env python3
"""
Sensor Fusion Module for Isaac AI Brain

This module implements sensor fusion algorithms combining camera, LiDAR, and IMU data
for robust environmental perception and state estimation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import threading
from collections import deque
import time


class SensorType(Enum):
    """Enumeration of supported sensor types."""
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    VSLAM = "vslam"


@dataclass
class SensorMeasurement:
    """Container for sensor measurements."""
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray
    covariance: Optional[np.ndarray] = None
    frame_id: str = "base_link"


@dataclass
class FusedState:
    """Container for fused robot state."""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # Quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    covariance: np.ndarray  # 12x12 state covariance matrix
    timestamp: float
    confidence: float = 1.0


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for multi-sensor fusion.
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    """
    
    def __init__(self):
        # State dimension
        self.state_dim = 13
        
        # Initialize state
        self.state = np.zeros(self.state_dim)
        self.state[6] = 1.0  # quaternion w component
        
        # Initialize covariance
        self.covariance = np.eye(self.state_dim) * 0.1
        
        # Process noise
        self.process_noise = np.eye(self.state_dim) * 0.01
        self.process_noise[0:3, 0:3] *= 0.001  # Position noise
        self.process_noise[3:6, 3:6] *= 0.01   # Velocity noise
        self.process_noise[6:10, 6:10] *= 0.001  # Orientation noise
        self.process_noise[10:13, 10:13] *= 0.01  # Angular velocity noise
        
        # Measurement noise (default values, can be overridden)
        self.measurement_noise = {
            SensorType.VSLAM: np.eye(7) * 0.01,  # Position + orientation
            SensorType.IMU: np.eye(6) * 0.001,   # Angular velocity + linear acceleration
            SensorType.LIDAR: np.eye(3) * 0.05,  # Position from scan matching
        }
        
        self.last_update_time = None
        self.lock = threading.Lock()
    
    def predict(self, dt: float):
        """
        Predict step of the EKF.
        
        Args:
            dt: Time step in seconds
        """
        with self.lock:
            # Extract state components
            position = self.state[0:3]
            velocity = self.state[3:6]
            quaternion = self.state[6:10]
            angular_velocity = self.state[10:13]
            
            # Predict new state
            # Position update
            new_position = position + velocity * dt
            
            # Orientation update using angular velocity
            omega_mag = np.linalg.norm(angular_velocity)
            if omega_mag > 1e-10:
                axis = angular_velocity / omega_mag
                angle = omega_mag * dt
                dq = np.array([
                    np.cos(angle / 2),
                    axis[0] * np.sin(angle / 2),
                    axis[1] * np.sin(angle / 2),
                    axis[2] * np.sin(angle / 2)
                ])
                new_quaternion = self._quaternion_multiply(quaternion, dq)
                new_quaternion = new_quaternion / np.linalg.norm(new_quaternion)
            else:
                new_quaternion = quaternion
            
            # Update state
            self.state[0:3] = new_position
            self.state[6:10] = new_quaternion
            
            # Compute Jacobian of state transition
            F = self._compute_state_jacobian(dt)
            
            # Update covariance
            self.covariance = F @ self.covariance @ F.T + self.process_noise * dt
    
    def update_vslam(self, position: np.ndarray, orientation: np.ndarray, 
                     covariance: Optional[np.ndarray] = None):
        """
        Update with VSLAM measurement (position + orientation).
        
        Args:
            position: [x, y, z] position estimate
            orientation: [w, x, y, z] quaternion orientation
            covariance: Optional 7x7 measurement covariance
        """
        with self.lock:
            # Measurement vector
            z = np.concatenate([position, orientation])
            
            # Measurement matrix (maps state to measurement)
            H = np.zeros((7, self.state_dim))
            H[0:3, 0:3] = np.eye(3)  # Position
            H[3:7, 6:10] = np.eye(4)  # Orientation
            
            # Measurement noise
            R = covariance if covariance is not None else self.measurement_noise[SensorType.VSLAM]
            
            # Innovation
            z_pred = H @ self.state
            y = z - z_pred
            
            # Handle quaternion sign ambiguity
            if np.dot(z[3:7], z_pred[3:7]) < 0:
                y[3:7] = z[3:7] + z_pred[3:7]
            
            # Innovation covariance
            S = H @ self.covariance @ H.T + R
            
            # Kalman gain
            K = self.covariance @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.state = self.state + K @ y
            
            # Normalize quaternion
            self.state[6:10] = self.state[6:10] / np.linalg.norm(self.state[6:10])
            
            # Update covariance
            I = np.eye(self.state_dim)
            self.covariance = (I - K @ H) @ self.covariance
    
    def update_imu(self, angular_velocity: np.ndarray, linear_acceleration: np.ndarray,
                   covariance: Optional[np.ndarray] = None):
        """
        Update with IMU measurement.
        
        Args:
            angular_velocity: [wx, wy, wz] angular velocity
            linear_acceleration: [ax, ay, az] linear acceleration
            covariance: Optional 6x6 measurement covariance
        """
        with self.lock:
            # Update angular velocity directly (high update rate)
            self.state[10:13] = angular_velocity
            
            # Use linear acceleration for velocity correction
            # This is a simplified approach; full IMU integration would be more complex
            orientation = self.state[6:10]
            
            # Rotate acceleration to world frame and remove gravity
            gravity = np.array([0, 0, 9.81])
            accel_world = self._rotate_vector(linear_acceleration, orientation)
            accel_world -= gravity
            
            # Simple velocity update (would use proper integration in production)
            # Here we just use acceleration as a correction signal
            alpha = 0.1  # Blending factor
            dt = 0.01  # Assumed IMU rate
            self.state[3:6] = (1 - alpha) * self.state[3:6] + alpha * (self.state[3:6] + accel_world * dt)
    
    def update_lidar(self, position: np.ndarray, covariance: Optional[np.ndarray] = None):
        """
        Update with LiDAR-based position estimate (e.g., from scan matching).
        
        Args:
            position: [x, y, z] position estimate
            covariance: Optional 3x3 measurement covariance
        """
        with self.lock:
            # Measurement matrix
            H = np.zeros((3, self.state_dim))
            H[0:3, 0:3] = np.eye(3)
            
            # Measurement noise
            R = covariance if covariance is not None else self.measurement_noise[SensorType.LIDAR]
            
            # Innovation
            z_pred = H @ self.state
            y = position - z_pred
            
            # Innovation covariance
            S = H @ self.covariance @ H.T + R
            
            # Kalman gain
            K = self.covariance @ H.T @ np.linalg.inv(S)
            
            # Update state and covariance
            self.state = self.state + K @ y
            I = np.eye(self.state_dim)
            self.covariance = (I - K @ H) @ self.covariance
    
    def get_state(self) -> FusedState:
        """Get the current fused state."""
        with self.lock:
            # Compute confidence based on covariance trace
            position_uncertainty = np.trace(self.covariance[0:3, 0:3])
            confidence = 1.0 / (1.0 + position_uncertainty)
            
            return FusedState(
                position=self.state[0:3].copy(),
                velocity=self.state[3:6].copy(),
                orientation=self.state[6:10].copy(),
                angular_velocity=self.state[10:13].copy(),
                covariance=self.covariance.copy(),
                timestamp=time.time(),
                confidence=confidence
            )
    
    def _compute_state_jacobian(self, dt: float) -> np.ndarray:
        """Compute the Jacobian of the state transition function."""
        F = np.eye(self.state_dim)
        
        # Position depends on velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Orientation depends on angular velocity (simplified)
        # Full implementation would linearize the quaternion update
        
        return F
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _rotate_vector(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        v_quat = np.array([0, v[0], v[1], v[2]])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        
        result = self._quaternion_multiply(
            self._quaternion_multiply(q, v_quat),
            q_conj
        )
        
        return result[1:4]


class SensorFusionManager:
    """
    High-level manager for multi-sensor fusion.
    
    Coordinates data from multiple sensors and maintains the fused state estimate.
    """
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize the sensor fusion manager.
        
        Args:
            buffer_size: Maximum number of measurements to buffer per sensor
        """
        self.ekf = ExtendedKalmanFilter()
        self.buffer_size = buffer_size
        
        # Measurement buffers for each sensor type
        self.buffers: Dict[SensorType, deque] = {
            sensor_type: deque(maxlen=buffer_size)
            for sensor_type in SensorType
        }
        
        # Sensor status tracking
        self.sensor_status: Dict[SensorType, Dict] = {
            sensor_type: {
                'last_update': None,
                'update_rate': 0.0,
                'is_active': False,
                'measurement_count': 0
            }
            for sensor_type in SensorType
        }
        
        # Configuration
        self.prediction_rate = 100.0  # Hz
        self.last_prediction_time = None
        
        self.lock = threading.Lock()
    
    def process_measurement(self, measurement: SensorMeasurement):
        """
        Process a new sensor measurement.
        
        Args:
            measurement: The sensor measurement to process
        """
        with self.lock:
            # Add to buffer
            self.buffers[measurement.sensor_type].append(measurement)
            
            # Update sensor status
            status = self.sensor_status[measurement.sensor_type]
            if status['last_update'] is not None:
                dt = measurement.timestamp - status['last_update']
                if dt > 0:
                    status['update_rate'] = 0.9 * status['update_rate'] + 0.1 * (1.0 / dt)
            status['last_update'] = measurement.timestamp
            status['is_active'] = True
            status['measurement_count'] += 1
        
        # Perform prediction step
        self._predict(measurement.timestamp)
        
        # Perform update based on sensor type
        if measurement.sensor_type == SensorType.VSLAM:
            position = measurement.data[0:3]
            orientation = measurement.data[3:7]
            self.ekf.update_vslam(position, orientation, measurement.covariance)
        
        elif measurement.sensor_type == SensorType.IMU:
            angular_velocity = measurement.data[0:3]
            linear_acceleration = measurement.data[3:6]
            self.ekf.update_imu(angular_velocity, linear_acceleration, measurement.covariance)
        
        elif measurement.sensor_type == SensorType.LIDAR:
            position = measurement.data[0:3]
            self.ekf.update_lidar(position, measurement.covariance)
    
    def _predict(self, current_time: float):
        """Perform prediction step up to current time."""
        if self.last_prediction_time is None:
            self.last_prediction_time = current_time
            return
        
        dt = current_time - self.last_prediction_time
        if dt > 0:
            self.ekf.predict(dt)
            self.last_prediction_time = current_time
    
    def get_fused_state(self) -> FusedState:
        """Get the current fused robot state."""
        return self.ekf.get_state()
    
    def get_sensor_status(self) -> Dict[SensorType, Dict]:
        """Get the status of all sensors."""
        with self.lock:
            # Update is_active based on time since last update
            current_time = time.time()
            for sensor_type, status in self.sensor_status.items():
                if status['last_update'] is not None:
                    time_since_update = current_time - status['last_update']
                    expected_interval = 1.0 / max(status['update_rate'], 1.0)
                    status['is_active'] = time_since_update < (expected_interval * 5)
            
            return {k: v.copy() for k, v in self.sensor_status.items()}
    
    def reset(self):
        """Reset the fusion state."""
        with self.lock:
            self.ekf = ExtendedKalmanFilter()
            for buffer in self.buffers.values():
                buffer.clear()
            for status in self.sensor_status.values():
                status['last_update'] = None
                status['update_rate'] = 0.0
                status['is_active'] = False
                status['measurement_count'] = 0
            self.last_prediction_time = None


# ROS 2 Node wrapper (for integration with ROS 2 system)
def create_perception_node():
    """
    Create and return a ROS 2 perception node.
    
    This is a factory function that creates the node when ROS 2 is available.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from sensor_msgs.msg import Imu
        from nav_msgs.msg import Odometry
        
        class PerceptionNode(Node):
            """ROS 2 node for sensor fusion."""
            
            def __init__(self):
                super().__init__('perception_fusion_node')
                
                self.fusion_manager = SensorFusionManager()
                
                # Create subscribers
                self.vslam_sub = self.create_subscription(
                    PoseWithCovarianceStamped,
                    '/visual_slam/pose',
                    self.vslam_callback,
                    10
                )
                
                self.imu_sub = self.create_subscription(
                    Imu,
                    '/imu/data',
                    self.imu_callback,
                    10
                )
                
                # Create publisher
                self.fused_pose_pub = self.create_publisher(
                    Odometry,
                    '/fused_pose',
                    10
                )
                
                # Create timer for publishing
                self.timer = self.create_timer(0.02, self.publish_fused_state)
                
                self.get_logger().info('Perception Fusion Node initialized')
            
            def vslam_callback(self, msg):
                """Process VSLAM pose measurement."""
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                
                position = np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ])
                
                orientation = np.array([
                    msg.pose.pose.orientation.w,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z
                ])
                
                data = np.concatenate([position, orientation])
                covariance = np.array(msg.pose.covariance).reshape(6, 6)
                
                # Expand to 7x7 for position + quaternion
                full_covariance = np.eye(7) * 0.01
                full_covariance[0:3, 0:3] = covariance[0:3, 0:3]
                full_covariance[3:6, 3:6] = covariance[3:6, 3:6]
                
                measurement = SensorMeasurement(
                    sensor_type=SensorType.VSLAM,
                    timestamp=timestamp,
                    data=data,
                    covariance=full_covariance,
                    frame_id=msg.header.frame_id
                )
                
                self.fusion_manager.process_measurement(measurement)
            
            def imu_callback(self, msg):
                """Process IMU measurement."""
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                
                angular_velocity = np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])
                
                linear_acceleration = np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])
                
                data = np.concatenate([angular_velocity, linear_acceleration])
                
                measurement = SensorMeasurement(
                    sensor_type=SensorType.IMU,
                    timestamp=timestamp,
                    data=data,
                    frame_id=msg.header.frame_id
                )
                
                self.fusion_manager.process_measurement(measurement)
            
            def publish_fused_state(self):
                """Publish the fused robot state."""
                state = self.fusion_manager.get_fused_state()
                
                odom_msg = Odometry()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = 'odom'
                odom_msg.child_frame_id = 'base_link'
                
                odom_msg.pose.pose.position.x = state.position[0]
                odom_msg.pose.pose.position.y = state.position[1]
                odom_msg.pose.pose.position.z = state.position[2]
                
                odom_msg.pose.pose.orientation.w = state.orientation[0]
                odom_msg.pose.pose.orientation.x = state.orientation[1]
                odom_msg.pose.pose.orientation.y = state.orientation[2]
                odom_msg.pose.pose.orientation.z = state.orientation[3]
                
                odom_msg.twist.twist.linear.x = state.velocity[0]
                odom_msg.twist.twist.linear.y = state.velocity[1]
                odom_msg.twist.twist.linear.z = state.velocity[2]
                
                odom_msg.twist.twist.angular.x = state.angular_velocity[0]
                odom_msg.twist.twist.angular.y = state.angular_velocity[1]
                odom_msg.twist.twist.angular.z = state.angular_velocity[2]
                
                self.fused_pose_pub.publish(odom_msg)
        
        return PerceptionNode
        
    except ImportError:
        return None


def main():
    """Main entry point for the perception module."""
    try:
        import rclpy
        
        rclpy.init()
        
        PerceptionNode = create_perception_node()
        if PerceptionNode is not None:
            node = PerceptionNode()
            rclpy.spin(node)
            node.destroy_node()
        
        rclpy.shutdown()
        
    except ImportError:
        print("ROS 2 not available. Running in standalone mode.")
        
        # Standalone demo
        fusion_manager = SensorFusionManager()
        
        # Simulate some measurements
        for i in range(100):
            # Simulate VSLAM measurement
            vslam_measurement = SensorMeasurement(
                sensor_type=SensorType.VSLAM,
                timestamp=i * 0.1,
                data=np.array([i * 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            )
            fusion_manager.process_measurement(vslam_measurement)
            
            # Simulate IMU measurements (higher rate)
            for j in range(10):
                imu_measurement = SensorMeasurement(
                    sensor_type=SensorType.IMU,
                    timestamp=i * 0.1 + j * 0.01,
                    data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.81]),
                )
                fusion_manager.process_measurement(imu_measurement)
        
        # Print final state
        state = fusion_manager.get_fused_state()
        print(f"Final position: {state.position}")
        print(f"Final velocity: {state.velocity}")
        print(f"Final orientation: {state.orientation}")
        print(f"Confidence: {state.confidence:.3f}")


if __name__ == '__main__':
    main()
