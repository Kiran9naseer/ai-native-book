#!/usr/bin/env python3
"""
Isaac Sim to ROS 2 Bridge Utilities

This module provides bridge utilities for connecting NVIDIA Isaac Sim
simulation with ROS 2 for humanoid robot control and perception.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any, Tuple
from enum import Enum
import threading
import time
import json
import struct


class SimulationState(Enum):
    """Simulation state enumeration."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class SimulatorConfig:
    """Configuration for Isaac Sim connection."""
    host: str = "localhost"
    port: int = 8211
    use_gpu: bool = True
    physics_dt: float = 1.0 / 60.0
    render_dt: float = 1.0 / 30.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class RobotState:
    """Robot state from simulation."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # Quaternion [w, x, y, z]
    linear_velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass  
class SensorData:
    """Sensor data from simulation."""
    sensor_name: str
    sensor_type: str  # "camera", "lidar", "imu"
    data: Any
    timestamp: float
    frame_id: str = ""


class IsaacSimBridge:
    """
    Bridge between Isaac Sim and ROS 2.
    
    Provides utilities for:
    - Connecting to Isaac Sim
    - Receiving robot state and sensor data
    - Sending control commands
    - Time synchronization
    """
    
    def __init__(self, config: Optional[SimulatorConfig] = None):
        """
        Initialize the Isaac Sim bridge.
        
        Args:
            config: Simulator configuration
        """
        self.config = config or SimulatorConfig()
        self.state = SimulationState.STOPPED
        
        # State storage
        self.robot_state: Optional[RobotState] = None
        self.sensor_data: Dict[str, SensorData] = {}
        
        # Callbacks
        self.on_robot_state: Optional[Callable[[RobotState], None]] = None
        self.on_sensor_data: Optional[Callable[[SensorData], None]] = None
        self.on_state_change: Optional[Callable[[SimulationState], None]] = None
        
        # Threading
        self.lock = threading.Lock()
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Time synchronization
        self.sim_time = 0.0
        self.real_time_offset = 0.0
    
    def connect(self) -> bool:
        """
        Connect to Isaac Sim.
        
        Returns:
            True if connection successful
        """
        try:
            # In a real implementation, this would establish connection
            # to Isaac Sim via its Python API or ROS 2 bridge
            print(f"Connecting to Isaac Sim at {self.config.host}:{self.config.port}")
            
            # Simulate connection
            self._set_state(SimulationState.STOPPED)
            self.real_time_offset = time.time()
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to Isaac Sim: {e}")
            self._set_state(SimulationState.ERROR)
            return False
    
    def disconnect(self):
        """Disconnect from Isaac Sim."""
        self.stop()
        self._set_state(SimulationState.STOPPED)
    
    def start(self):
        """Start the simulation and data streaming."""
        if self.state == SimulationState.ERROR:
            return
        
        self.running = True
        self._set_state(SimulationState.RUNNING)
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def stop(self):
        """Stop the simulation and data streaming."""
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            self.update_thread = None
        
        self._set_state(SimulationState.STOPPED)
    
    def pause(self):
        """Pause the simulation."""
        self._set_state(SimulationState.PAUSED)
    
    def resume(self):
        """Resume the simulation."""
        if self.state == SimulationState.PAUSED:
            self._set_state(SimulationState.RUNNING)
    
    def get_robot_state(self) -> Optional[RobotState]:
        """Get the current robot state."""
        with self.lock:
            return self.robot_state
    
    def get_sensor_data(self, sensor_name: str) -> Optional[SensorData]:
        """Get data from a specific sensor."""
        with self.lock:
            return self.sensor_data.get(sensor_name)
    
    def send_joint_command(self, joint_positions: Dict[str, float]):
        """
        Send joint position commands to the robot.
        
        Args:
            joint_positions: Dictionary of joint_name -> target_position
        """
        # In a real implementation, this would send commands to Isaac Sim
        pass
    
    def send_velocity_command(self, linear: np.ndarray, angular: np.ndarray):
        """
        Send velocity commands to the robot.
        
        Args:
            linear: Linear velocity [vx, vy, vz]
            angular: Angular velocity [wx, wy, wz]
        """
        # In a real implementation, this would send commands to Isaac Sim
        pass
    
    def get_sim_time(self) -> float:
        """Get the current simulation time."""
        return self.sim_time
    
    def _update_loop(self):
        """Main update loop for receiving data from Isaac Sim."""
        while self.running:
            if self.state == SimulationState.RUNNING:
                self._update_robot_state()
                self._update_sensor_data()
                self.sim_time += self.config.physics_dt
            
            time.sleep(self.config.physics_dt)
    
    def _update_robot_state(self):
        """Update robot state from simulation."""
        # Simulate robot state (in real implementation, get from Isaac Sim)
        with self.lock:
            self.robot_state = RobotState(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                linear_velocity=np.array([0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                joint_positions={},
                joint_velocities={},
                timestamp=self.sim_time
            )
        
        if self.on_robot_state and self.robot_state:
            self.on_robot_state(self.robot_state)
    
    def _update_sensor_data(self):
        """Update sensor data from simulation."""
        # Simulate sensor updates
        sensors = ['camera_rgb', 'camera_depth', 'lidar', 'imu']
        
        for sensor_name in sensors:
            sensor_type = 'camera' if 'camera' in sensor_name else sensor_name
            
            with self.lock:
                self.sensor_data[sensor_name] = SensorData(
                    sensor_name=sensor_name,
                    sensor_type=sensor_type,
                    data=None,  # Actual data would come from Isaac Sim
                    timestamp=self.sim_time,
                    frame_id=f"{sensor_name}_link"
                )
            
            if self.on_sensor_data:
                self.on_sensor_data(self.sensor_data[sensor_name])
    
    def _set_state(self, new_state: SimulationState):
        """Set simulation state and trigger callback."""
        old_state = self.state
        self.state = new_state
        
        if self.on_state_change and old_state != new_state:
            self.on_state_change(new_state)


class ROS2Bridge:
    """
    ROS 2 side of the Isaac Sim bridge.
    
    Publishes simulation data to ROS 2 topics and receives commands.
    """
    
    def __init__(self, sim_bridge: IsaacSimBridge):
        """
        Initialize the ROS 2 bridge.
        
        Args:
            sim_bridge: Isaac Sim bridge instance
        """
        self.sim_bridge = sim_bridge
        self.node = None
        
        # Set up callbacks
        self.sim_bridge.on_robot_state = self._on_robot_state
        self.sim_bridge.on_sensor_data = self._on_sensor_data
    
    def initialize_ros(self):
        """Initialize ROS 2 node and publishers/subscribers."""
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image, PointCloud2, Imu, JointState
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Twist, TransformStamped
            from tf2_ros import TransformBroadcaster
            
            class SimBridgeNode(Node):
                def __init__(self, parent_bridge):
                    super().__init__('isaac_sim_bridge')
                    self.parent = parent_bridge
                    
                    # Publishers
                    self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
                    self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
                    self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 10)
                    self.camera_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
                    self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
                    self.lidar_pub = self.create_publisher(PointCloud2, '/lidar/points', 10)
                    
                    # Subscribers
                    self.cmd_vel_sub = self.create_subscription(
                        Twist, '/cmd_vel', self._cmd_vel_callback, 10
                    )
                    
                    # TF broadcaster
                    self.tf_broadcaster = TransformBroadcaster(self)
                    
                    # Timer for clock publishing
                    self.clock_timer = self.create_timer(0.01, self._publish_clock)
                    
                    self.get_logger().info('Isaac Sim Bridge Node initialized')
                
                def _cmd_vel_callback(self, msg):
                    linear = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
                    angular = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
                    self.parent.sim_bridge.send_velocity_command(linear, angular)
                
                def _publish_clock(self):
                    # Publish simulation clock for use_sim_time
                    pass
                
                def publish_odom(self, state: RobotState):
                    msg = Odometry()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = 'odom'
                    msg.child_frame_id = 'base_link'
                    
                    msg.pose.pose.position.x = state.position[0]
                    msg.pose.pose.position.y = state.position[1]
                    msg.pose.pose.position.z = state.position[2]
                    
                    msg.pose.pose.orientation.w = state.orientation[0]
                    msg.pose.pose.orientation.x = state.orientation[1]
                    msg.pose.pose.orientation.y = state.orientation[2]
                    msg.pose.pose.orientation.z = state.orientation[3]
                    
                    msg.twist.twist.linear.x = state.linear_velocity[0]
                    msg.twist.twist.linear.y = state.linear_velocity[1]
                    msg.twist.twist.linear.z = state.linear_velocity[2]
                    
                    msg.twist.twist.angular.x = state.angular_velocity[0]
                    msg.twist.twist.angular.y = state.angular_velocity[1]
                    msg.twist.twist.angular.z = state.angular_velocity[2]
                    
                    self.odom_pub.publish(msg)
                    
                    # Broadcast TF
                    t = TransformStamped()
                    t.header = msg.header
                    t.child_frame_id = 'base_link'
                    t.transform.translation.x = state.position[0]
                    t.transform.translation.y = state.position[1]
                    t.transform.translation.z = state.position[2]
                    t.transform.rotation = msg.pose.pose.orientation
                    self.tf_broadcaster.sendTransform(t)
                
                def publish_joint_states(self, state: RobotState):
                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.name = list(state.joint_positions.keys())
                    msg.position = list(state.joint_positions.values())
                    msg.velocity = list(state.joint_velocities.values())
                    self.joint_state_pub.publish(msg)
            
            self.node = SimBridgeNode(self)
            return True
            
        except ImportError:
            print("ROS 2 not available")
            return False
    
    def _on_robot_state(self, state: RobotState):
        """Callback when robot state is updated."""
        if self.node:
            self.node.publish_odom(state)
            self.node.publish_joint_states(state)
    
    def _on_sensor_data(self, data: SensorData):
        """Callback when sensor data is updated."""
        if not self.node:
            return
        
        # Publish sensor data to appropriate topics
        # Implementation depends on sensor type
        pass
    
    def spin(self):
        """Spin the ROS 2 node."""
        if self.node:
            import rclpy
            rclpy.spin(self.node)


class SyntheticDataGenerator:
    """
    Generator for synthetic training data from Isaac Sim.
    
    Captures simulation data for AI model training.
    """
    
    def __init__(self, sim_bridge: IsaacSimBridge, output_dir: str = "./synthetic_data"):
        """
        Initialize the synthetic data generator.
        
        Args:
            sim_bridge: Isaac Sim bridge instance
            output_dir: Directory for saving generated data
        """
        self.sim_bridge = sim_bridge
        self.output_dir = output_dir
        self.frame_count = 0
        self.dataset_metadata: List[Dict] = []
    
    def capture_frame(self) -> Dict:
        """
        Capture a single frame of data.
        
        Returns:
            Dictionary containing captured data
        """
        robot_state = self.sim_bridge.get_robot_state()
        camera_data = self.sim_bridge.get_sensor_data('camera_rgb')
        depth_data = self.sim_bridge.get_sensor_data('camera_depth')
        lidar_data = self.sim_bridge.get_sensor_data('lidar')
        
        frame_data = {
            'frame_id': self.frame_count,
            'timestamp': self.sim_bridge.get_sim_time(),
            'robot_state': {
                'position': robot_state.position.tolist() if robot_state else None,
                'orientation': robot_state.orientation.tolist() if robot_state else None,
            } if robot_state else None,
            'sensors': {
                'camera_rgb': camera_data.frame_id if camera_data else None,
                'camera_depth': depth_data.frame_id if depth_data else None,
                'lidar': lidar_data.frame_id if lidar_data else None,
            }
        }
        
        self.dataset_metadata.append(frame_data)
        self.frame_count += 1
        
        return frame_data
    
    def save_dataset(self, name: str = "dataset"):
        """
        Save the captured dataset.
        
        Args:
            name: Dataset name
        """
        import os
        
        dataset_path = os.path.join(self.output_dir, name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'name': name,
                'frame_count': self.frame_count,
                'frames': self.dataset_metadata
            }, f, indent=2)
        
        print(f"Dataset saved to {dataset_path}")


def main():
    """Main entry point."""
    # Create bridge configuration
    config = SimulatorConfig(
        host="localhost",
        port=8211,
        use_gpu=True,
        physics_dt=1.0 / 60.0
    )
    
    # Create Isaac Sim bridge
    sim_bridge = IsaacSimBridge(config)
    
    # Connect to simulator
    if not sim_bridge.connect():
        print("Failed to connect to Isaac Sim")
        return
    
    try:
        # Try to initialize ROS 2 bridge
        ros_bridge = ROS2Bridge(sim_bridge)
        
        if ros_bridge.initialize_ros():
            # Start simulation
            sim_bridge.start()
            
            # Spin ROS 2 node
            ros_bridge.spin()
        else:
            # Run in standalone mode
            print("Running in standalone mode (no ROS 2)")
            sim_bridge.start()
            
            # Generate some synthetic data
            data_gen = SyntheticDataGenerator(sim_bridge)
            
            for i in range(100):
                data_gen.capture_frame()
                time.sleep(0.1)
            
            data_gen.save_dataset("humanoid_test")
            
    except KeyboardInterrupt:
        pass
    finally:
        sim_bridge.stop()
        sim_bridge.disconnect()


if __name__ == '__main__':
    main()
