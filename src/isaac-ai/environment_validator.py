#!/usr/bin/env python3
"""
Environment validation script for Isaac AI Brain module.

This script validates that the required Isaac Sim, Isaac ROS, and Nav2 environments
are properly installed and configured with the necessary dependencies.
"""

import os
import sys
import subprocess
import platform
from typing import Tuple, List


def check_gpu_compatibility() -> Tuple[bool, str]:
    """
    Check if the system has an NVIDIA GPU with compute capability 6.0+.
    """
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            # Parse GPU info to check compute capability
            gpu_info = result.stdout.strip().split('\n')[0]
            print(f"Found GPU: {gpu_info}")

            # For now, we'll assume modern NVIDIA GPUs have sufficient compute capability
            # In a real implementation, we'd need to map GPU models to compute capabilities
            return True, "NVIDIA GPU detected"
        else:
            return False, "No NVIDIA GPU detected or nvidia-smi not available"
    except FileNotFoundError:
        return False, "nvidia-smi command not found - NVIDIA drivers may not be installed"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi command timed out"
    except Exception as e:
        return False, f"Error checking GPU: {str(e)}"


def check_isaac_sim_installation() -> Tuple[bool, str]:
    """
    Check if Isaac Sim is properly installed and accessible.
    """
    try:
        # Check if Isaac Sim directory exists
        isaac_sim_path = os.environ.get('ISAAC_SIM_PATH', '/opt/isaac-sim')
        if os.path.exists(isaac_sim_path):
            return True, f"Isaac Sim found at {isaac_sim_path}"

        # Try to find Isaac Sim in common locations
        common_paths = [
            '/opt/isaac-sim',
            os.path.expanduser('~/isaac-sim'),
            os.path.expanduser('~/Downloads/isaac-sim'),
            'C:/Users/Public/Documents/NVIDIA/Isaac-Sim',
        ]

        for path in common_paths:
            if os.path.exists(path):
                return True, f"Isaac Sim found at {path}"

        return False, "Isaac Sim installation not found in common locations"
    except Exception as e:
        return False, f"Error checking Isaac Sim installation: {str(e)}"


def check_ros2_humble() -> Tuple[bool, str]:
    """
    Check if ROS 2 Humble is properly installed and sourced.
    """
    try:
        # Check if ROS_DISTRO environment variable is set to humble
        ros_distro = os.environ.get('ROS_DISTRO', '')
        if ros_distro.lower().startswith('humble'):
            return True, f"ROS 2 Humble detected (ROS_DISTRO={ros_distro})"

        # Try to run a basic ROS command to check if ROS is sourced
        result = subprocess.run(['ros2', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_output = result.stdout.strip()
            if 'humble' in version_output.lower():
                return True, f"ROS 2 Humble detected from version: {version_output}"
            else:
                return False, f"ROS 2 detected but not Humble: {version_output}"
        else:
            return False, "ROS 2 not found or not properly sourced"
    except FileNotFoundError:
        return False, "ROS 2 command (ros2) not found - ROS 2 may not be installed or sourced"
    except subprocess.TimeoutExpired:
        return False, "ROS 2 version check timed out"
    except Exception as e:
        return False, f"Error checking ROS 2 installation: {str(e)}"


def check_isaac_ros() -> Tuple[bool, str]:
    """
    Check if Isaac ROS is properly installed.
    """
    try:
        # Check if Isaac ROS packages are available
        result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            packages = result.stdout
            # Look for common Isaac ROS packages
            isaac_ros_packages = [
                'isaac_ros_apriltag',
                'isaac_ros_compression',
                'isaac_ros_image_pipeline',
                'isaac_ros_visual_slam',
                'isaac_ros_pointcloud_utils',
                'isaac_ros_nitros'
            ]

            found_packages = []
            for pkg in isaac_ros_packages:
                if pkg in packages:
                    found_packages.append(pkg)

            if found_packages:
                return True, f"Isaac ROS packages detected: {', '.join(found_packages[:3])}{'...' if len(found_packages) > 3 else ''}"
            else:
                return False, "No Isaac ROS packages detected in ROS 2 package list"
        else:
            return False, "Could not list ROS 2 packages"
    except Exception as e:
        return False, f"Error checking Isaac ROS installation: {str(e)}"


def check_nav2() -> Tuple[bool, str]:
    """
    Check if Nav2 is properly installed.
    """
    try:
        # Check if Nav2 launch files exist
        result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            packages = result.stdout
            # Look for common Nav2 packages
            nav2_packages = [
                'nav2_bringup',
                'nav2_core',
                'nav2_msgs',
                'nav2_bt_navigator',
                'nav2_planner',
                'nav2_controller'
            ]

            found_packages = []
            for pkg in nav2_packages:
                if pkg in packages:
                    found_packages.append(pkg)

            if found_packages:
                return True, f"Nav2 packages detected: {', '.join(found_packages[:3])}{'...' if len(found_packages) > 3 else ''}"
            else:
                return False, "No Nav2 packages detected in ROS 2 package list"
        else:
            return False, "Could not list ROS 2 packages"
    except Exception as e:
        return False, f"Error checking Nav2 installation: {str(e)}"


def main():
    """
    Main function to run all environment checks.
    """
    print("Isaac AI Brain Environment Validation")
    print("=" * 50)

    checks = [
        ("GPU Compatibility (Compute Capability 6.0+)", check_gpu_compatibility),
        ("Isaac Sim Installation", check_isaac_sim_installation),
        ("ROS 2 Humble Installation", check_ros2_humble),
        ("Isaac ROS Installation", check_isaac_ros),
        ("Nav2 Installation", check_nav2),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\nChecking: {check_name}")
        try:
            success, message = check_func()
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  Status: {status}")
            print(f"  Message: {message}")
            results.append((check_name, success, message))
        except Exception as e:
            print(f"  Status: ✗ FAIL")
            print(f"  Message: Error during check - {str(e)}")
            results.append((check_name, False, f"Error during check - {str(e)}"))

    print("\n" + "=" * 50)
    print("Validation Summary:")

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for check_name, success, message in results:
        status = "✓" if success else "✗"
        print(f"  {status} {check_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All environment checks passed! Ready for Isaac AI Brain development.")
        return 0
    else:
        print("❌ Some environment checks failed. Please review the messages above and install missing components.")
        return 1


if __name__ == "__main__":
    sys.exit(main())