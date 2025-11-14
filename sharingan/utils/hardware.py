"""Hardware detection and optimization path selection."""

import os
import platform
from typing import Dict, Any, Optional
import torch


class HardwareAbstraction:
    """Hardware detection and optimization path selection."""

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """
        Detect available hardware capabilities.

        Returns:
            Dict with keys: has_gpu, gpu_memory, cpu_cores, device_type
        """
        info = {
            "platform": platform.system(),
            "cpu_cores": os.cpu_count() or 1,
            "has_gpu": torch.cuda.is_available(),
            "gpu_count": 0,
            "gpu_memory": 0,
            "device_type": "cpu",
        }

        if info["has_gpu"]:
            info["gpu_count"] = torch.cuda.device_count()
            if info["gpu_count"] > 0:
                # Get memory of first GPU in GB
                gpu_props = torch.cuda.get_device_properties(0)
                info["gpu_memory"] = gpu_props.total_memory / (1024 ** 3)
                info["gpu_name"] = gpu_props.name
                info["device_type"] = "gpu"

        # Detect if running on edge device
        if HardwareAbstraction._is_edge_device():
            info["device_type"] = "edge"

        return info

    @staticmethod
    def _is_edge_device() -> bool:
        """Check if running on edge device (Raspberry Pi, Jetson)."""
        # Check for Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry Pi' in f.read():
                    return True
        except:
            pass

        # Check for Jetson
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                if 'Jetson' in f.read():
                    return True
        except:
            pass

        return False

    @staticmethod
    def select_execution_path(user_preference: Optional[str] = None) -> str:
        """
        Select optimal execution path based on hardware.

        Args:
            user_preference: Optional override ("cpu", "gpu", "tiny")

        Returns:
            Execution path: "cpu_optimized", "gpu_accelerated", "tiny_device"
        """
        if user_preference:
            path_map = {
                "cpu": "cpu_optimized",
                "gpu": "gpu_accelerated",
                "tiny": "tiny_device",
                "edge": "tiny_device",
            }
            return path_map.get(user_preference, "cpu_optimized")

        # Auto-detect
        hw_info = HardwareAbstraction.detect_hardware()

        if hw_info["device_type"] == "edge":
            return "tiny_device"
        elif hw_info["has_gpu"] and hw_info["gpu_memory"] >= 2.0:
            return "gpu_accelerated"
        else:
            return "cpu_optimized"

    @staticmethod
    def get_optimal_batch_size(device_type: str) -> int:
        """
        Get optimal batch size for device type.

        Args:
            device_type: Device type string

        Returns:
            Optimal batch size
        """
        batch_size_map = {
            "cpu_optimized": 2,
            "gpu_accelerated": 16,
            "tiny_device": 1,
        }
        return batch_size_map.get(device_type, 4)

    @staticmethod
    def configure_for_device(device_type: str) -> Dict[str, Any]:
        """
        Get device-specific configuration parameters.

        Args:
            device_type: Device type string

        Returns:
            Configuration dictionary
        """
        configs = {
            "cpu_optimized": {
                "device": "cpu",
                "batch_size": 2,
                "target_fps": 5.0,
                "use_fp16": False,
                "enable_quantization": True,
                "temporal_modules": ["gating", "memory"],
                "max_history": 16,
            },
            "gpu_accelerated": {
                "device": "cuda",
                "batch_size": 16,
                "target_fps": 30.0,
                "use_fp16": True,
                "enable_quantization": False,
                "temporal_modules": ["tas", "gating", "tda", "pooling", "memory"],
                "max_history": 32,
            },
            "tiny_device": {
                "device": "cpu",
                "batch_size": 1,
                "target_fps": 2.0,
                "use_fp16": False,
                "enable_quantization": True,
                "temporal_modules": ["tas"],
                "max_history": 8,
            },
        }
        return configs.get(device_type, configs["cpu_optimized"])

    @staticmethod
    def get_device_string() -> str:
        """
        Get PyTorch device string.

        Returns:
            Device string ("cpu" or "cuda")
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def print_hardware_info() -> None:
        """Print hardware information."""
        info = HardwareAbstraction.detect_hardware()
        print("Hardware Information:")
        print(f"  Platform: {info['platform']}")
        print(f"  CPU Cores: {info['cpu_cores']}")
        print(f"  Has GPU: {info['has_gpu']}")
        if info['has_gpu']:
            print(f"  GPU Count: {info['gpu_count']}")
            print(f"  GPU Memory: {info['gpu_memory']:.2f} GB")
            if 'gpu_name' in info:
                print(f"  GPU Name: {info['gpu_name']}")
        print(f"  Device Type: {info['device_type']}")

        execution_path = HardwareAbstraction.select_execution_path()
        print(f"\nRecommended Execution Path: {execution_path}")
