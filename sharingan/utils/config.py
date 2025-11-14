"""Configuration management utilities."""

import os
import yaml
import json
from typing import Dict, Any


class Config:
    """Global configuration management."""

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            path: Path to config file (YAML or JSON)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is not supported
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        elif ext == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    @staticmethod
    def get_default() -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            # Video settings
            "video": {
                "backend": "opencv",
                "sampling_strategy": "uniform",
                "target_fps": 5.0,
            },
            # VLM settings
            "vlm": {
                "model_name": "clip-vit-b32",
                "device": "auto",
                "batch_size": 8,
            },
            # Temporal settings
            "temporal": {
                "enable_tas": True,
                "enable_gating": True,
                "enable_tda": True,
                "enable_pooling": True,
                "enable_memory": True,
                "max_history": 32,
                "num_memory_tokens": 8,
            },
            # Tracking settings
            "tracking": {
                "max_age": 30,
                "min_hits": 3,
            },
            # Event detection settings
            "events": {
                "sensitivity": 0.5,
            },
            # Query settings
            "query": {
                "index_type": "faiss",
                "top_k": 10,
            },
        }

    @staticmethod
    def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config.merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def from_env(prefix: str = "SHARINGAN_") -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variables

        Returns:
            Configuration dictionary from environment
        """
        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                # Try to parse as JSON, otherwise use as string
                try:
                    config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    config[config_key] = value
        return config

    @staticmethod
    def save(config: Dict[str, Any], path: str) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            path: Path to save config file

        Raises:
            ValueError: If file format is not supported
        """
        ext = os.path.splitext(path)[1].lower()

        if ext in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif ext == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
