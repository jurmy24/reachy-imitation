import os
import yaml
from pathlib import Path


def get_project_root():
    """Get the absolute path to the project root directory."""
    # Assuming this file is in src/utils
    current_file = Path(__file__)
    # Go up 3 levels: utils -> src -> project_root
    return current_file.parent.parent.parent


def load_config(config_file):
    """
    Load a YAML configuration file.

    Args:
        config_file (str): Relative path to the config file from the project root

    Returns:
        dict: Configuration parameters
    """
    root_dir = get_project_root()
    config_path = os.path.join(root_dir, config_file)

    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return {}


def get_robot_dimensions():
    """
    Get the Reachy robot dimensions from the configuration file.

    Returns:
        dict: Robot dimensions
    """
    return load_config("config/robot_model.yaml")
