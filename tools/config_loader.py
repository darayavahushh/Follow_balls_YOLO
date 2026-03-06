"""
=============================================================================
Configuration Loader
=============================================================================
Utilities for loading and processing YAML configuration files.

Usage:
    from tools.config_loader import load_config, resolve_paths
    
    config = load_config("configs/config.yaml")
    config = resolve_paths(config, project_root)
=============================================================================
"""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary with all settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file contains invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    return config


def resolve_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    Resolve relative paths in config to absolute paths.
    
    Converts all paths in the 'paths' section of the config from relative
    to absolute paths based on the project root directory.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory (absolute path)
        
    Returns:
        Config with resolved absolute paths
    """
    paths = config.get('paths', {})
    
    for key, value in paths.items():
        if value and not Path(value).is_absolute():
            paths[key] = str(project_root / value)
    
    return config


def get_nested_config(config: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    Safely get nested configuration value.
    
    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to traverse
        default: Default value if key path doesn't exist
        
    Returns:
        Configuration value or default
    """
    result = config
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    
    return result
