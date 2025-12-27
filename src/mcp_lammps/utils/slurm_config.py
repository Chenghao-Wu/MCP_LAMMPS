"""
SLURM Configuration - Configuration management for SLURM job submissions.

This module handles loading, validating, and managing SLURM configuration
with support for global defaults and per-simulation overrides.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class SlurmConfig:
    """
    Configuration manager for SLURM job parameters.
    
    Supports:
    - Global default configuration
    - Per-simulation overrides
    - YAML configuration file loading
    - Parameter validation
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "partition": "compute",
        "nodes": 1,
        "ntasks_per_node": 4,
        "time_limit": "24:00:00",
        "memory_per_cpu": "4G",
        "lammps_module": None,  # Optional module load command
        "lammps_executable": "lmp",
        "mpi_command": "mpirun -np {ntasks}",
        "additional_modules": [],  # List of additional modules to load
        "mail_type": None,  # Email notification (NONE, BEGIN, END, FAIL, ALL)
        "mail_user": None,  # Email address for notifications
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize SLURM configuration.
        
        Args:
            config_file: Optional path to YAML configuration file
        """
        self.global_defaults = self.DEFAULT_CONFIG.copy()
        self.per_simulation_overrides: Dict[str, Dict[str, Any]] = {}
        
        if config_file and config_file.exists():
            self.load_config(config_file)
        else:
            logger.info("Using default SLURM configuration")
    
    def load_config(self, config_file: Path) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.warning(f"Empty configuration file: {config_file}")
                return
            
            # Update global defaults
            if "global_defaults" in config_data:
                self.global_defaults.update(config_data["global_defaults"])
                logger.info(f"Loaded global defaults from {config_file}")
            
            # Load per-simulation overrides
            if "per_simulation_overrides" in config_data:
                self.per_simulation_overrides = config_data["per_simulation_overrides"]
                logger.info(f"Loaded {len(self.per_simulation_overrides)} simulation overrides")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            raise
    
    def save_config(self, config_file: Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_file: Path to save YAML configuration
        """
        try:
            config_data = {
                "global_defaults": self.global_defaults,
                "per_simulation_overrides": self.per_simulation_overrides
            }
            
            with open(config_file, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
            raise
    
    def get_config(self, simulation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a simulation.
        
        Args:
            simulation_id: Optional simulation ID for overrides
            
        Returns:
            Configuration dictionary with defaults and any overrides applied
        """
        config = self.global_defaults.copy()
        
        # Apply per-simulation overrides if available
        if simulation_id and simulation_id in self.per_simulation_overrides:
            config.update(self.per_simulation_overrides[simulation_id])
            logger.debug(f"Applied overrides for simulation {simulation_id}")
        
        return config
    
    def set_simulation_override(self, simulation_id: str, overrides: Dict[str, Any]) -> None:
        """
        Set configuration overrides for a specific simulation.
        
        Args:
            simulation_id: Simulation ID
            overrides: Dictionary of configuration overrides
        """
        # Validate overrides
        validated_overrides = self.validate_config(overrides)
        self.per_simulation_overrides[simulation_id] = validated_overrides
        logger.info(f"Set overrides for simulation {simulation_id}")
    
    def remove_simulation_override(self, simulation_id: str) -> None:
        """
        Remove configuration overrides for a simulation.
        
        Args:
            simulation_id: Simulation ID
        """
        if simulation_id in self.per_simulation_overrides:
            del self.per_simulation_overrides[simulation_id]
            logger.info(f"Removed overrides for simulation {simulation_id}")
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        validated = {}
        
        for key, value in config.items():
            if key == "nodes":
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"nodes must be a positive integer, got: {value}")
                validated[key] = value
            
            elif key == "ntasks_per_node":
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"ntasks_per_node must be a positive integer, got: {value}")
                validated[key] = value
            
            elif key == "time_limit":
                if not self._validate_time_format(value):
                    raise ValueError(f"Invalid time_limit format: {value}. Use HH:MM:SS or D-HH:MM:SS")
                validated[key] = value
            
            elif key == "memory_per_cpu":
                if not self._validate_memory_format(value):
                    raise ValueError(f"Invalid memory_per_cpu format: {value}. Use number with G/M suffix")
                validated[key] = value
            
            elif key == "partition":
                if not isinstance(value, str) or not value:
                    raise ValueError(f"partition must be a non-empty string, got: {value}")
                validated[key] = value
            
            elif key == "mail_type":
                valid_types = ["NONE", "BEGIN", "END", "FAIL", "ALL", None]
                if value not in valid_types:
                    raise ValueError(f"mail_type must be one of {valid_types}, got: {value}")
                validated[key] = value
            
            elif key in ["lammps_executable", "lammps_module", "mpi_command", "mail_user"]:
                validated[key] = value
            
            elif key == "additional_modules":
                if not isinstance(value, list):
                    raise ValueError(f"additional_modules must be a list, got: {type(value)}")
                validated[key] = value
            
            else:
                # Allow other keys but log a warning
                logger.warning(f"Unknown configuration key: {key}")
                validated[key] = value
        
        return validated
    
    @staticmethod
    def _validate_time_format(time_str: str) -> bool:
        """
        Validate SLURM time format.
        
        Args:
            time_str: Time string to validate
            
        Returns:
            True if valid, False otherwise
        """
        import re
        
        # Support formats: HH:MM:SS, D-HH:MM:SS, MM:SS, MM
        patterns = [
            r'^\d+:\d{2}:\d{2}$',  # HH:MM:SS
            r'^\d+-\d+:\d{2}:\d{2}$',  # D-HH:MM:SS
            r'^\d+:\d{2}$',  # MM:SS
            r'^\d+$',  # MM
        ]
        
        return any(re.match(pattern, time_str) for pattern in patterns)
    
    @staticmethod
    def _validate_memory_format(memory_str: str) -> bool:
        """
        Validate SLURM memory format.
        
        Args:
            memory_str: Memory string to validate
            
        Returns:
            True if valid, False otherwise
        """
        import re
        
        # Support formats: 4G, 4000M, 4096K, etc.
        pattern = r'^\d+[KMGT]?$'
        return bool(re.match(pattern, memory_str))
    
    def get_total_tasks(self, simulation_id: Optional[str] = None) -> int:
        """
        Calculate total number of MPI tasks.
        
        Args:
            simulation_id: Optional simulation ID
            
        Returns:
            Total number of tasks (nodes * ntasks_per_node)
        """
        config = self.get_config(simulation_id)
        return config["nodes"] * config["ntasks_per_node"]
    
    def format_mpi_command(self, simulation_id: Optional[str] = None) -> str:
        """
        Format MPI command with total tasks.
        
        Args:
            simulation_id: Optional simulation ID
            
        Returns:
            Formatted MPI command string
        """
        config = self.get_config(simulation_id)
        ntasks = self.get_total_tasks(simulation_id)
        return config["mpi_command"].format(ntasks=ntasks)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "global_defaults": self.global_defaults,
            "per_simulation_overrides": self.per_simulation_overrides
        }
    
    @classmethod
    def create_default_config_file(cls, config_file: Path) -> None:
        """
        Create a default configuration file.
        
        Args:
            config_file: Path where to create the config file
        """
        default_config = {
            "global_defaults": cls.DEFAULT_CONFIG.copy(),
            "per_simulation_overrides": {}
        }
        
        try:
            with open(config_file, 'w') as f:
                yaml.safe_dump(default_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Created default configuration file: {config_file}")
            
        except Exception as e:
            logger.error(f"Error creating default configuration file: {e}")
            raise

