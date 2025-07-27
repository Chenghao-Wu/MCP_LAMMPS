"""
Validators - Utility functions for validating input data.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def validate_simulation_name(name: str) -> bool:
    """
    Validate simulation name.
    
    Args:
        name: Simulation name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or not isinstance(name, str):
        return False
    
    # Check length
    if len(name) < 1 or len(name) > 100:
        return False
    
    # Check for invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, name):
        return False
    
    return True


def validate_temperature(temperature: float) -> bool:
    """
    Validate temperature value.
    
    Args:
        temperature: Temperature to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(temperature, (int, float)):
        return False
    
    # Check reasonable range (0K to 10000K)
    if temperature < 0 or temperature > 10000:
        return False
    
    return True


def validate_pressure(pressure: float) -> bool:
    """
    Validate pressure value.
    
    Args:
        pressure: Pressure to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(pressure, (int, float)):
        return False
    
    # Check reasonable range (0 atm to 10000 atm)
    if pressure < 0 or pressure > 10000:
        return False
    
    return True


def validate_timestep(timestep: float) -> bool:
    """
    Validate timestep value.
    
    Args:
        timestep: Timestep to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(timestep, (int, float)):
        return False
    
    # Check reasonable range (0.0001 ps to 1 ps)
    if timestep < 0.0001 or timestep > 1.0:
        return False
    
    return True


def validate_steps(steps: int) -> bool:
    """
    Validate number of steps.
    
    Args:
        steps: Number of steps to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(steps, int):
        return False
    
    # Check reasonable range (1 to 1000000000)
    if steps < 1 or steps > 1000000000:
        return False
    
    return True


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: File path to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not file_path:
        return False
    
    try:
        path = Path(file_path)
        return True
    except Exception:
        return False


def validate_force_field(force_field: str) -> bool:
    """
    Validate force field name.
    
    Args:
        force_field: Force field name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(force_field, str):
        return False
    
    # Common LAMMPS force fields
    valid_force_fields = [
        "lj/cut", "lj/cut/coul/cut", "lj/cut/coul/long", "lj/cut/coul/msm",
        "eam", "eam/alloy", "eam/fs", "eam/cd", "eam/gw", "eam/gw/zbl",
        "sw", "tersoff", "tersoff/zbl", "reax", "reax/c", "rebo", "airebo",
        "airebo/morse", "opls", "gromos", "charmm", "amber", "dreiding",
        "uff", "tip3p", "tip4p", "spce", "spc", "oplsaa"
    ]
    
    return force_field.lower() in [ff.lower() for ff in valid_force_fields]


def validate_simulation_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate simulation configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["name"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate name
    if "name" in config and not validate_simulation_name(config["name"]):
        errors.append("Invalid simulation name")
    
    # Validate temperature
    if "temperature" in config and not validate_temperature(config["temperature"]):
        errors.append("Invalid temperature value")
    
    # Validate pressure
    if "pressure" in config and not validate_pressure(config["pressure"]):
        errors.append("Invalid pressure value")
    
    # Validate timestep
    if "timestep" in config and not validate_timestep(config["timestep"]):
        errors.append("Invalid timestep value")
    
    # Validate steps
    for step_field in ["equilibration_steps", "production_steps"]:
        if step_field in config and not validate_steps(config[step_field]):
            errors.append(f"Invalid {step_field} value")
    
    # Validate force field
    if "force_field" in config and not validate_force_field(config["force_field"]):
        errors.append("Invalid force field")
    
    # Validate structure file
    if "structure_file" in config and not validate_file_path(config["structure_file"]):
        errors.append("Invalid structure file path")
    
    return errors


def validate_simulation_id(simulation_id: str) -> bool:
    """
    Validate simulation ID.
    
    Args:
        simulation_id: Simulation ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not simulation_id or not isinstance(simulation_id, str):
        return False
    
    # Check length
    if len(simulation_id) < 1 or len(simulation_id) > 100:
        return False
    
    # Check for invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, simulation_id):
        return False
    
    return True


def validate_monitoring_parameters(duration_seconds: int, interval_seconds: int) -> List[str]:
    """
    Validate monitoring parameters.
    
    Args:
        duration_seconds: Monitoring duration in seconds
        interval_seconds: Monitoring interval in seconds
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate duration
    if not isinstance(duration_seconds, int) or duration_seconds < 1 or duration_seconds > 3600:
        errors.append("Invalid monitoring duration (must be 1-3600 seconds)")
    
    # Validate interval
    if not isinstance(interval_seconds, int) or interval_seconds < 1 or interval_seconds > 300:
        errors.append("Invalid monitoring interval (must be 1-300 seconds)")
    
    # Check that interval is less than duration
    if duration_seconds <= interval_seconds:
        errors.append("Monitoring interval must be less than duration")
    
    return errors


def validate_plot_type(plot_type: str) -> bool:
    """
    Validate plot type.
    
    Args:
        plot_type: Plot type to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_plot_types = ["thermo", "trajectory", "energy", "structure"]
    return plot_type.lower() in valid_plot_types


def validate_export_format(export_format: str) -> bool:
    """
    Validate export format.
    
    Args:
        export_format: Export format to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_formats = ["csv", "json", "npy", "txt"]
    return export_format.lower() in valid_formats


def validate_data_type(data_type: str) -> bool:
    """
    Validate data type.
    
    Args:
        data_type: Data type to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_data_types = ["thermo", "trajectory", "results", "structure"]
    return data_type.lower() in valid_data_types 