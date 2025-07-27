"""
Helpers - Utility helper functions for the MCP LAMMPS server.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    def default_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    return json.dumps(obj, default=default_serializer, indent=2)


def check_command_available(command: str) -> bool:
    """
    Check if a command is available in the system.
    
    Args:
        command: Command to check
        
    Returns:
        True if command is available, False otherwise
    """
    try:
        result = subprocess.run(
            [command, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        System information dictionary
    """
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "python_executable": sys.executable
    }
    
    # Get CPU info
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo", "r") as f:
                cpu_info = f.read()
                for line in cpu_info.split('\n'):
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(':')[1].strip()
                        break
        elif sys.platform == "darwin":
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                info["cpu_model"] = result.stdout.strip()
    except Exception:
        pass
    
    # Get memory info
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo", "r") as f:
                mem_info = f.read()
                for line in mem_info.split('\n'):
                    if line.startswith("MemTotal"):
                        mem_kb = int(line.split()[1])
                        info["total_memory_gb"] = mem_kb / (1024 * 1024)
                        break
        elif sys.platform == "darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                info["total_memory_gb"] = mem_bytes / (1024 * 1024 * 1024)
    except Exception:
        pass
    
    return info


def create_backup(file_path: Union[str, Path]) -> Optional[Path]:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to file to backup
        
    Returns:
        Path to backup file or None if failed
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        backup_path = path.with_suffix(f"{path.suffix}.backup")
        counter = 1
        while backup_path.exists():
            backup_path = path.with_suffix(f"{path.suffix}.backup.{counter}")
            counter += 1
        
        import shutil
        shutil.copy2(path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        return None


def cleanup_old_files(directory: Union[str, Path], pattern: str = "*", 
                     max_age_days: int = 30) -> int:
    """
    Clean up old files in a directory.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        max_age_days: Maximum age in days
        
    Returns:
        Number of files cleaned up
    """
    try:
        import time
        from datetime import datetime, timedelta
        
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old files in {directory}")
        
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Failed to clean up old files in {directory}: {e}")
        return 0


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_relative_path(base_path: Union[str, Path], target_path: Union[str, Path]) -> str:
    """
    Get relative path from base path to target path.
    
    Args:
        base_path: Base path
        target_path: Target path
        
    Returns:
        Relative path string
    """
    try:
        base = Path(base_path).resolve()
        target = Path(target_path).resolve()
        return str(target.relative_to(base))
    except ValueError:
        return str(target_path)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized


def parse_lammps_version(version_string: str) -> Optional[Dict[str, Any]]:
    """
    Parse LAMMPS version string.
    
    Args:
        version_string: LAMMPS version string
        
    Returns:
        Version information dictionary or None
    """
    import re
    
    try:
        # Common LAMMPS version patterns
        patterns = [
            r'LAMMPS\s+\((\d+)\s+(\w+)\s+(\d+)\)',
            r'LAMMPS\s+version\s+(\d+)\s+(\w+)\s+(\d+)',
            r'LAMMPS\s+(\d+)\s+(\w+)\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, version_string, re.IGNORECASE)
            if match:
                return {
                    "year": int(match.group(1)),
                    "month": match.group(2),
                    "day": int(match.group(3)),
                    "full_string": version_string.strip()
                }
        
        return None
        
    except Exception:
        return None


def estimate_simulation_time(num_atoms: int, num_steps: int, 
                           timestep: float = 0.001) -> float:
    """
    Estimate simulation time based on system size and parameters.
    
    Args:
        num_atoms: Number of atoms
        num_steps: Number of simulation steps
        timestep: Simulation timestep in ps
        
    Returns:
        Estimated time in seconds
    """
    # Rough estimation based on typical performance
    # This is a very approximate calculation
    atoms_per_second = 1000  # Rough estimate of atoms processed per second
    
    simulation_time_ps = num_steps * timestep
    estimated_seconds = (num_atoms * simulation_time_ps) / atoms_per_second
    
    return estimated_seconds


def format_simulation_summary(simulation_data: Dict[str, Any]) -> str:
    """
    Format simulation data into a readable summary.
    
    Args:
        simulation_data: Simulation data dictionary
        
    Returns:
        Formatted summary string
    """
    lines = []
    
    # Basic info
    lines.append(f"Simulation: {simulation_data.get('name', 'Unknown')}")
    lines.append(f"ID: {simulation_data.get('simulation_id', 'Unknown')}")
    lines.append(f"State: {simulation_data.get('state', 'Unknown')}")
    
    # Progress
    progress = simulation_data.get('progress', {})
    if progress:
        current = progress.get('current_step', 0)
        total = progress.get('total_steps', 0)
        percentage = progress.get('percentage', 0.0)
        lines.append(f"Progress: {current}/{total} steps ({percentage:.1f}%)")
    
    # Timing
    timing = simulation_data.get('timing', {})
    if timing:
        created = timing.get('created')
        started = timing.get('started')
        completed = timing.get('completed')
        
        if created:
            lines.append(f"Created: {created}")
        if started:
            lines.append(f"Started: {started}")
        if completed:
            lines.append(f"Completed: {completed}")
    
    # Results
    results_count = simulation_data.get('results_count', 0)
    if results_count > 0:
        lines.append(f"Results: {results_count} data points")
    
    # Error
    error = simulation_data.get('error')
    if error:
        lines.append(f"Error: {error}")
    
    return "\n".join(lines) 