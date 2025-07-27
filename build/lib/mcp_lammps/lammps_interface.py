"""
LAMMPS Interface - Python wrapper for LAMMPS molecular dynamics engine.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class LAMMPSInterface:
    """
    Interface for interacting with LAMMPS molecular dynamics engine.
    
    This class provides a high-level interface for:
    - Checking LAMMPS availability and version
    - Creating and managing LAMMPS instances
    - Running simulations
    - Extracting data from simulations
    """
    
    def __init__(self):
        """Initialize the LAMMPS interface."""
        self._lammps_available = False
        self._lammps_version = None
        self._python_lammps_available = False
        
        # Check availability on initialization
        self._check_availability()
    
    def _check_availability(self) -> None:
        """Check if LAMMPS is available and get version information."""
        try:
            # Check if LAMMPS executable is available
            result = subprocess.run(
                ["lmp", "-help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self._lammps_available = True
                # Extract version from help output
                for line in result.stdout.split('\n'):
                    if 'LAMMPS' in line and 'version' in line.lower():
                        self._lammps_version = line.strip()
                        break
                logger.info(f"LAMMPS executable found: {self._lammps_version}")
            else:
                logger.warning("LAMMPS executable not found or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            logger.warning("LAMMPS executable not found")
        
        # Check if Python LAMMPS interface is available
        try:
            import lammps
            self._python_lammps_available = True
            logger.info("Python LAMMPS interface available")
        except ImportError:
            logger.warning("Python LAMMPS interface not available")
    
    async def check_availability(self) -> Dict[str, Any]:
        """
        Check LAMMPS availability and return status information.
        
        Returns:
            Dictionary with availability status and version information
        """
        return {
            "lammps_executable": self._lammps_available,
            "python_interface": self._python_lammps_available,
            "version": self._lammps_version,
            "fully_available": self._lammps_available and self._python_lammps_available
        }
    
    def create_instance(self, script_file: Optional[Path] = None) -> Optional[Any]:
        """
        Create a new LAMMPS instance.
        
        Args:
            script_file: Optional LAMMPS script file to load
            
        Returns:
            LAMMPS instance or None if not available
        """
        if not self._python_lammps_available:
            logger.error("Python LAMMPS interface not available")
            return None
        
        try:
            import lammps
            lmp = lammps.lammps()
            
            if script_file and script_file.exists():
                lmp.file(str(script_file))
                logger.info(f"Loaded LAMMPS script: {script_file}")
            
            return lmp
        except Exception as e:
            logger.error(f"Failed to create LAMMPS instance: {e}")
            return None
    
    def run_command(self, lmp: Any, command: str) -> bool:
        """
        Run a LAMMPS command.
        
        Args:
            lmp: LAMMPS instance
            command: LAMMPS command to run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            lmp.command(command)
            return True
        except Exception as e:
            logger.error(f"Failed to run LAMMPS command '{command}': {e}")
            return False
    
    def run_script(self, lmp: Any, script_content: str) -> bool:
        """
        Run LAMMPS script content.
        
        Args:
            lmp: LAMMPS instance
            script_content: LAMMPS script content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for line in script_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    lmp.command(line)
            return True
        except Exception as e:
            logger.error(f"Failed to run LAMMPS script: {e}")
            return False
    
    def get_system_info(self, lmp: Any) -> Dict[str, Any]:
        """
        Get system information from LAMMPS instance.
        
        Args:
            lmp: LAMMPS instance
            
        Returns:
            Dictionary with system information
        """
        try:
            info = {}
            
            # Get number of atoms
            info['num_atoms'] = lmp.get_natoms()
            
            # Get simulation box
            box = lmp.extract_box()
            if box:
                info['box'] = {
                    'xlo': box[0][0], 'xhi': box[0][1],
                    'ylo': box[1][0], 'yhi': box[1][1],
                    'zlo': box[2][0], 'zhi': box[2][1]
                }
            
            # Get atom types
            info['num_types'] = lmp.extract_global('ntypes')
            
            return info
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}
    
    def get_thermo_data(self, lmp: Any) -> Dict[str, float]:
        """
        Get thermodynamic data from LAMMPS instance.
        
        Args:
            lmp: LAMMPS instance
            
        Returns:
            Dictionary with thermodynamic data
        """
        try:
            thermo = {}
            
            # Get current step
            thermo['step'] = lmp.extract_global('ntimestep')
            
            # Get temperature
            thermo['temperature'] = lmp.extract_global('temp')
            
            # Get pressure
            thermo['pressure'] = lmp.extract_global('press')
            
            # Get potential energy
            thermo['pe'] = lmp.extract_global('pe')
            
            # Get kinetic energy
            thermo['ke'] = lmp.extract_global('ke')
            
            # Get total energy
            thermo['etotal'] = thermo['pe'] + thermo['ke']
            
            return thermo
        except Exception as e:
            logger.error(f"Failed to get thermo data: {e}")
            return {}
    
    def get_atom_data(self, lmp: Any) -> Dict[str, np.ndarray]:
        """
        Get atom data from LAMMPS instance.
        
        Args:
            lmp: LAMMPS instance
            
        Returns:
            Dictionary with atom data arrays
        """
        try:
            data = {}
            
            # Get number of atoms
            num_atoms = lmp.get_natoms()
            
            # Get positions
            positions = lmp.gather_atoms('x', 1, 3)
            if positions:
                data['positions'] = np.array(positions).reshape(-1, 3)
            
            # Get velocities
            velocities = lmp.gather_atoms('v', 1, 3)
            if velocities:
                data['velocities'] = np.array(velocities).reshape(-1, 3)
            
            # Get forces
            forces = lmp.gather_atoms('f', 1, 3)
            if forces:
                data['forces'] = np.array(forces).reshape(-1, 3)
            
            # Get atom types
            types = lmp.gather_atoms('type', 0, 1)
            if types:
                data['types'] = np.array(types)
            
            return data
        except Exception as e:
            logger.error(f"Failed to get atom data: {e}")
            return {}
    
    def create_basic_script(
        self,
        structure_file: str,
        force_field: str = "lj/cut",
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 0.001,
        equilibration_steps: int = 1000,
        production_steps: int = 10000
    ) -> str:
        """
        Create a basic LAMMPS script for molecular dynamics simulation.
        
        Args:
            structure_file: Path to structure file
            force_field: Force field to use
            temperature: Simulation temperature (K)
            pressure: Simulation pressure (atm)
            timestep: Simulation timestep (ps)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            
        Returns:
            LAMMPS script content
        """
        script = f"""
# LAMMPS script generated by MCP LAMMPS Server

# Initialize simulation
units           real
boundary        p p p
atom_style      full

# Read structure
read_data       {structure_file}

# Define force field
pair_style      {force_field}
pair_coeff      * *

# Settings
neighbor        2.0 bin
neigh_modify   every 1 delay 0 check yes

# Minimization
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

# Equilibration
timestep        {timestep}
fix             1 all nvt temp {temperature} {temperature} 100.0
run             {equilibration_steps}
unfix           1

# Production
fix             2 all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0
run             {production_steps}
unfix           2
"""
        return script
    
    def create_water_script(
        self,
        water_file: str,
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 0.001,
        equilibration_steps: int = 1000,
        production_steps: int = 10000
    ) -> str:
        """
        Create a LAMMPS script for water simulation using TIP3P force field.
        
        Args:
            num_molecules: Number of water molecules
            box_size: Simulation box size (Angstroms)
            temperature: Simulation temperature (K)
            pressure: Simulation pressure (atm)
            timestep: Simulation timestep (ps)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            
        Returns:
            LAMMPS script content
        """
        script = f"""
# LAMMPS script for water simulation (TIP3P)

# Initialize simulation
units           real
boundary        p p p
atom_style      full

# Create water box
read_data       {water_file}

# Define TIP3P force field
pair_style      lj/cut/coul/cut 10.0
pair_coeff      1 1 0.0 0.0
pair_coeff      2 2 0.0 0.0
bond_style      harmonic
bond_coeff      1 450.0 0.9572
angle_style     harmonic
angle_coeff     1 55.0 104.52

# Settings
neighbor        2.0 bin
neigh_modify   every 1 delay 0 check yes
special_bonds   lj/coul 0.0 0.0 0.5

# Minimization
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

# Equilibration
timestep        {timestep}
fix             1 all nvt temp {temperature} {temperature} 100.0
run             {equilibration_steps}
unfix           1

# Production
fix             1 all nvt temp {temperature} {temperature} 100.0
run             {production_steps}
unfix           2
"""
        return script
    
    def is_available(self) -> bool:
        """
        Check if LAMMPS is fully available.
        
        Returns:
            True if both executable and Python interface are available
        """
        return self._lammps_available and self._python_lammps_available
    
    def get_version(self) -> Optional[str]:
        """
        Get LAMMPS version.
        
        Returns:
            LAMMPS version string or None if not available
        """
        return self._lammps_version 