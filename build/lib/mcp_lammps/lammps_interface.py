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

# Import force field utilities
try:
    from .utils.forcefield_utils import forcefield_utils
    FORCEFIELD_AVAILABLE = True
except ImportError:
    logger.warning("ForceField utilities not available")
    FORCEFIELD_AVAILABLE = False


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
                    if 'Large-scale Atomic/Molecular Massively Parallel Simulator' in line:
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
        # Generate force field parameters based on force field type
        if force_field.lower() in ["lj/cut", "lj/cut/coul/cut", "lj/cut/coul/long"]:
            ff_type = "lj"
        else:
            ff_type = force_field
            
        ff_parameters = self._write_force_field_parameters(ff_type, structure_file=structure_file)
        
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

# Force field parameters
{ff_parameters}

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
            water_file: Path to water file
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
pair_style      lj/cut/coul/long 10.0
pair_modify     mix arithmetic
kspace_style    pppm 1.0e-4

# TIP3P pair coefficients (O-O, H-H interactions)
pair_coeff      1 1 0.1521 3.188    # O-O TIP3P parameters
pair_coeff      2 2 0.0000 0.000    # H-H (no LJ interaction)
pair_coeff      3 3 0.0000 0.000    # H-H (no LJ interaction)
pair_coeff      1 2 0.0000 0.000    # O-H cross terms
pair_coeff      1 3 0.0000 0.000    # O-H cross terms
pair_coeff      2 3 0.0000 0.000    # H-H cross terms

# TIP3P bond and angle parameters
bond_style      harmonic
bond_coeff      1 450.0 0.9572      # O-H bond
angle_style     harmonic
angle_coeff     1 55.0 104.52       # H-O-H angle

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
fix             2 all nvt temp {temperature} {temperature} 100.0
run             {production_steps}
unfix           2
"""
        return script
    
    def create_amber_script(
        self,
        structure_file: str,
        force_field: str = "gaff",
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 0.001,
        equilibration_steps: int = 1000,
        production_steps: int = 10000
    ) -> str:
        """
        Create a LAMMPS script for AMBER/GAFF force field simulations.
        
        Args:
            structure_file: Path to structure file
            force_field: Force field type (gaff, amber)
            temperature: Simulation temperature (K)
            pressure: Simulation pressure (atm)
            timestep: Simulation timestep (ps)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            
        Returns:
            LAMMPS script content
        """
        # Generate force field parameters
        ff_parameters = self._write_force_field_parameters(force_field, structure_file=structure_file)
        
        script = f"""
# LAMMPS script for AMBER/GAFF simulation generated by MCP LAMMPS Server

# Initialize simulation
units           real
boundary        p p p
atom_style      full

# Read structure
read_data       {structure_file}

# Define AMBER/GAFF force field
pair_style      lj/charmm/coul/long 9.0 10.0 10.0
pair_modify     mix arithmetic
kspace_style    pppm 1.0e-4

# Bond, angle, and dihedral styles for AMBER/GAFF
bond_style      harmonic
angle_style     harmonic
dihedral_style  fourier

# Special bonds for AMBER/GAFF (1-4 interactions scaled)
special_bonds   amber

# Force field parameters
{ff_parameters}

# Settings
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Minimization
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

# Equilibration (NVT)
timestep        {timestep}
fix             1 all nvt temp {temperature} {temperature} 100.0
thermo          100
thermo_style    custom step temp press pe ke etotal vol density
run             {equilibration_steps}
unfix           1

# Production (NPT)
fix             2 all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0
dump            1 all custom 1000 trajectory.lammpstrj id type x y z
dump_modify     1 sort id
run             {production_steps}
unfix           2

# Clean up
undump          1
"""
        return script
    
    def create_liquid_script(
        self,
        structure_file: str,
        force_field: str = "gaff",
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 0.001,
        equilibration_steps: int = 5000,
        production_steps: int = 50000,
        density_target: Optional[float] = None,
        atom_types: Optional[Dict[int, str]] = None,
        topology: Optional[Dict[str, Any]] = None,
        validate_consistency: bool = True
    ) -> str:
        """
        Create a LAMMPS script optimized for liquid-phase simulations.

        Args:
            structure_file: Path to structure file
            force_field: Force field type
            temperature: Simulation temperature (K)
            pressure: Simulation pressure (atm)
            timestep: Simulation timestep (ps)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            density_target: Target density for equilibration (g/cm³)
            atom_types: Dictionary mapping atom indices to GAFF atom types
            topology: Molecular topology information
            validate_consistency: Whether to validate type consistency between data file and script

        Returns:
            LAMMPS script content
        """
        # Generate force field parameters
        ff_parameters = self._write_force_field_parameters(
            force_field,
            structure_file=structure_file,
            atom_types=atom_types,
            topology=topology
        )
        
        script = f"""
# LAMMPS script for liquid simulation generated by MCP LAMMPS Server

# Initialize simulation
units           real
boundary        p p p
atom_style      full

# Read structure
read_data       {structure_file}

# Define force field
pair_style      hybrid lj/charmm/coul/long 9.0 10.0 10.0
pair_modify     mix arithmetic tail yes
kspace_style    pppm 1.0e-4

# Bond, angle, and dihedral styles
bond_style      hybrid harmonic
angle_style     hybrid harmonic
dihedral_style  hybrid fourier

# Special bonds for organic molecules
special_bonds   amber

# Force field parameters
{ff_parameters}

# Settings optimized for liquids
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# Compute properties for liquid analysis
compute         msd all msd
compute         rdf all rdf 100 1 1 cutoff 10.0
compute         temp_mol all temp
compute         press_mol all pressure temp_mol

# Minimization
minimize        1.0e-4 1.0e-6 1000 10000
reset_timestep  0

# Initial equilibration (NVT) - shorter timestep for stability
timestep        {timestep/2}
fix             1 all nvt temp {temperature/2} {temperature} 50.0
thermo          100
thermo_style    custom step temp press pe ke etotal vol density c_temp_mol c_press_mol
run             {equilibration_steps//2}
unfix           1

# Increase temperature gradually
fix             2 all nvt temp {temperature} {temperature} 100.0
timestep        {timestep}
run             {equilibration_steps//2}
unfix           2
"""
        
        # Add density equilibration if target is specified
        if density_target:
            script += f"""
# Density equilibration cycles
variable        target_density equal {density_target}
variable        current_density equal density
variable        density_diff equal abs(v_current_density-v_target_density)

label           density_loop
fix             3 all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0
run             2000
if              "${{density_diff}} < 0.05" then "jump SELF density_done"
unfix           3
jump            SELF density_loop

label           density_done
"""
        
        script += f"""
# Production run (NPT)
fix             4 all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0

# Output for analysis
dump            1 all custom 1000 trajectory.lammpstrj id type x y z vx vy vz
dump_modify     1 sort id
fix             5 all ave/time 100 10 1000 c_msd[4] file msd.dat
fix             6 all ave/correlate 100 10 1000 c_press_mol file pressure_autocorr.dat

# Enhanced thermodynamic output
thermo          500
thermo_style    custom step temp press pe ke etotal vol density c_msd[4] c_temp_mol c_press_mol

run             {production_steps}

# Clean up
unfix           4
unfix           5
unfix           6
undump          1
"""
        
        # Validate type consistency if requested
        if validate_consistency and FORCEFIELD_AVAILABLE:
            try:
                from pathlib import Path
                data_file_path = Path(structure_file)
                is_consistent, issues = forcefield_utils.validate_type_consistency(
                    data_file_path, script, None
                )
                if not is_consistent:
                    logger.warning(f"Type consistency issues detected: {issues}")
                    # Add validation results as comments to the script
                    script += f"\n# Type consistency validation: {'PASSED' if is_consistent else 'FAILED'}\n"
                    if issues:
                        script += "# Issues found:\n"
                        for issue in issues:
                            script += f"# - {issue}\n"
                else:
                    logger.info("Type consistency validation passed")
                    script += "\n# Type consistency validation: PASSED\n"
            except Exception as e:
                logger.warning(f"Type consistency validation failed: {e}")
                script += f"\n# Type consistency validation: ERROR - {e}\n"
        
        return script
    
    def setup_property_computes(self, lmp: Any) -> None:
        """
        Setup computes for transport property calculations.
        
        Args:
            lmp: LAMMPS instance
        """
        try:
            if not lmp:
                logger.error("Invalid LAMMPS instance")
                return
            
            # Mean squared displacement for diffusion coefficient
            self.run_command(lmp, "compute msd all msd")
            
            # Radial distribution function
            self.run_command(lmp, "compute rdf all rdf 100")
            
            # Stress tensor for viscosity calculation
            self.run_command(lmp, "compute stress all stress/atom NULL")
            
            # Temperature and pressure computes
            self.run_command(lmp, "compute temp_mol all temp")
            self.run_command(lmp, "compute press_mol all pressure temp_mol")
            
            # Velocity autocorrelation function
            self.run_command(lmp, "compute vacf all vacf")
            
            logger.info("Setup property computes for transport calculations")
            
        except Exception as e:
            logger.error(f"Failed to setup property computes: {e}")
    
    def run_density_equilibration(
        self,
        lmp: Any,
        target_density: float,
        temperature: float = 300.0,
        pressure: float = 1.0,
        max_cycles: int = 10,
        tolerance: float = 0.05
    ) -> bool:
        """
        Run iterative density equilibration to achieve target density.
        
        Args:
            lmp: LAMMPS instance
            target_density: Target density (g/cm³)
            temperature: Temperature (K)
            pressure: Pressure (atm)
            max_cycles: Maximum equilibration cycles
            tolerance: Density tolerance (g/cm³)
            
        Returns:
            True if target density achieved, False otherwise
        """
        try:
            if not lmp:
                logger.error("Invalid LAMMPS instance")
                return False
            
            logger.info(f"Starting density equilibration to {target_density} g/cm³")
            
            for cycle in range(max_cycles):
                # Run NPT equilibration
                self.run_command(lmp, f"fix eq all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0")
                self.run_command(lmp, "run 2000")
                self.run_command(lmp, "unfix eq")
                
                # Get current density
                current_density = self.get_system_property(lmp, "density")
                if current_density is None:
                    logger.error("Failed to get current density")
                    return False
                
                density_diff = abs(current_density - target_density)
                logger.info(f"Cycle {cycle+1}: density = {current_density:.3f} g/cm³, diff = {density_diff:.3f}")
                
                if density_diff < tolerance:
                    logger.info(f"Target density achieved in {cycle+1} cycles")
                    return True
                
                # Adjust pressure for next cycle if needed
                if current_density < target_density:
                    pressure *= 1.1  # Increase pressure to compress
                else:
                    pressure *= 0.9  # Decrease pressure to expand
            
            logger.warning(f"Failed to achieve target density after {max_cycles} cycles")
            return False
            
        except Exception as e:
            logger.error(f"Error in density equilibration: {e}")
            return False
    
    def get_system_property(self, lmp: Any, property_name: str) -> Optional[float]:
        """
        Get a specific system property from LAMMPS.
        
        Args:
            lmp: LAMMPS instance
            property_name: Name of the property (density, temp, press, etc.)
            
        Returns:
            Property value or None if failed
        """
        try:
            if not lmp:
                return None
            
            # Map property names to LAMMPS thermo keywords
            property_map = {
                "density": "density",
                "temperature": "temp",
                "pressure": "press",
                "volume": "vol",
                "potential_energy": "pe",
                "kinetic_energy": "ke",
                "total_energy": "etotal"
            }
            
            lammps_keyword = property_map.get(property_name.lower())
            if not lammps_keyword:
                logger.error(f"Unknown property: {property_name}")
                return None
            
            # Use LAMMPS extract to get the property
            if hasattr(lmp, 'extract_global'):
                value = lmp.extract_global(lammps_keyword, 1)  # 1 for double precision
                return float(value) if value is not None else None
            else:
                logger.error("LAMMPS extract_global not available")
                return None
                
        except Exception as e:
            logger.error(f"Error getting system property {property_name}: {e}")
            return None
    
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
    
    def _write_force_field_parameters(
        self,
        force_field_type: str,
        atom_types: Optional[Dict[int, str]] = None,
        topology: Optional[Dict[str, Any]] = None,
        structure_file: Optional[str] = None
    ) -> str:
        """
        Generate force field parameter section for LAMMPS scripts.
        
        Args:
            force_field_type: Type of force field (gaff, tip3p, lj, etc.)
            atom_types: Dictionary mapping atom indices to force field types
            topology: Topology information with bonds, angles, dihedrals
            structure_file: Path to structure file for parameter extraction
            
        Returns:
            String containing force field parameter commands
        """
        if not FORCEFIELD_AVAILABLE:
            logger.warning("ForceField utilities not available, using generic parameters")
            return "# Generic force field parameters\npair_coeff * *\n"
        
        parameters = []
        
        # Try to extract parameters from structure file if available
        if structure_file and force_field_type.lower() in ["gaff", "amber"] and not atom_types:
            try:
                extracted_params = self._extract_parameters_from_structure(structure_file, force_field_type)
                if extracted_params:
                    atom_types = extracted_params.get("atom_types")
                    topology = extracted_params.get("topology")
                    # Use unified topology mapping for consistency
                    if topology:
                        topology_types = forcefield_utils.create_unified_topology_mapping(topology)
                    logger.info(f"Extracted parameters from structure file: {structure_file}")
            except Exception as e:
                logger.warning(f"Could not extract parameters from structure file {structure_file}: {e}")
        
        if force_field_type.lower() == "tip3p":
            # TIP3P water parameters
            parameters.extend([
                "# TIP3P water force field parameters",
                "pair_coeff      1 1 0.1521 3.188    # O-O TIP3P",
                "pair_coeff      2 2 0.0000 0.000    # H-H (no LJ)",
                "pair_coeff      3 3 0.0000 0.000    # H-H (no LJ)",
                "pair_coeff      1 2 0.0000 0.000    # O-H cross",
                "pair_coeff      1 3 0.0000 0.000    # O-H cross",
                "pair_coeff      2 3 0.0000 0.000    # H-H cross",
                "",
                "bond_coeff      1 450.0 0.9572      # O-H bond",
                "angle_coeff     1 55.0 104.52       # H-O-H angle",
            ])
            
        elif force_field_type.lower() in ["gaff", "amber"]:
            # Comprehensive GAFF/AMBER parameters using new system
            if atom_types and topology:
                # Use unified topology type mappings for consistent bond/angle/dihedral types
                if 'topology_types' not in locals():
                    topology_types = forcefield_utils.create_unified_topology_mapping(topology)
                parameters.extend(self._generate_gaff_parameters(atom_types, topology, topology_types))
            else:
                # Import the new forcefield utilities
                from .utils.forcefield_utils import forcefield_utils
                ff_settings = forcefield_utils.get_force_field_settings()
                parameters.extend([
                    "# Comprehensive GAFF/AMBER force field parameters",
                    "# Note: Specific parameters should be generated based on molecule structure",
                    "# Force field settings available from comprehensive database",
                    f"# Available atom types: {len(forcefield_utils.atom_types)}",
                    f"# Available parameters: {forcefield_utils.get_statistics()}",
                ])
                
        elif force_field_type.lower() in ["lj/cut", "lj"]:
            # Generic Lennard-Jones parameters
            parameters.extend([
                "# Generic Lennard-Jones parameters",
                "pair_coeff      * * 1.0 3.5         # Generic LJ parameters",
            ])
            
        else:
            # Generic fallback
            parameters.extend([
                f"# {force_field_type} force field parameters",
                "# Note: Specific parameters not implemented for this force field",
                "pair_coeff      * *",
            ])
        
        return "\n".join(parameters)
    
    def _extract_parameters_from_structure(
        self,
        structure_file: str,
        force_field_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract force field parameters from a structure file.
        
        Args:
            structure_file: Path to structure file
            force_field_type: Type of force field
            
        Returns:
            Dictionary containing atom_types and topology, or None if extraction fails
        """
        try:
            from pathlib import Path
            import json
            
            structure_path = Path(structure_file)
            
            # Check if this is a LAMMPS data file with metadata
            if structure_path.suffix.lower() == '.data':
                # Look for corresponding metadata file
                metadata_file = structure_path.with_suffix('.json')
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        atom_types = metadata.get("atom_types", {})
                        topology = metadata.get("topology", {})
                        
                        if atom_types and topology:
                            logger.info(f"Extracted parameters from metadata file: {metadata_file}")
                            return {
                                "atom_types": atom_types,
                                "topology": topology,
                                "charges": metadata.get("charges", {}),
                                "properties": metadata.get("properties", {})
                            }
                    except Exception as e:
                        logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                
                # Fallback to parsing LAMMPS data file directly
                return self._extract_from_lammps_data_file(structure_path)
            
            # For other file types, try to use DataHandler to assign parameters
            logger.info(f"Parameter extraction from {structure_file} not yet implemented for file type {structure_path.suffix}")
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting parameters from structure file: {e}")
            return None
    
    def _extract_from_lammps_data_file(self, data_file: Path) -> Optional[Dict[str, Any]]:
        """
        Extract atom types and topology from a LAMMPS data file.

        Args:
            data_file: Path to LAMMPS data file

        Returns:
            Dictionary with atom_types and topology information
        """
        try:
            if not data_file.exists():
                return None

            with open(data_file, 'r') as f:
                content = f.read()

            # Parse LAMMPS data file to extract atom types and topology
            lines = content.split('\n')

            # Look for atom type information in comments or masses section
            atom_types = {}
            topology = {"bonds": [], "angles": [], "dihedrals": []}

            # This is a simplified parser - in practice, you'd need more robust parsing
            # For now, we'll return None to indicate we couldn't extract parameters
            logger.info("LAMMPS data file parameter extraction not fully implemented")
            return None

        except Exception as e:
            logger.warning(f"Error parsing LAMMPS data file: {e}")
            return None

    def _get_unique_topology_types(self, topology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get unique bond, angle, and dihedral types from topology.

        Args:
            topology: Topology information with bonds, angles, dihedrals

        Returns:
            Dictionary containing unique types and their mappings
        """
        try:
            # Collect all unique bond types
            all_bond_types = set()
            all_angle_types = set()
            all_dihedral_types = set()

            # Collect bond types
            for bond_info in topology.get("bonds", []):
                bond_type = tuple(sorted(bond_info.get("types", [])))
                all_bond_types.add(bond_type)

            # Collect angle types
            for angle_info in topology.get("angles", []):
                angle_type = tuple(angle_info.get("types", []))
                all_angle_types.add(angle_type)

            # Collect dihedral types
            for dihedral_info in topology.get("dihedrals", []):
                dihedral_type = tuple(dihedral_info.get("types", []))
                all_dihedral_types.add(dihedral_type)

            # Create type mappings (sorted for deterministic ordering)
            unique_bond_types = sorted(list(all_bond_types))
            unique_angle_types = sorted(list(all_angle_types))
            unique_dihedral_types = sorted(list(all_dihedral_types))

            # Create mappings with STRING KEYS (not tuples) for JSON compatibility
            bond_type_mapping = {"-".join(bond_type): i+1 for i, bond_type in enumerate(unique_bond_types)}
            angle_type_mapping = {"-".join(angle_type): i+1 for i, angle_type in enumerate(unique_angle_types)}
            dihedral_type_mapping = {"-".join(dihedral_type): i+1 for i, dihedral_type in enumerate(unique_dihedral_types)}

            return {
                "unique_bond_types": [list(bt) for bt in unique_bond_types],  # Convert tuples to lists for JSON
                "unique_angle_types": [list(at) for at in unique_angle_types],
                "unique_dihedral_types": [list(dt) for dt in unique_dihedral_types],
                "bond_type_mapping": bond_type_mapping,
                "angle_type_mapping": angle_type_mapping,
                "dihedral_type_mapping": dihedral_type_mapping
            }

        except Exception as e:
            logger.error(f"Error getting unique topology types: {e}")
            return {
                "unique_bond_types": [],
                "unique_angle_types": [],
                "unique_dihedral_types": [],
                "bond_type_mapping": {},
                "angle_type_mapping": {},
                "dihedral_type_mapping": {}
            }
    
    def _generate_gaff_parameters(
        self,
        atom_types: Dict[int, str],
        topology: Dict[str, Any],
        topology_types: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate comprehensive GAFF force field parameters from atom types and topology.
        
        Args:
            atom_types: Dictionary mapping atom indices to GAFF atom types
            topology: Topology information with bonds, angles, dihedrals
            topology_types: Optional pre-computed topology type mappings for consistency
            
        Returns:
            List of parameter strings
        """
        parameters = ["# Comprehensive GAFF2 force field parameters"]
        
        # Add force field settings first
        ff_settings = forcefield_utils.get_force_field_settings()
        if ff_settings:
            parameters.append("\n# Force field settings")
            for setting, value in ff_settings.items():
                parameters.append(f"# {setting} {value}")
        
        # Generate pair coefficients from comprehensive VdW parameters
        # Sort unique types for deterministic ordering
        unique_types = sorted(set(atom_types.values()))
        type_mapping = {atype: i+1 for i, atype in enumerate(unique_types)}
        
        # Use unified topology mapping if provided, otherwise create it
        if topology_types is None:
            topology_types = forcefield_utils.create_unified_topology_mapping(topology)
            logger.info("Created unified topology mapping for script generation")
        
        parameters.append("\n# Pair coefficients (Lennard-Jones parameters)")
        for atype in unique_types:
            type_id = type_mapping[atype]
            pair_params = forcefield_utils.get_pair_parameters(atype)
            
            if pair_params:
                epsilon = pair_params["epsilon"]
                sigma = pair_params["sigma"]
                style = pair_params.get("style", "lj/charmm/coul/long")
                parameters.append(f"pair_coeff      {type_id} {type_id} {style} {epsilon:.4f} {sigma:.4f}    # {atype}")
            else:
                # Fallback parameters
                logger.warning(f"No pair parameters found for atom type: {atype}")
                parameters.append(f"pair_coeff      {type_id} {type_id} lj/charmm/coul/long 0.1000 3.5000    # {atype} (fallback)")
        
        # Generate bond coefficients using comprehensive database
        if "bonds" in topology and topology["bonds"]:
            parameters.append("\n# Bond coefficients")
            bond_type_map = {}
            bond_type_mapping = topology_types.get('bond_type_mapping', {})
            
            for bond in topology["bonds"]:
                bond_types = bond.get("types", [])
                if len(bond_types) >= 2:
                    # Use consistent bond type key generation (sorted for bonds, then joined as string)
                    bond_type_key = "-".join(sorted(bond_types[:2]))
                    bond_id = bond_type_mapping.get(bond_type_key, 1)
                    
                    if bond_type_key not in bond_type_map:
                        bond_params = forcefield_utils.get_bond_parameters(bond_types[0], bond_types[1])
                        
                        if bond_params:
                            k = bond_params["k"]
                            r0 = bond_params["r0"]
                            style = bond_params.get("style", "harmonic")
                            comment = bond_params.get("comment", "")
                            parameters.append(f"bond_coeff      {bond_id} {style} {k:.1f} {r0:.4f}    # {bond_type_key} {comment}")
                        else:
                            # Fallback parameters
                            logger.warning(f"No bond parameters found for: {bond_type_key}")
                            parameters.append(f"bond_coeff      {bond_id} harmonic 300.0 1.500    # {bond_type_key} (fallback)")
                        
                        bond_type_map[bond_type_key] = bond_id
                        
        
        # Generate angle coefficients using comprehensive database
        if "angles" in topology and topology["angles"]:
            parameters.append("\n# Angle coefficients")
            angle_type_map = {}
            angle_type_mapping = topology_types.get('angle_type_mapping', {})
            
            for angle in topology["angles"]:
                angle_types = angle.get("types", [])
                if len(angle_types) >= 3:
                    # Use consistent angle type key generation (maintain order for angles, then join as string)
                    angle_type_key = "-".join(angle_types[:3])
                    angle_id = angle_type_mapping.get(angle_type_key, 1)
                    
                    if angle_type_key not in angle_type_map:
                        angle_params = forcefield_utils.get_angle_parameters(angle_types[0], angle_types[1], angle_types[2])
                        
                        if angle_params:
                            k = angle_params["k"]
                            theta0 = angle_params["theta0"]
                            style = angle_params.get("style", "harmonic")
                            comment = angle_params.get("comment", "")
                            parameters.append(f"angle_coeff     {angle_id} {style} {k:.2f} {theta0:.2f}    # {angle_type_key} {comment}")
                        else:
                            # Fallback parameters
                            logger.warning(f"No angle parameters found for: {angle_type_key}")
                            parameters.append(f"angle_coeff     {angle_id} harmonic 50.0 109.5    # {angle_type_key} (fallback)")
                        
                        angle_type_map[angle_type_key] = angle_id
                        
        
        # Generate dihedral coefficients using comprehensive database
        if "dihedrals" in topology and topology["dihedrals"]:
            parameters.append("\n# Dihedral coefficients")
            dihedral_type_map = {}
            dihedral_type_mapping = topology_types.get('dihedral_type_mapping', {})
            
            for dihedral in topology["dihedrals"]:
                dihedral_types = dihedral.get("types", [])
                if len(dihedral_types) >= 4:
                    # Use consistent dihedral type key generation (maintain order for dihedrals, then join as string)
                    dihedral_type_key = "-".join(dihedral_types[:4])
                    dihedral_id = dihedral_type_mapping.get(dihedral_type_key, 1)

                    if dihedral_type_key not in dihedral_type_map:
                        # get_dihedral_parameters expects a tuple, so convert string key back to tuple
                        dihedral_type_tuple = tuple(dihedral_types[:4])
                        dihedral_params = forcefield_utils.get_dihedral_parameters(dihedral_type_tuple)
                        
                        if dihedral_params and len(dihedral_params) > 0:
                            # Use the first dihedral parameter set (GAFF typically has multiple terms)
                            param = dihedral_params[0]
                            k = param.get("k", 0.0)
                            n = param.get("n", 1)
                            phase = param.get("phase", 0.0)
                            n_terms = param.get("n_terms", 1)
                            style = param.get("style", "fourier")
                            comment = param.get("comment", "")
                            parameters.append(f"dihedral_coeff  {dihedral_id} {style} {n_terms} {k:.3f} {n} {phase:.1f}    # {dihedral_type_key} {comment}")
                        else:
                            # Fallback parameters
                            logger.warning(f"No dihedral parameters found for: {dihedral_type_key}")
                            parameters.append(f"dihedral_coeff  {dihedral_id} fourier 1 0.000 1 0.0    # {dihedral_type_key} (fallback)")
                        
                        dihedral_type_map[dihedral_type_key] = dihedral_id
        
        # Generate improper dihedral coefficients using comprehensive database
        if "impropers" in topology and topology["impropers"]:
            parameters.append("\n# Improper dihedral coefficients")
            improper_type_map = {}
            improper_id = 1
            
            for improper in topology["impropers"]:
                improper_types = improper.get("types", [])
                if len(improper_types) == 4:
                    improper_key = tuple(improper_types)
                    if improper_key not in improper_type_map:
                        # For impropers, we use the improper coefficient database
                        # This is more complex as impropers are defined differently in GAFF
                        style = "cvff"  # Default GAFF improper style
                        parameters.append(f"improper_coeff  {improper_id} {style} 1.1 -1 2    # {'-'.join(improper_key)} (default)")
                        
                        improper_type_map[improper_key] = improper_id
                        improper_id += 1
        
        # Add parameter statistics
        stats = forcefield_utils.get_statistics()
        parameters.append(f"\n# Parameter database statistics: {stats}")
        
        return parameters 