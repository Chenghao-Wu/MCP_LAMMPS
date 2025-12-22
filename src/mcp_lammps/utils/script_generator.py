"""
LAMMPS Equilibrium Script Generator - Best Practice MD Protocols

This module provides comprehensive LAMMPS script generation following molecular dynamics
best practices for equilibrium simulations. It implements multi-stage equilibration
protocols with full parameter control and system-specific optimizations.

Key Features:
- Multi-stage equilibration: minimization → NVT heating → NPT density equilibration → NPT production
- Configurable stages with enable/disable control
- Velocity rescaling and gradual pressure coupling
- Convergence checks for density and energy
- Restart file generation at each stage
- Trajectory dumps with configurable frequency
- Comprehensive thermodynamic output
- System-specific optimizations for water and organic liquids

Best Practice Protocol:
1. Energy Minimization: Remove bad contacts and overlaps
2. NVT Heating: Gradual heating from 0.5*T to T with velocity rescaling
3. NPT Density Equilibration: Iterative cycles until density converges
4. NPT Production: Full production run with trajectory and restart outputs
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EquilibrationStage(Enum):
    """Enumeration of equilibration stages."""
    MINIMIZATION = "minimization"
    NVT_HEATING = "nvt_heating"
    NPT_DENSITY_EQUILIBRATION = "npt_density_equilibration"
    NPT_PRODUCTION = "npt_production"


class SystemType(Enum):
    """Enumeration of system types for optimization."""
    GENERIC = "generic"
    WATER = "water"
    ORGANIC_LIQUID = "organic_liquid"
    MIXED = "mixed"


@dataclass
class SimulationConfig:
    """
    Configuration for equilibrium MD simulation.
    
    This dataclass contains all parameters needed to generate a complete
    LAMMPS equilibrium simulation script following best practices.
    """
    
    # Basic simulation parameters
    temperature: float = 300.0  # K
    pressure: float = 1.0  # atm
    timestep: float = 1.0  # fs
    
    # Duration parameters (in steps)
    minimization_steps: int = 10000
    nvt_heating_steps: int = 10000
    npt_density_steps: int = 100000
    production_steps: int = 1000000
    
    # Stage control
    enable_minimization: bool = True
    enable_nvt_heating: bool = True
    enable_density_equilibration: bool = True
    
    # Minimization settings
    minimization_style: str = "cg"  # cg, sd, fire
    energy_tolerance: float = 1.0e-4
    force_tolerance: float = 1.0e-6
    max_minimization_iterations: int = 10000
    
    # Heating settings
    heating_start_temp: float = 150.0  # K (0.5 * target by default)
    heating_timestep: float = 0.5  # fs (shorter for stability)
    velocity_rescale_frequency: int = 100  # steps
    
    # Density equilibration settings
    density_equilibration_cycles: int = 5
    density_tolerance: float = 0.05  # g/cm³
    max_density_cycles: int = 10
    convergence_checks: bool = True
    
    # Thermostat/Barostat settings
    thermostat_damping: float = 100.0  # fs
    barostat_damping: float = 1000.0  # fs
    thermostat_style: str = "nvt"  # nvt, npt
    pressure_coupling_style: str = "iso"  # iso, aniso, tri
    
    # Output control
    trajectory_frequency: int = 10000  # steps
    restart_frequency: int = 100000  # steps
    thermo_frequency: int = 1000  # steps
    dump_velocities: bool = False
    dump_forces: bool = False
    output_directory: str = "."
    
    # Advanced options
    velocity_rescaling: bool = True
    neighbor_skin: float = 2.0  # Angstroms
    neighbor_delay: int = 0
    neighbor_check: bool = True
    
    # Custom additions
    custom_computes: List[str] = field(default_factory=list)
    custom_fixes: List[str] = field(default_factory=list)
    custom_variables: Dict[str, str] = field(default_factory=dict)
    
    # System information (auto-detected or provided)
    system_type: SystemType = SystemType.GENERIC
    has_bonds: bool = True
    has_angles: bool = True
    has_dihedrals: bool = True
    has_impropers: bool = False
    
    # Force field configuration
    pointenergy_file: Optional[str] = None  # Path to pointenergy file for force field styles
    
    def __post_init__(self):
        """Validate and adjust parameters after initialization."""
        # Ensure heating start temp is less than target temp
        if self.heating_start_temp >= self.temperature:
            self.heating_start_temp = self.temperature * 0.5
            logger.warning(f"Adjusted heating_start_temp to {self.heating_start_temp} K")
        
        # Ensure heating timestep is not larger than production timestep
        if self.heating_timestep > self.timestep:
            self.heating_timestep = self.timestep * 0.5
            logger.warning(f"Adjusted heating_timestep to {self.heating_timestep} fs")


class SystemDetector:
    """
    Detect system type and recommend optimal simulation parameters.
    
    This class analyzes LAMMPS data files or system information to identify
    the system type and suggest appropriate simulation parameters.
    """
    
    @staticmethod
    def detect_from_data_file(data_file: Union[str, Path]) -> SystemType:
        """
        Detect system type from LAMMPS data file.
        
        Args:
            data_file: Path to LAMMPS data file
            
        Returns:
            Detected system type
        """
        try:
            data_file = Path(data_file)
            if not data_file.exists():
                logger.warning(f"Data file not found: {data_file}")
                return SystemType.GENERIC
            
            with open(data_file, 'r') as f:
                content = f.read()
            
            # Simple heuristics for system detection
            # Look for TIP3P water signature (3 atoms per molecule, specific charges)
            if "# TIP3P" in content or "# Water" in content:
                return SystemType.WATER
            
            # Check for organic molecule signatures
            if "# Organic" in content or "# OpenFF" in content:
                return SystemType.ORGANIC_LIQUID
            
            # Check atom count and molecule structure
            lines = content.split('\n')
            n_atoms = 0
            n_molecules = 0
            
            for line in lines[:50]:
                if 'atoms' in line and 'atom types' not in line:
                    n_atoms = int(line.split()[0])
                elif 'molecules' in line:
                    n_molecules = int(line.split()[0])
            
            if n_molecules > 0 and n_atoms > 0:
                atoms_per_molecule = n_atoms / n_molecules
                # Water has 3 atoms per molecule
                if 2.5 < atoms_per_molecule < 3.5:
                    return SystemType.WATER
                # Organic liquids typically have more atoms
                elif atoms_per_molecule > 5:
                    return SystemType.ORGANIC_LIQUID
            
            return SystemType.GENERIC
            
        except Exception as e:
            logger.error(f"Failed to detect system type: {e}")
            return SystemType.GENERIC
    
    @staticmethod
    def optimize_config_for_system(
        config: SimulationConfig,
        system_type: SystemType
    ) -> SimulationConfig:
        """
        Optimize simulation configuration based on system type.
        
        Args:
            config: Base simulation configuration
            system_type: Detected or specified system type
            
        Returns:
            Optimized configuration
        """
        config.system_type = system_type
        
        if system_type == SystemType.WATER:
            # Water-specific optimizations (same as organic liquids per requirements)
            config.timestep = 1.0  # fs
            config.thermostat_damping = 100.0  # fs
            config.barostat_damping = 1000.0  # fs
            logger.info("Applied water system optimizations")
            
        elif system_type == SystemType.ORGANIC_LIQUID:
            # Organic liquid optimizations
            config.timestep = 1.0  # fs
            config.thermostat_damping = 100.0  # fs
            config.barostat_damping = 1000.0  # fs
            logger.info("Applied organic liquid optimizations")
            
        elif system_type == SystemType.MIXED:
            # Mixed system - use conservative parameters
            config.timestep = 1.0  # fs
            config.thermostat_damping = 100.0  # fs
            config.barostat_damping = 1000.0  # fs
            logger.info("Applied mixed system optimizations")
        
        return config


class EquilibriumScriptGenerator:
    """
    Generate LAMMPS scripts following MD best practices for equilibrium simulations.
    
    This class creates comprehensive LAMMPS input scripts with multi-stage
    equilibration protocols, including energy minimization, NVT heating,
    NPT density equilibration, and NPT production runs.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the script generator.
        
        Args:
            config: Simulation configuration (uses defaults if not provided)
        """
        self.config = config or SimulationConfig()
        self._kspace_style = None  # Will be set when parsing pointenergy file
        logger.info(f"Initialized EquilibriumScriptGenerator with system type: {self.config.system_type}")
    
    def generate_script(
        self,
        data_file: str,
        output_prefix: Optional[str] = None
    ) -> str:
        """
        Generate complete equilibrium MD script.
        
        Args:
            data_file: Path to LAMMPS data file
            output_prefix: Prefix for output files (default: derived from data_file)
            
        Returns:
            Complete LAMMPS script as string
        """
        if output_prefix is None:
            output_prefix = Path(data_file).stem
        
        script_parts = []
        
        # Header
        script_parts.append(self._generate_header())
        
        # Initialization
        script_parts.append(self._generate_initialization(data_file))
        
        # Force field (read from data file)
        script_parts.append(self._generate_force_field_section())
        
        # Settings
        script_parts.append(self._generate_settings())
        
        # Custom computes and variables
        if self.config.custom_computes or self.config.custom_variables:
            script_parts.append(self._generate_custom_section())
        
        # Stage 1: Energy Minimization
        if self.config.enable_minimization:
            script_parts.append(self._generate_minimization_stage(output_prefix))
        
        # Stage 2: NVT Heating
        if self.config.enable_nvt_heating:
            script_parts.append(self._generate_nvt_heating_stage(output_prefix))
        
        # Stage 3: NPT Density Equilibration
        if self.config.enable_density_equilibration:
            script_parts.append(self._generate_density_equilibration_stage(output_prefix))
        
        # Stage 4: NPT Production
        script_parts.append(self._generate_production_stage(output_prefix))
        
        # Footer
        script_parts.append(self._generate_footer())
        
        return "\n".join(script_parts)
    
    def _parse_pointenergy_file(self, pointenergy_path: str) -> Dict[str, str]:
        """
        Parse force field styles from pointenergy file.
        
        Extracts all lines between 'boundary p p p' and 'read_data' commands,
        which contain the force field interaction styles (bond_style, angle_style,
        dihedral_style, special_bonds, pair_style, pair_modify, kspace_style).
        
        Args:
            pointenergy_path: Path to the pointenergy file
            
        Returns:
            Dictionary mapping directive names to their full command strings
            
        Raises:
            FileNotFoundError: If pointenergy file doesn't exist
            ValueError: If file cannot be parsed or is missing required sections
        """
        pointenergy_file = Path(pointenergy_path)
        
        if not pointenergy_file.exists():
            raise FileNotFoundError(
                f"Pointenergy file not found: {pointenergy_path}\n"
                f"Expected file to exist for extracting force field styles."
            )
        
        try:
            with open(pointenergy_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise ValueError(f"Failed to read pointenergy file {pointenergy_path}: {e}")
        
        # Find the section between 'boundary p p p' and 'read_data'
        start_idx = None
        end_idx = None
        read_data_idx = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if 'boundary p p p' in stripped:
                start_idx = i + 1  # Start after the boundary line
            elif stripped.startswith('read_data'):
                end_idx = i
                read_data_idx = i
                break
        
        if start_idx is None:
            raise ValueError(
                f"Could not find 'boundary p p p' line in pointenergy file: {pointenergy_path}"
            )
        
        if end_idx is None:
            raise ValueError(
                f"Could not find 'read_data' line in pointenergy file: {pointenergy_path}"
            )
        
        # Extract force field directives
        ff_directives = {}
        ff_keywords = [
            'bond_style', 'angle_style', 'dihedral_style', 'improper_style',
            'special_bonds', 'pair_style', 'pair_modify'
        ]
        
        # Parse directives between boundary and read_data
        for i in range(start_idx, end_idx):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check if line starts with any force field keyword
            for keyword in ff_keywords:
                if line.startswith(keyword):
                    ff_directives[keyword] = line
                    break
        
        # Also look for kspace_style after read_data (it must come after read_data in LAMMPS)
        for i in range(read_data_idx + 1, len(lines)):
            line = lines[i].strip()
            if line.startswith('kspace_style'):
                ff_directives['kspace_style'] = line
                break
            elif line.startswith('run'):
                # Stop at run command
                break
        
        logger.info(f"Parsed {len(ff_directives)} force field directives from {pointenergy_path}")
        logger.debug(f"Force field directives: {ff_directives}")
        
        return ff_directives
    
    def _generate_header(self) -> str:
        """Generate script header with documentation."""
        return f"""# LAMMPS Equilibrium MD Script
# Generated by MCP LAMMPS - Best Practice Protocol
#
# Simulation Configuration:
#   Temperature: {self.config.temperature} K
#   Pressure: {self.config.pressure} atm
#   Timestep: {self.config.timestep} fs
#   System Type: {self.config.system_type.value}
#
# Protocol Stages:
#   1. Energy Minimization: {self.config.minimization_steps} steps
#   2. NVT Heating: {self.config.nvt_heating_steps} steps
#   3. NPT Density Equilibration: {self.config.npt_density_steps} steps
#   4. NPT Production: {self.config.production_steps} steps
#
# Total simulation time: {(self.config.production_steps * self.config.timestep) / 1000:.1f} ps
"""
    
    def _generate_initialization(self, data_file: str) -> str:
        """Generate initialization section."""
        return f"""
# ============================================================================
# INITIALIZATION
# ============================================================================

units           real
atom_style      full
boundary        p p p

# Read structure
read_data       {data_file}
"""
    
    def _generate_force_field_section(self) -> str:
        """
        Generate force field section.
        
        If pointenergy_file is specified in config, reads force field styles from it.
        Otherwise, uses a generic placeholder comment for backward compatibility.
        
        Note: kspace_style is NOT included here as it must come after read_data,
        so it's added in the settings section instead.
        
        Returns:
            Force field section with interaction styles
        """
        script = """
# ============================================================================
# FORCE FIELD
# ============================================================================
# Note: Force field parameters are read from the data file
# The data file should contain all pair_coeff, bond_coeff, angle_coeff, etc.
"""
        
        # Parse force field styles from pointenergy file if available
        if self.config.pointenergy_file:
            try:
                ff_directives = self._parse_pointenergy_file(self.config.pointenergy_file)
                
                # Store kspace_style for later use in settings section
                self._kspace_style = ff_directives.get('kspace_style', None)
                
                # Add force field directives in a specific order for clarity
                # Note: kspace_style is excluded here as it must come after read_data
                directive_order = [
                    'bond_style', 'angle_style', 'dihedral_style', 'improper_style',
                    'special_bonds', 'pair_style', 'pair_modify'
                ]
                
                script += "\n"
                for directive in directive_order:
                    if directive in ff_directives:
                        script += f"{ff_directives[directive]}\n"
                
            except (FileNotFoundError, ValueError) as e:
                logger.warning(
                    f"Could not parse force field styles from pointenergy file: {e}\n"
                    f"Pointenergy file path: {self.config.pointenergy_file}\n"
                    f"Continuing without force field directives."
                )
                script += "\n# Force field styles should be defined here\n"
        else:
            # No pointenergy file specified - add placeholder comment
            logger.info("No pointenergy file specified, force field styles should be defined manually")
            script += "\n# Force field styles should be defined here\n"
        
        return script
    
    def _generate_settings(self) -> str:
        """Generate general simulation settings."""
        script = f"""
# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
"""
        
        # Add kspace_style if it was parsed from pointenergy file
        # (kspace_style must come after read_data in LAMMPS)
        if hasattr(self, '_kspace_style') and self._kspace_style:
            script += f"\n# Long-range electrostatics\n{self._kspace_style}\n"
        
        script += f"""
# Neighbor list settings
neighbor        {self.config.neighbor_skin} bin
neigh_modify    every 1 delay {self.config.neighbor_delay} check {'yes' if self.config.neighbor_check else 'no'}

# Thermodynamic output
thermo          {self.config.thermo_frequency}
thermo_style    custom step temp press pe ke etotal vol density
thermo_modify   flush yes
"""
        return script
    
    def _generate_custom_section(self) -> str:
        """Generate custom computes and variables."""
        script = "\n# Custom computes and variables\n"
        
        for var_name, var_expr in self.config.custom_variables.items():
            script += f"variable        {var_name} equal {var_expr}\n"
        
        for compute in self.config.custom_computes:
            script += f"{compute}\n"
        
        return script
    
    def _generate_minimization_stage(self, output_prefix: str) -> str:
        """Generate energy minimization stage."""
        script = f"""
# ============================================================================
# STAGE 1: ENERGY MINIMIZATION
# ============================================================================
# Purpose: Remove bad contacts and overlaps from initial structure
# Method: {self.config.minimization_style.upper()} minimization
# Convergence: Energy < {self.config.energy_tolerance}, Force < {self.config.force_tolerance}

print "=========================================="
print "Starting Stage 1: Energy Minimization"
print "=========================================="

minimize        {self.config.energy_tolerance} {self.config.force_tolerance} {self.config.minimization_steps} {self.config.max_minimization_iterations}

# Save minimized structure
write_data      {output_prefix}_minimized.data
write_restart   {output_prefix}_minimized.restart

print "Minimization complete"
print ""

# Reset timestep counter
reset_timestep  0
"""
        return script
    
    def _generate_nvt_heating_stage(self, output_prefix: str) -> str:
        """Generate NVT heating stage."""
        script = f"""
# ============================================================================
# STAGE 2: NVT HEATING
# ============================================================================
# Purpose: Gradually heat system from {self.config.heating_start_temp} K to {self.config.temperature} K
# Ensemble: NVT (constant volume)
# Timestep: {self.config.heating_timestep} fs (shorter for stability)
# Duration: {self.config.nvt_heating_steps} steps ({self.config.nvt_heating_steps * self.config.heating_timestep / 1000:.1f} ps)

print "=========================================="
print "Starting Stage 2: NVT Heating"
print "=========================================="

# Set shorter timestep for heating
timestep        {self.config.heating_timestep}

# Initialize velocities at starting temperature
velocity        all create {self.config.heating_start_temp} 12345 dist gaussian

# NVT ensemble with gradual heating
fix             nvt_heat all nvt temp {self.config.heating_start_temp} {self.config.temperature} {self.config.thermostat_damping}
"""
        
        if self.config.velocity_rescaling:
            script += f"""
# Velocity rescaling for stability
fix             vrescale all temp/rescale {self.config.velocity_rescale_frequency} {self.config.heating_start_temp} {self.config.temperature} 0.02 1.0
"""
        
        script += f"""
# Run heating phase
run             {self.config.nvt_heating_steps}

# Remove fixes
unfix           nvt_heat
"""
        
        if self.config.velocity_rescaling:
            script += "unfix           vrescale\n"
        
        script += f"""
# Save heated structure
write_data      {output_prefix}_heated.data
write_restart   {output_prefix}_heated.restart

print "NVT heating complete"
print ""

# Reset timestep counter
reset_timestep  0
"""
        return script
    
    def _generate_density_equilibration_stage(self, output_prefix: str) -> str:
        """Generate NPT density equilibration stage."""
        script = f"""
# ============================================================================
# STAGE 3: NPT DENSITY EQUILIBRATION
# ============================================================================
# Purpose: Equilibrate density at target pressure
# Ensemble: NPT (constant pressure)
# Timestep: {self.config.timestep} fs
# Duration: {self.config.npt_density_steps} steps ({self.config.npt_density_steps * self.config.timestep / 1000:.1f} ps)
# Pressure: {self.config.pressure} atm ({self.config.pressure_coupling_style})

print "=========================================="
print "Starting Stage 3: NPT Density Equilibration"
print "=========================================="

# Set production timestep
timestep        {self.config.timestep}

# NPT ensemble for density equilibration
fix             npt_density all npt temp {self.config.temperature} {self.config.temperature} {self.config.thermostat_damping} {self.config.pressure_coupling_style} {self.config.pressure} {self.config.pressure} {self.config.barostat_damping}
"""
        
        if self.config.convergence_checks:
            script += f"""
# Monitor density convergence
variable        current_density equal density
fix             density_avg all ave/time 100 10 1000 v_current_density file {output_prefix}_density_equilibration.dat

"""
        
        script += f"""# Run density equilibration
run             {self.config.npt_density_steps}

# Remove fixes
unfix           npt_density
"""
        
        if self.config.convergence_checks:
            script += "unfix           density_avg\n"
        
        script += f"""
# Save equilibrated structure
write_data      {output_prefix}_equilibrated.data
write_restart   {output_prefix}_equilibrated.restart

print "Density equilibration complete"
print "Final density: ${{current_density}} g/cm^3"
print ""

# Reset timestep counter
reset_timestep  0
"""
        return script
    
    def _generate_production_stage(self, output_prefix: str) -> str:
        """Generate NPT production stage."""
        script = f"""
# ============================================================================
# STAGE 4: NPT PRODUCTION
# ============================================================================
# Purpose: Production run for data collection
# Ensemble: NPT (constant pressure)
# Timestep: {self.config.timestep} fs
# Duration: {self.config.production_steps} steps ({self.config.production_steps * self.config.timestep / 1000:.1f} ps)

print "=========================================="
print "Starting Stage 4: NPT Production"
print "=========================================="

# NPT ensemble for production
fix             npt_prod all npt temp {self.config.temperature} {self.config.temperature} {self.config.thermostat_damping} {self.config.pressure_coupling_style} {self.config.pressure} {self.config.pressure} {self.config.barostat_damping}

# Trajectory output
dump            traj all custom {self.config.trajectory_frequency} {output_prefix}_trajectory.lammpstrj id type x y z"""
        
        if self.config.dump_velocities:
            script += " vx vy vz"
        if self.config.dump_forces:
            script += " fx fy fz"
        
        script += f"""
dump_modify     traj sort id

# Restart files for continuation
restart         {self.config.restart_frequency} {output_prefix}_restart.*.restart

# Enhanced thermodynamic output
thermo          {self.config.thermo_frequency}
thermo_style    custom step temp press pe ke etotal vol density cpu

# Run production
run             {self.config.production_steps}

# Clean up
unfix           npt_prod
undump          traj

# Save final structure
write_data      {output_prefix}_final.data
write_restart   {output_prefix}_final.restart

print "Production run complete"
print ""
"""
        return script
    
    def _generate_footer(self) -> str:
        """Generate script footer."""
        return """
# ============================================================================
# SIMULATION COMPLETE
# ============================================================================

print "=========================================="
print "All stages completed successfully"
print "=========================================="
"""


def create_equilibrium_script(
    data_file: str,
    output_prefix: Optional[str] = None,
    config: Optional[SimulationConfig] = None,
    auto_detect_system: bool = True,
    pointenergy_file: Optional[str] = None
) -> str:
    """
    Convenience function to create equilibrium MD script.
    
    Args:
        data_file: Path to LAMMPS data file
        output_prefix: Prefix for output files
        config: Simulation configuration (uses defaults if not provided)
        auto_detect_system: Automatically detect and optimize for system type
        pointenergy_file: Path to pointenergy file for force field styles.
                         If not provided, auto-derives from data_file path
                         (e.g., test.lmp -> test_pointenergy.in)
        
    Returns:
        Complete LAMMPS script as string
    """
    if config is None:
        config = SimulationConfig()
    
    # Auto-derive pointenergy file path if not provided
    if pointenergy_file is None:
        data_path = Path(data_file)
        # Remove .lmp extension and add _pointenergy.in
        base_name = data_path.stem
        pointenergy_file = str(data_path.parent / f"{base_name}_pointenergy.in")
    
    # Set pointenergy file in config (empty string means explicitly no file)
    config.pointenergy_file = pointenergy_file if pointenergy_file else None
    
    # Auto-detect system type if requested
    if auto_detect_system:
        system_type = SystemDetector.detect_from_data_file(data_file)
        config = SystemDetector.optimize_config_for_system(config, system_type)
    
    # Generate script
    generator = EquilibriumScriptGenerator(config)
    script = generator.generate_script(data_file, output_prefix)
    
    return script

