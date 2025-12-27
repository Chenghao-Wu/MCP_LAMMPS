"""
OpenFF Utilities - Complete integration with OpenFF Toolkit for force field assignment.

This module provides comprehensive OpenFF functionality following patterns from the
OpenFF Toolkit examples, using:
- OpenFF Toolkit for molecule creation and topology
- OpenFF Interchange for system creation and LAMMPS export
- OpenFF Sage 2.2.0 force field (standard release)
- NAGL for modern charge assignment (AM1-BCC-ELF10/GNN)
- TIP3P water model for proper water molecule handling
- Packmol for physically realistic molecular packing

Key Features:
- Automatic water detection and TIP3P model application
- Constraint-free force field for LAMMPS compatibility
- Mixed organic/water system support without export errors

Based on the workflow pattern:
    molecule = Molecule.from_smiles("CCO")
    topology = Topology.from_pdb(pdb_file, unique_molecules=[molecule])
    forcefield = ForceField("openff-2.2.0.offxml")
    interchange = forcefield.create_interchange(topology)
    interchange.to_lammps("output")
"""

import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from openff.toolkit import ForceField, Molecule, Topology
    from openff.interchange import Interchange
    from openff.units import unit
    OPENFF_AVAILABLE = True
except ImportError:
    logger.error("OpenFF Toolkit not available. Please install: pip install openff-toolkit openff-interchange")
    OPENFF_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Some functionality will be limited.")
    RDKIT_AVAILABLE = False


class OpenFFForceField:
    """
    Main interface for OpenFF force field operations.
    
    This class provides a clean API matching the OpenFF Toolkit examples:
    - SMILES to OpenFF Molecule conversion with 3D geometry
    - PDB loading with molecule templates
    - Force field parametrization via Interchange
    - AM1-BCC charge assignment with normalization
    - LAMMPS export (constraints disabled for compatibility)
    - Multi-component liquid box building with Packmol
    - TIP3P water model for accurate water representation
    
    Special Features:
    - Automatic water detection from SMILES
    - TIP3P geometry and charges for water molecules
    - Constraint-free export to avoid LAMMPS compatibility issues
    - Support for mixed organic/water systems
    """
    
    def __init__(self, ff_name: str = 'openff-2.2.0.offxml'):
        """
        Initialize OpenFF force field interface.
        
        Args:
            ff_name: OpenFF force field name (default: Sage 2.2.0)
        """
        if not OPENFF_AVAILABLE:
            raise ImportError("OpenFF Toolkit is required but not available")
        
        self.ff_name = ff_name
        # Load force field and disable constraints for LAMMPS compatibility
        self.force_field = ForceField(ff_name, load_plugins=False)
        
        # Remove Constraints handler if present to avoid LAMMPS export issues
        if 'Constraints' in self.force_field.registered_parameter_handlers:
            self.force_field.deregister_parameter_handler('Constraints')
            logger.info("Disabled Constraints handler for LAMMPS compatibility")
        
        logger.info(f"Initialized OpenFF force field: {ff_name}")
    
    @staticmethod
    def is_water_molecule(smiles: str, molecule: Optional[Molecule] = None) -> bool:
        """
        Detect if a molecule is water based on SMILES or molecular formula.
        
        Args:
            smiles: SMILES string representation
            molecule: Optional OpenFF Molecule object for additional checks
            
        Returns:
            True if molecule is water, False otherwise
        """
        # Check common water SMILES patterns
        water_smiles = ['O', '[H]O[H]', '[OH2]', 'O([H])[H]', '[H][OH]']
        if smiles in water_smiles:
            return True
        
        # If molecule provided, check formula and atom count
        if molecule is not None:
            # Check atom count (water has exactly 3 atoms: 1 O + 2 H)
            if molecule.n_atoms == 3:
                symbols = [atom.symbol for atom in molecule.atoms]
                if symbols.count('O') == 1 and symbols.count('H') == 2:
                    return True
        
        return False
    
    def create_tip3p_water(self, name: str = "water") -> Molecule:
        """
        Create a TIP3P water molecule with proper geometry and charges.
        
        TIP3P is a rigid 3-site water model widely used in molecular dynamics:
        - O-H bond length: 0.9572 Å
        - H-O-H angle: 104.52°
        - Charges: O = -0.834 e, H = +0.417 e each
        
        Args:
            name: Molecule name (default: "water")
            
        Returns:
            OpenFF Molecule object with TIP3P parameters
        """
        try:
            # Create water molecule from SMILES
            water = Molecule.from_smiles('[H]O[H]', allow_undefined_stereo=True)
            water.name = name
            
            # Set TIP3P geometry (in Angstroms)
            # O at origin, H atoms at proper angle
            import math
            
            # TIP3P parameters
            oh_bond_length = 0.9572  # Angstroms
            hoh_angle = 104.52 * math.pi / 180.0  # Convert to radians
            
            # Calculate H atom positions
            # Place O at origin, H1 and H2 symmetrically
            half_angle = hoh_angle / 2.0
            h1_x = oh_bond_length * math.sin(half_angle)
            h1_y = oh_bond_length * math.cos(half_angle)
            h2_x = -h1_x
            h2_y = h1_y
            
            # Create conformer with TIP3P geometry
            positions = np.array([
                [0.0, 0.0, 0.0],  # O atom
                [h1_x, h1_y, 0.0],  # H1 atom
                [h2_x, h2_y, 0.0]   # H2 atom
            ]) * unit.angstrom
            
            water.add_conformer(positions)
            
            # Assign TIP3P charges
            # O: -0.834 e, H: +0.417 e each
            tip3p_charges = np.array([-0.834, 0.417, 0.417]) * unit.elementary_charge
            water.partial_charges = tip3p_charges
            
            logger.info(f"Created TIP3P water molecule: {name}")
            logger.debug(f"TIP3P geometry: O-H = {oh_bond_length:.4f} Å, H-O-H = {104.52:.2f}°")
            
            return water
            
        except Exception as e:
            logger.error(f"Failed to create TIP3P water: {e}")
            raise
    
    def from_smiles(
        self,
        smiles: str,
        name: str,
        optimize: bool = True,
        num_conformers: int = 1
    ) -> Molecule:
        """
        Create OpenFF Molecule from SMILES string with 3D coordinates.
        
        Follows the pattern: molecule = Molecule.from_smiles("CCO")
        
        Args:
            smiles: SMILES string representation
            name: Molecule name
            optimize: Whether to optimize geometry with RDKit UFF
            num_conformers: Number of conformers to generate
            
        Returns:
            OpenFF Molecule object with 3D coordinates
        """
        try:
            # Create OpenFF molecule from SMILES
            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            molecule.name = name
            
            # Generate 3D conformer using RDKit
            if RDKIT_AVAILABLE and optimize:
                # Convert to RDKit for conformer generation
                rdmol = molecule.to_rdkit()
                
                # Generate conformers
                AllChem.EmbedMultipleConfs(
                    rdmol,
                    numConfs=num_conformers,
                    randomSeed=23,
                    useRandomCoords=True
                )
                
                # Optimize geometry with UFF
                if optimize:
                    for conf_id in range(rdmol.GetNumConformers()):
                        AllChem.UFFOptimizeMolecule(rdmol, confId=conf_id)
                
                # Convert back to OpenFF with coordinates
                molecule = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
                molecule.name = name
            
            logger.info(f"Created OpenFF molecule from SMILES: {name} ({smiles})")
            return molecule
            
        except Exception as e:
            logger.error(f"Failed to convert SMILES to OpenFF molecule: {e}")
            raise
    
    def from_pdb(
        self,
        pdb_file: Union[str, Path],
        unique_molecules: List[Molecule]
    ) -> Topology:
        """
        Load topology from PDB file with molecule templates.
        
        Follows the pattern from notebook:
            topology = Topology.from_pdb(pdb_file, unique_molecules=[ethanol, cyclohexane])
        
        Args:
            pdb_file: Path to PDB file
            unique_molecules: List of unique molecule templates with bond orders
            
        Returns:
            OpenFF Topology object
        """
        try:
            topology = Topology.from_pdb(
                str(pdb_file),
                unique_molecules=unique_molecules
            )
            
            logger.info(f"Loaded topology from PDB: {pdb_file} with {topology.n_molecules} molecules")
            return topology
            
        except Exception as e:
            logger.error(f"Failed to load topology from PDB: {e}")
            raise
    
    def create_interchange(
        self,
        topology: Topology,
        charge_from_molecules: Optional[List[Molecule]] = None,
        allow_nonintegral_charges: bool = False
    ) -> Interchange:
        """
        Parametrize topology with force field and create Interchange.
        
        Follows the pattern from notebook:
            forcefield = ForceField("openff-2.2.0.offxml")
            interchange = forcefield.create_interchange(topology)
        
        Note: Constraints are disabled in the force field initialization
        to ensure compatibility with LAMMPS export for water molecules.
        
        Args:
            topology: OpenFF Topology with positions and box
            charge_from_molecules: Optional list of molecules with pre-assigned charges
            allow_nonintegral_charges: Allow non-integral total charges
            
        Returns:
            Interchange object ready for export
        """
        try:
            # Create Interchange from SMIRNOFF force field
            # This automatically assigns AM1-BCC charges if not provided
            system = self.force_field.create_interchange(
                topology,
                charge_from_molecules=charge_from_molecules,
                allow_nonintegral_charges=allow_nonintegral_charges
            )
            
            logger.info(f"Created Interchange system with {system.topology.n_atoms} atoms")
            
            return system
            
        except Exception as e:
            logger.error(f"Failed to create Interchange system: {e}")
            raise
    
    def assign_charges(
        self,
        molecule: Molecule,
        charge_model: str = 'am1bcc'
    ) -> Molecule:
        """
        Assign partial charges using AM1-BCC method.
        
        This uses the AM1-BCC charge model from AmberTools, which is the
        default charge method specified by the OpenFF Sage force field.
        
        Args:
            molecule: OpenFF Molecule object
            charge_model: Charge model to use (default: 'am1bcc')
            
        Returns:
            Molecule with assigned partial charges
        """
        try:
            # Use molecule's built-in method which uses the toolkit registry
            molecule.assign_partial_charges(
                partial_charge_method=charge_model,
                use_conformers=molecule.conformers
            )
            
            # Normalize charges to ensure they sum to exactly zero
            molecule.partial_charges = self._normalize_charges(
                molecule.partial_charges,
                decimals=6
            )
            
            logger.info(f"Assigned {charge_model} charges to molecule: {molecule.name}")
            logger.debug(f"Total charge: {sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges]):.6f}")
            
            return molecule
            
        except Exception as e:
            logger.error(f"Failed to assign charges: {e}")
            raise
    
    def _normalize_charges(
        self,
        charges: Any,
        decimals: int = 6
    ) -> Any:
        """
        Round partial charges to fixed decimals and renormalize to sum to zero.
        
        This ensures charges sum to exactly zero for neutral molecules.
        
        Args:
            charges: OpenFF Quantity array of partial charges
            decimals: Number of decimal places (default: 6)
            
        Returns:
            Normalized charges with proper units
        """
        # Convert to numpy array without units
        charges_array = np.asarray(charges.m_as(unit.elementary_charge), dtype=float)
        
        # First correction: distribute error uniformly
        correction = charges_array.sum()
        charges_array -= correction / len(charges_array)
        
        # Round to specified decimals
        charges_array = np.round(charges_array, decimals)
        
        # Second correction: incremental adjustment for floating-point errors
        correction = charges_array.sum()
        counts = int(correction * 10**decimals)
        
        for count in range(abs(counts)):
            if correction > 0:
                charges_array[count] -= 10**(-decimals)
            elif correction < 0:
                charges_array[count] += 10**(-decimals)
        
        # Final rounding
        charges_array = np.round(charges_array, decimals)
        
        # Verify charge neutrality
        final_sum = charges_array.sum()
        if abs(final_sum) > 1e-6:
            logger.warning(f"Charge normalization may be imperfect: total = {final_sum:.8f}")
        
        # Return with proper units
        return charges_array * unit.elementary_charge
    
    def to_lammps(
        self,
        interchange: Interchange,
        output_prefix: Union[str, Path],
        script_type: str = "point_energy",
        simulation_config: Optional[Any] = None
    ) -> Tuple[Path, Path]:
        """
        Export Interchange system to LAMMPS files.
        
        Follows the pattern: interchange.to_lammps("output_prefix")
        
        This generates both .data and .in files with complete force field parameters.
        
        Args:
            interchange: Interchange system
            output_prefix: Output file prefix (without extension)
            script_type: Type of script to generate:
                - "point_energy": Simple point energy calculation (default, backward compatible)
                - "equilibrium": Full equilibrium MD protocol with multi-stage equilibration
            simulation_config: SimulationConfig object for equilibrium scripts (optional)
            
        Returns:
            Tuple of (data_file_path, script_file_path)
        """
        try:
            output_prefix = str(output_prefix)
            
            # Export to LAMMPS format (always generates data file + point energy script)
            interchange.to_lammps(output_prefix)
            
            data_file = Path(f"{output_prefix}.lmp")
            point_energy_script = Path(f"{output_prefix}_pointenergy.in")
            
            if not data_file.exists():
                raise FileNotFoundError(f"LAMMPS data file not created: {data_file}")
            if not point_energy_script.exists():
                raise FileNotFoundError(f"LAMMPS script file not created: {point_energy_script}")
            
            # Determine which script to return
            if script_type == "equilibrium":
                # Generate equilibrium MD script using new generator
                from .script_generator import create_equilibrium_script, SimulationConfig
                
                # Use provided config or create default
                if simulation_config is None:
                    simulation_config = SimulationConfig()
                
                # Generate equilibrium script with pointenergy file for force field styles
                equilibrium_script_content = create_equilibrium_script(
                    str(data_file),
                    output_prefix,
                    simulation_config,
                    auto_detect_system=True,
                    pointenergy_file=str(point_energy_script)
                )
                
                # Write equilibrium script
                script_file = Path(f"{output_prefix}_equilibrium.in")
                with open(script_file, 'w') as f:
                    f.write(equilibrium_script_content)
                
                logger.info(f"Generated equilibrium MD script: {script_file}")
                
                # Keep point energy script for reference
                logger.info(f"Point energy script also available: {point_energy_script}")
                
            else:
                # Use point energy script (default, backward compatible)
                script_file = point_energy_script
            
            logger.info(f"Exported Interchange to LAMMPS files: {output_prefix}")
            
            return data_file, script_file
            
        except Exception as e:
           logger.error(f"Failed to export Interchange to LAMMPS: {e}")
           raise
    
    def create_topology(
        self,
        molecules: List[Molecule],
        box_vectors: Optional[Any] = None,
        positions: Optional[np.ndarray] = None
    ) -> Topology:
        """
        Create OpenFF Topology from list of molecules.
        
        Args:
            molecules: List of OpenFF Molecule objects (can contain duplicates)
            box_vectors: Box vectors with units (3x3 array)
            positions: Atomic positions (Nx3 array) in Angstroms
            
        Returns:
            OpenFF Topology object
        """
        try:
            topology = Topology()
            
            # Add all molecules to topology
            for molecule in molecules:
                topology.add_molecule(molecule)
            
            # Set positions if provided
            if positions is not None:
                # Ensure positions have proper units
                if not hasattr(positions, 'units'):
                    positions = positions * unit.angstrom
                topology.set_positions(positions)
            
            # Set box vectors if provided
            if box_vectors is not None:
                topology.box_vectors = box_vectors
            
            logger.info(f"Created topology with {topology.n_molecules} molecules, "
                       f"{topology.n_atoms} atoms")
            
            return topology
            
        except Exception as e:
            logger.error(f"Failed to create topology: {e}")
            raise
    
    def calculate_box_size(
        self,
        molecules: List[Molecule],
        counts: List[int],
        target_density: float = 0.5
    ) -> float:
        """
        Calculate cubic box size based on target density.
        
        Args:
            molecules: List of unique molecule types
            counts: Number of each molecule type
            target_density: Target density in g/cm³
            
        Returns:
            Box size in Angstroms
        """
        try:
            # Calculate total mass in atomic mass units
            total_mass_amu = 0.0
            
            for molecule, count in zip(molecules, counts):
                mol_mass = sum([atom.mass.m_as(unit.amu) for atom in molecule.atoms])
                total_mass_amu += mol_mass * count
            
            # Convert mass to grams
            # 1 amu = 1.66054e-24 g (or 1 g/mol per Avogadro's number)
            avogadro = 6.022e23  # molecules/mol
            total_mass_g = (total_mass_amu / avogadro)
            
            # Calculate volume from density
            # density = mass / volume  =>  volume = mass / density
            volume_cm3 = total_mass_g / target_density
            
            # Convert volume from cm³ to ų
            volume_angstrom3 = volume_cm3 * 1e24  # 1 cm³ = 10²⁴ ų
            
            # Calculate box length for cubic box
            box_size_angstrom = volume_angstrom3 ** (1/3)
            
            logger.info(f"Calculated box size: {box_size_angstrom:.2f} Å for density {target_density} g/cm³")
            logger.info(f"  Total mass: {total_mass_amu:.2f} amu, Volume: {volume_angstrom3:.2f} ų")
            
            return box_size_angstrom
            
        except Exception as e:
            logger.error(f"Failed to calculate box size: {e}")
            raise
    
    def generate_packmol_positions(
        self,
        molecules: List[Molecule],
        counts: List[int],
        box_size: float,
        tolerance: float = 2.0
    ) -> np.ndarray:
        """
        Generate well-packed molecular positions using Packmol.
        
        This method uses the Packmol molecular packing tool to generate
        physically realistic initial configurations without atomic overlaps.
        
        Args:
            molecules: List of unique molecule types
            counts: Number of each molecule type
            box_size: Box size in Angstroms
            tolerance: Minimum distance between atoms in Angstroms
            
        Returns:
            Positions array (N_atoms x 3) in Angstroms with units
            
        Raises:
            PackmolError: If Packmol is not available or packing fails
        """
        try:
            from .packmol_utils import packmol_utils, PackmolError
            
            logger.info(f"Using Packmol to pack {sum(counts)} molecules into "
                       f"{box_size:.2f} Å box with tolerance {tolerance} Å")
            
            # Use Packmol to generate packed positions
            positions_array = packmol_utils.pack_molecules(
                molecules=molecules,
                counts=counts,
                box_size=box_size,
                tolerance=tolerance
            )
            
            # Add units to positions
            positions_with_units = positions_array * unit.angstrom
            
            logger.info(f"Successfully packed {len(positions_array)} atoms using Packmol")
            
            return positions_with_units
            
        except PackmolError as e:
            logger.error(f"Packmol packing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate Packmol positions: {e}")
            raise
    
    def build_liquid_box(
        self,
        molecule_specs: List[Dict[str, Any]],
        target_density: float = 1.0,
        box_type: str = 'cubic',
        output_prefix: Optional[Union[str, Path]] = None,
        water_model: str = 'tip3p',
        generate_equilibrium_script: bool = True,
        simulation_config: Optional[Any] = None
    ) -> Tuple[Interchange, Optional[Path], Optional[Path]]:
        """
        Build complete liquid box system from molecule specifications.
        
        This is a high-level method that combines all steps:
        1. Convert SMILES to molecules with 3D coordinates (or TIP3P for water)
        2. Assign AM1-BCC charges (skipped for water with TIP3P)
        3. Calculate box size from target density
        4. Pack molecules using Packmol (replaces random positioning)
        5. Create topology
        6. Create Interchange (parametrize with force field)
        7. Export to LAMMPS files with equilibrium MD script
        
        Special handling for water:
        - Water molecules (SMILES='O') are automatically detected
        - Uses TIP3P water model with rigid geometry and preset charges
        - TIP3P avoids constraint issues during LAMMPS export
        
        Args:
            molecule_specs: List of dicts with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            box_type: Box type (cubic, orthorhombic)
            output_prefix: Output file prefix for LAMMPS files
            water_model: Water model to use ('tip3p', 'tip4p', 'spce'), default 'tip3p'
            generate_equilibrium_script: Generate equilibrium MD script (default: True)
            simulation_config: SimulationConfig for equilibrium script customization
            
        Returns:
            Tuple of (Interchange, data_file_path, script_file_path)
        """
        try:
            # Process all molecules
            molecules = []
            unique_molecules = []
            counts = []
            
            for spec in molecule_specs:
                smiles = spec['smiles']
                count = spec['count']
                name = spec.get('name', f'mol_{len(unique_molecules)}')
                
                # Check if this is water and use appropriate model
                if self.is_water_molecule(smiles):
                    logger.info(f"Detected water molecule, using {water_model.upper()} model")
                    if water_model.lower() == 'tip3p':
                        molecule = self.create_tip3p_water(name)
                        # TIP3P already has charges assigned, skip charge assignment
                    else:
                        # For other water models, fall back to standard method
                        logger.warning(f"Water model '{water_model}' not fully implemented, using TIP3P")
                        molecule = self.create_tip3p_water(name)
                else:
                    # Create molecule with 3D coordinates
                    molecule = self.from_smiles(smiles, name)
                    # Assign charges for non-water molecules
                    self.assign_charges(molecule)
                
                unique_molecules.append(molecule)
                counts.append(count)
                
                # Add molecule 'count' times to list
                for _ in range(count):
                    molecules.append(molecule)
            
            # Calculate box size
            box_size = self.calculate_box_size(unique_molecules, counts, target_density)
            
            # Pack molecules using Packmol
            # rescale box size to be a multiple of 2 to make sure the box size is large enough for the molecules to be separated
            box_size_rescaled = box_size * 2
            positions = self.generate_packmol_positions(unique_molecules, counts, box_size_rescaled )
            
            # Create box vectors
            if box_type == 'cubic':
                box_vectors = np.array([
                    [box_size_rescaled, 0, 0],
                    [0, box_size_rescaled, 0],
                    [0, 0, box_size_rescaled]
                ]) * unit.angstrom
            else:
                # For simplicity, use cubic for now
                box_vectors = np.array([
                    [box_size_rescaled, 0, 0],
                    [0, box_size_rescaled, 0],
                    [0, 0, box_size_rescaled]
                ]) * unit.angstrom
            
            # Create topology
            topology = self.create_topology(molecules, box_vectors, positions)
            
            # Create Interchange system
            interchange = self.create_interchange(topology, unique_molecules)
            
            # Export to LAMMPS if output prefix provided
            if output_prefix:
                # Determine script type based on generate_equilibrium_script flag
                script_type = "equilibrium" if generate_equilibrium_script else "point_energy"
                
                data_file, script_file = self.to_lammps(
                    interchange, 
                    output_prefix,
                    script_type=script_type,
                    simulation_config=simulation_config
                )
                return interchange, data_file, script_file
            
            return interchange, None, None
            
        except Exception as e:
            logger.error(f"Failed to build liquid box: {e}")
            raise
    
    def get_system_info(self, interchange: Interchange) -> Dict[str, Any]:
        """
        Extract system information from Interchange.
        
        Args:
            interchange: Interchange system
            
        Returns:
            Dictionary with system information
        """
        try:
            info = {
                "n_atoms": interchange.topology.n_atoms,
                "n_molecules": interchange.topology.n_molecules,
                "n_bonds": interchange.topology.n_bonds,
                "box_vectors": interchange.box.m_as(unit.angstrom) if interchange.box is not None else None,
                "force_field": self.ff_name,
                "charge_model": "am1bcc"
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}


# Create a global instance for easy access
openff_forcefield = OpenFFForceField()
