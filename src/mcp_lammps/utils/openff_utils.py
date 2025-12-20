"""
OpenFF Utilities - Complete integration with OpenFF Toolkit for force field assignment.

This module provides comprehensive OpenFF functionality following patterns from the
OpenFF Toolkit examples, using:
- OpenFF Toolkit for molecule creation and topology
- OpenFF Interchange for system creation and LAMMPS export
- OpenFF Sage 2.2.0 force field (standard release)
- NAGL for modern charge assignment (AM1-BCC-ELF10/GNN)

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
    - LAMMPS export
    - Multi-component liquid box building
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
        self.force_field = ForceField(ff_name)
        
        logger.info(f"Initialized OpenFF force field: {ff_name}")
    
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
        charge_from_molecules: Optional[List[Molecule]] = None
    ) -> Interchange:
        """
        Parametrize topology with force field and create Interchange.
        
        Follows the pattern from notebook:
            forcefield = ForceField("openff-2.2.0.offxml")
            interchange = forcefield.create_interchange(topology)
        
        Args:
            topology: OpenFF Topology with positions and box
            charge_from_molecules: Optional list of molecules with pre-assigned charges
            
        Returns:
            Interchange object ready for export
        """
        try:
            # Create Interchange from SMIRNOFF force field
            # This automatically assigns AM1-BCC charges if not provided
            system = self.force_field.create_interchange(
                topology,
                charge_from_molecules=charge_from_molecules
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
        output_prefix: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """
        Export Interchange system to LAMMPS files.
        
        Follows the pattern: interchange.to_lammps("output_prefix")
        
        This generates both .data and .in files with complete force field parameters.
        
        Args:
            interchange: Interchange system
            output_prefix: Output file prefix (without extension)
            
        Returns:
            Tuple of (data_file_path, script_file_path)
        """
        try:
            output_prefix = str(output_prefix)
            
            # Export to LAMMPS format
            interchange.to_lammps(output_prefix)
            
            data_file = Path(f"{output_prefix}.lmp")
            script_file = Path(f"{output_prefix}_pointenergy.in")
            
            if not data_file.exists():
                raise FileNotFoundError(f"LAMMPS data file not created: {data_file}")
            if not script_file.exists():
                raise FileNotFoundError(f"LAMMPS script file not created: {script_file}")
            
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
            # Calculate total mass
            total_mass = 0.0 * unit.amu
            
            for molecule, count in zip(molecules, counts):
                mol_mass = sum([atom.mass for atom in molecule.atoms])
                total_mass += mol_mass * count 
            
            # Calculate box size
            # density = mass / volume
            # volume = mass / density
            # L = (volume)^(1/3)
            avogadro = unit.avogadro_number
            density_with_units = target_density * unit.gram / unit.centimeter**3
            
            volume = (total_mass / avogadro) / density_with_units
            box_length = volume ** (1/3)
            
            # Convert to Angstroms
            box_size_angstrom = box_length.m_as(unit.angstrom)
            
            logger.info(f"Calculated box size: {box_size_angstrom:.2f} Å for density {target_density} g/cm³")
            
            return box_size_angstrom
            
        except Exception as e:
            logger.error(f"Failed to calculate box size: {e}")
            raise
    
    def generate_random_positions(
        self,
        molecules: List[Molecule],
        counts: List[int],
        box_size: float
    ) -> np.ndarray:
        """
        Generate random positions for molecules in a box.
        
        Args:
            molecules: List of unique molecule types
            counts: Number of each molecule type
            box_size: Box size in Angstroms
            
        Returns:
            Positions array (N_atoms x 3) in Angstroms with units
        """
        try:
            import random
            import math
            
            all_positions = []
            
            for molecule, count in zip(molecules, counts):
                conformer = molecule.conformers[0]
                mol_positions = conformer.m_as(unit.angstrom)
                
                for mol_instance in range(count):
                    # Random position for molecule center
                    center_x = random.uniform(-box_size/3, box_size/3)
                    center_y = random.uniform(-box_size/3, box_size/3)
                    center_z = random.uniform(-box_size/3, box_size/3)
                    
                    # Random rotation angle
                    rotation = random.uniform(0, 2*math.pi)
                    
                    # Apply rotation and translation to molecule
                    for atom_pos in mol_positions:
                        x = atom_pos[0] * math.cos(rotation) - atom_pos[1] * math.sin(rotation) + center_x
                        y = atom_pos[0] * math.sin(rotation) + atom_pos[1] * math.cos(rotation) + center_y
                        z = atom_pos[2] + center_z
                        
                        all_positions.append([x, y, z])
            
            positions_array = np.array(all_positions) * unit.angstrom
            
            logger.info(f"Generated random positions for {len(all_positions)} atoms")
            
            return positions_array
            
        except Exception as e:
            logger.error(f"Failed to generate random positions: {e}")
            raise
    
    def build_liquid_box(
        self,
        molecule_specs: List[Dict[str, Any]],
        target_density: float = 1.0,
        box_type: str = 'cubic',
        output_prefix: Optional[Union[str, Path]] = None
    ) -> Tuple[Interchange, Optional[Path], Optional[Path]]:
        """
        Build complete liquid box system from molecule specifications.
        
        This is a high-level method that combines all steps:
        1. Convert SMILES to molecules with 3D coordinates
        2. Assign AM1-BCC charges
        3. Calculate box size from target density
        4. Generate random positions
        5. Create topology
        6. Create Interchange (parametrize with force field)
        7. Export to LAMMPS files
        
        Args:
            molecule_specs: List of dicts with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            box_type: Box type (cubic, orthorhombic)
            output_prefix: Output file prefix for LAMMPS files
            
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
                
                # Create molecule with 3D coordinates
                molecule = self.from_smiles(smiles, name)
                
                # Assign charges
                self.assign_charges(molecule)
                
                unique_molecules.append(molecule)
                counts.append(count)
                
                # Add molecule 'count' times to list
                for _ in range(count):
                    molecules.append(molecule)
            
            # Calculate box size
            box_size = self.calculate_box_size(unique_molecules, counts, target_density)
            
            # Generate random positions
            positions = self.generate_random_positions(unique_molecules, counts, box_size)
            
            # Create box vectors
            if box_type == 'cubic':
                box_vectors = np.array([
                    [box_size, 0, 0],
                    [0, box_size, 0],
                    [0, 0, box_size]
                ]) * unit.angstrom
            else:
                # For simplicity, use cubic for now
                box_vectors = np.array([
                    [box_size, 0, 0],
                    [0, box_size, 0],
                    [0, 0, box_size]
                ]) * unit.angstrom
            
            # Create topology
            topology = self.create_topology(molecules, box_vectors, positions)
            
            # Create Interchange system
            interchange = self.create_interchange(topology, unique_molecules)
            
            # Export to LAMMPS if output prefix provided
            if output_prefix:
                data_file, script_file = self.to_lammps(interchange, output_prefix)
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
