"""
Force Field Utilities - High-level wrapper for OpenFF force field operations.

This module provides a backward-compatible interface to the refactored OpenFF utilities.
It wraps the OpenFFForceField class to maintain existing API compatibility while using
OpenFF Sage 2.2.0 and following the notebook's clean patterns.

For direct access to OpenFF operations, use openff_utils.OpenFFForceField.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from .openff_utils import openff_forcefield, OpenFFForceField, OPENFF_AVAILABLE
except ImportError:
    logger.error("Failed to import openff_utils")
    OPENFF_AVAILABLE = False


class ForceFieldUtils:
    """
    High-level wrapper for OpenFF force field utilities.
    
    This class provides a simplified interface to OpenFF functionality while
    maintaining backward compatibility with existing code:
    - Charge assignment using AM1-BCC
    - System creation from SMILES
    - LAMMPS file generation
    - Multi-component system building
    - PDB loading with molecule templates
    """
    
    def __init__(self, ff_name: str = 'openff-2.2.0.offxml'):
        """
        Initialize force field utilities with OpenFF.
        
        Args:
            ff_name: OpenFF force field name (default: Sage 2.2.0)
        """
        if not OPENFF_AVAILABLE:
            raise ImportError("OpenFF Toolkit is required but not available")
        
        self.openff = OpenFFForceField(ff_name)
        self.ff_name = ff_name
        
        logger.info(f"Initialized ForceFieldUtils with OpenFF: {ff_name}")
    
    def assign_charges(self, molecule, charge_model='am1bcc'):
        """
        Assign partial charges to a molecule using AM1-BCC.
        
        Args:
            molecule: OpenFF Molecule object
            charge_model: Charge model to use (default: 'am1bcc')
            
        Returns:
            Molecule with assigned charges
        """
        return self.openff.assign_charges(molecule, charge_model)
    
    def create_system_from_smiles(
        self,
        smiles: str,
        name: str,
        optimize: bool = True
    ):
        """
        Complete workflow: SMILES -> Interchange system.
        
        Args:
            smiles: SMILES string
            name: Molecule name
            optimize: Whether to optimize geometry
            
        Returns:
            Interchange system ready for export
        """
        try:
            # Create molecule from SMILES
            molecule = self.openff.from_smiles(smiles, name, optimize)
            
            # Assign charges
            self.openff.assign_charges(molecule)
            
            # Create topology with single molecule
            topology = self.openff.create_topology([molecule])
            
            # Create Interchange system
            system = self.openff.create_interchange(topology, [molecule])
            
            logger.info(f"Created system from SMILES: {name}")
            return system
            
        except Exception as e:
            logger.error(f"Failed to create system from SMILES: {e}")
            raise
    
    def create_lammps_files(
        self,
        interchange,
        output_prefix: str
    ) -> Tuple[Path, Path]:
        """
        Export Interchange system to LAMMPS files.
        
        Args:
            interchange: Interchange system
            output_prefix: Output file prefix
            
        Returns:
            Tuple of (data_file_path, script_file_path)
        """
        return self.openff.to_lammps(interchange, output_prefix)
    
    def build_multi_component_system(
        self,
        molecule_specs: List[Dict[str, Any]],
        target_density: float = 1.0,
        output_prefix: Optional[str] = None
    ):
        """
        Build multi-component liquid system.
        
        Args:
            molecule_specs: List of dicts with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            output_prefix: Output file prefix for LAMMPS files
            
        Returns:
            Interchange system (and file paths if output_prefix provided)
        """
        return self.openff.build_liquid_box(
            molecule_specs,
            target_density,
            box_type='cubic',
            output_prefix=output_prefix
        )
    
    def get_force_field_info(self) -> Dict[str, str]:
        """
        Get information about the loaded force field.
        
        Returns:
            Dictionary with force field information
        """
        return {
            "force_field": self.ff_name,
            "type": "OpenFF",
            "version": "Sage 2.2.0",
            "charge_model": "am1bcc"
        }
    
    def validate_molecule(self, molecule) -> Tuple[bool, List[str]]:
        """
        Validate that a molecule is properly set up.
        
        Args:
            molecule: OpenFF Molecule object
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check for conformers
            if len(molecule.conformers) == 0:
                issues.append("Molecule has no 3D conformers")
            
            # Check for charges
            if molecule.partial_charges is None:
                issues.append("Molecule has no partial charges assigned")
            else:
                # Check charge neutrality
                from openff.units import unit
                total_charge = sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges])
                if abs(total_charge) > 1e-5:
                    issues.append(f"Molecule is not charge-neutral: total charge = {total_charge:.6f}")
            
            # Check for proper atom types
            if molecule.n_atoms == 0:
                issues.append("Molecule has no atoms")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.debug(f"Molecule {molecule.name} validation passed")
            else:
                logger.warning(f"Molecule {molecule.name} validation issues: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating molecule: {e}")
            return False, [f"Validation error: {e}"]
    
    # Additional backward-compatible methods
    
    def smiles_to_openff_molecule(
        self,
        smiles: str,
        name: str,
        optimize: bool = True,
        num_conformers: int = 1
    ):
        """
        Convert SMILES string to OpenFF Molecule with 3D coordinates.
        
        Backward-compatible alias for from_smiles().
        
        Args:
            smiles: SMILES string representation
            name: Molecule name
            optimize: Whether to optimize geometry
            num_conformers: Number of conformers to generate
            
        Returns:
            OpenFF Molecule object with 3D coordinates
        """
        return self.openff.from_smiles(smiles, name, optimize, num_conformers)
    
    def assign_charges_nagl(
        self,
        molecule,
        charge_model: str = 'am1bcc'
    ):
        """
        Assign partial charges using AM1-BCC.
        
        Backward-compatible alias for assign_charges().
        Note: Despite the name, this now uses AM1-BCC (not NAGL).
        
        Args:
            molecule: OpenFF Molecule object
            charge_model: Charge model to use (default: 'am1bcc')
            
        Returns:
            Molecule with assigned partial charges
        """
        return self.openff.assign_charges(molecule, charge_model)
    
    def create_topology(
        self,
        molecules: List,
        box_vectors: Optional[Any] = None,
        positions: Optional[Any] = None
    ):
        """
        Create OpenFF Topology from list of molecules.
        
        Args:
            molecules: List of OpenFF Molecule objects
            box_vectors: Box vectors with units (3x3 array)
            positions: Atomic positions (Nx3 array)
            
        Returns:
            OpenFF Topology object
        """
        return self.openff.create_topology(molecules, box_vectors, positions)
    
    def create_interchange_system(
        self,
        topology,
        molecules_with_charges: List
    ):
        """
        Create OpenFF Interchange system for LAMMPS export.
        
        Args:
            topology: OpenFF Topology with positions and box
            molecules_with_charges: List of unique molecules with assigned charges
            
        Returns:
            Interchange object ready for export
        """
        return self.openff.create_interchange(topology, molecules_with_charges)
    
    def interchange_to_lammps(
        self,
        interchange,
        output_prefix: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """
        Export Interchange system to LAMMPS files.
        
        Args:
            interchange: Interchange system
            output_prefix: Output file prefix (without extension)
            
        Returns:
            Tuple of (data_file_path, script_file_path)
        """
        return self.openff.to_lammps(interchange, output_prefix)
    
    def calculate_box_size(
        self,
        molecules: List,
        counts: List[int],
        target_density: float = 1.0
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
        return self.openff.calculate_box_size(molecules, counts, target_density)
    
    def generate_random_positions(
        self,
        molecules: List,
        counts: List[int],
        box_size: float
    ):
        """
        Generate random positions for molecules in a box.
        
        Args:
            molecules: List of unique molecule types
            counts: Number of each molecule type
            box_size: Box size in Angstroms
            
        Returns:
            Positions array (N_atoms x 3) in Angstroms
        """
        return self.openff.generate_random_positions(molecules, counts, box_size)
    
    def build_liquid_box(
        self,
        molecule_specs: List[Dict[str, Any]],
        target_density: float = 1.0,
        box_type: str = 'cubic',
        output_prefix: Optional[Union[str, Path]] = None
    ):
        """
        Build complete liquid box system from molecule specifications.
        
        Args:
            molecule_specs: List of dicts with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            box_type: Box type (cubic, orthorhombic)
            output_prefix: Output file prefix for LAMMPS files
            
        Returns:
            Tuple of (Interchange, data_file_path, script_file_path)
        """
        return self.openff.build_liquid_box(
            molecule_specs,
            target_density,
            box_type,
            output_prefix
        )
    
    def get_system_info(self, interchange) -> Dict[str, Any]:
        """
        Extract system information from Interchange.
        
        Args:
            interchange: Interchange system
            
        Returns:
            Dictionary with system information
        """
        return self.openff.get_system_info(interchange)


# Create a global instance for easy access
forcefield_utils = ForceFieldUtils()
