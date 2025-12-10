"""
Molecular Utilities - Tools for molecular structure handling and manipulation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolAlign import AlignMol
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Some molecular functionality will be limited.")
    RDKIT_AVAILABLE = False

try:
    import openbabel as ob
    OPENBABEL_AVAILABLE = True
except ImportError:
    logger.warning("OpenBabel not available. Some format conversion functionality will be limited.")
    OPENBABEL_AVAILABLE = False


class MolecularUtils:
    """
    Utilities for molecular structure handling and manipulation.
    
    This class provides functionality for:
    - Converting SMILES to 3D structures
    - Calculating molecular properties
    - Format conversions between different molecular file formats
    - Molecular validation and sanitization
    """
    
    def __init__(self):
        """Initialize the molecular utilities."""
        self.rdkit_available = RDKIT_AVAILABLE
        self.openbabel_available = OPENBABEL_AVAILABLE
        
        if not (self.rdkit_available or self.openbabel_available):
            logger.error("Neither RDKit nor OpenBabel available. Molecular functionality severely limited.")
    
    def smiles_to_3d(
        self,
        smiles: str,
        num_conformers: int = 1,
        optimize: bool = True,
        random_seed: int = 42
    ) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to 3D molecular structure.
        
        Args:
            smiles: SMILES string representation of the molecule
            num_conformers: Number of conformers to generate
            optimize: Whether to optimize the geometry
            random_seed: Random seed for reproducible conformer generation
            
        Returns:
            RDKit molecule object with 3D coordinates or None if failed
        """
        if not self.rdkit_available:
            logger.error("RDKit not available for SMILES to 3D conversion")
            return None
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Failed to parse SMILES: {smiles}")
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D conformer(s)
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed
            
            if num_conformers == 1:
                result = AllChem.EmbedMolecule(mol, params)
                if result != 0:
                    logger.error(f"Failed to embed molecule from SMILES: {smiles}")
                    return None
            else:
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
                if len(conf_ids) == 0:
                    logger.error(f"Failed to embed multiple conformers from SMILES: {smiles}")
                    return None
            
            # Optimize geometry
            if optimize:
                if num_conformers == 1:
                    AllChem.MMFFOptimizeMolecule(mol)
                else:
                    for conf_id in range(num_conformers):
                        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            
            logger.info(f"Successfully generated 3D structure from SMILES: {smiles}")
            return mol
            
        except Exception as e:
            logger.error(f"Error converting SMILES to 3D: {e}")
            return None
    
    def calculate_molecular_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate molecular properties.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of molecular properties
        """
        if not self.rdkit_available or mol is None:
            return {}
        
        try:
            properties = {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol),
                "formal_charge": Chem.rdmolops.GetFormalCharge(mol)
            }
            
            logger.info(f"Calculated properties for molecule with {properties['heavy_atoms']} heavy atoms")
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating molecular properties: {e}")
            return {}
    
    def mol_to_xyz(self, mol: Chem.Mol, conf_id: int = 0) -> str:
        """
        Convert RDKit molecule to XYZ format string.
        
        Args:
            mol: RDKit molecule object
            conf_id: Conformer ID to use
            
        Returns:
            XYZ format string
        """
        if not self.rdkit_available or mol is None:
            return ""
        
        try:
            conf = mol.GetConformer(conf_id)
            num_atoms = mol.GetNumAtoms()
            
            xyz_lines = [str(num_atoms), "Generated by MCP LAMMPS"]
            
            for i in range(num_atoms):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                symbol = atom.GetSymbol()
                xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
            
            return "\n".join(xyz_lines)
            
        except Exception as e:
            logger.error(f"Error converting molecule to XYZ: {e}")
            return ""
    
    def mol_to_pdb(self, mol: Chem.Mol, conf_id: int = 0) -> str:
        """
        Convert RDKit molecule to PDB format string.
        
        Args:
            mol: RDKit molecule object
            conf_id: Conformer ID to use
            
        Returns:
            PDB format string
        """
        if not self.rdkit_available or mol is None:
            return ""
        
        try:
            pdb_block = Chem.MolToPDBBlock(mol, confId=conf_id)
            return pdb_block
            
        except Exception as e:
            logger.error(f"Error converting molecule to PDB: {e}")
            return ""
    
    def detect_molecule_type(self, mol: Chem.Mol) -> str:
        """
        Detect the type of organic molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Molecule type string
        """
        if not self.rdkit_available or mol is None:
            return "unknown"
        
        try:
            # Get basic properties
            num_atoms = mol.GetNumAtoms()
            num_rings = Descriptors.RingCount(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            # Check for specific functional groups
            has_alcohol = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH]"))) > 0
            has_carbonyl = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[C]=[O]"))) > 0
            has_amine = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N]"))) > 0
            has_ether = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[O]([C])[C]"))) > 0
            
            # Classify molecule type
            if num_atoms <= 10:
                molecule_type = "small_organic"
            elif num_aromatic_rings > 0:
                molecule_type = "aromatic"
            elif has_alcohol:
                molecule_type = "alcohol"
            elif has_carbonyl:
                molecule_type = "carbonyl"
            elif has_amine:
                molecule_type = "amine"
            elif has_ether:
                molecule_type = "ether"
            elif num_rings > 0:
                molecule_type = "cyclic"
            else:
                molecule_type = "aliphatic"
            
            logger.info(f"Detected molecule type: {molecule_type}")
            return molecule_type
            
        except Exception as e:
            logger.error(f"Error detecting molecule type: {e}")
            return "unknown"
    
    def validate_molecule(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        Validate molecular structure and return issues.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if not self.rdkit_available or mol is None:
            return False, ["RDKit not available or invalid molecule"]
        
        issues = []
        
        try:
            # Check for basic validity
            if mol.GetNumAtoms() == 0:
                issues.append("Molecule has no atoms")
            
            # Check for unreasonable molecular weight
            mw = Descriptors.MolWt(mol)
            if mw > 1000:
                issues.append(f"Large molecular weight: {mw:.1f}")
            elif mw < 10:
                issues.append(f"Very small molecular weight: {mw:.1f}")
            
            # Check for formal charge
            formal_charge = Chem.rdmolops.GetFormalCharge(mol)
            if abs(formal_charge) > 2:
                issues.append(f"High formal charge: {formal_charge}")
            
            # Check for unusual valences
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                issues.append(f"Sanitization failed: {str(e)}")
            
            # Check for 3D coordinates
            try:
                conf = mol.GetConformer()
                if conf.Is3D():
                    # Check for reasonable bond lengths
                    for bond in mol.GetBonds():
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
                        begin_pos = conf.GetAtomPosition(begin_idx)
                        end_pos = conf.GetAtomPosition(end_idx)
                        
                        distance = np.sqrt(
                            (begin_pos.x - end_pos.x)**2 +
                            (begin_pos.y - end_pos.y)**2 +
                            (begin_pos.z - end_pos.z)**2
                        )
                        
                        if distance > 3.0:  # Very long bond
                            issues.append(f"Unusually long bond: {distance:.2f} Å")
                        elif distance < 0.5:  # Very short bond
                            issues.append(f"Unusually short bond: {distance:.2f} Å")
            except:
                issues.append("No 3D coordinates available")
            
            is_valid = len(issues) == 0
            logger.info(f"Molecule validation: {'passed' if is_valid else 'failed'} with {len(issues)} issues")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating molecule: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def sanitize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Sanitize and clean up molecular structure.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Sanitized molecule or None if failed
        """
        if not self.rdkit_available or mol is None:
            return None
        
        try:
            # Make a copy
            mol_copy = Chem.Mol(mol)
            
            # Remove hydrogens and re-add them
            mol_copy = Chem.RemoveHs(mol_copy)
            mol_copy = Chem.AddHs(mol_copy)
            
            # Sanitize
            Chem.SanitizeMol(mol_copy)
            
            # Assign stereochemistry
            Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
            
            logger.info("Successfully sanitized molecule")
            return mol_copy
            
        except Exception as e:
            logger.error(f"Error sanitizing molecule: {e}")
            return None
    
    def convert_format(
        self,
        input_data: str,
        input_format: str,
        output_format: str
    ) -> Optional[str]:
        """
        Convert between different molecular file formats using OpenBabel.
        
        Args:
            input_data: Input molecular data as string
            input_format: Input format (smi, mol2, sdf, pdb, xyz)
            output_format: Output format (smi, mol2, sdf, pdb, xyz)
            
        Returns:
            Converted molecular data as string or None if failed
        """
        if not self.openbabel_available:
            logger.error("OpenBabel not available for format conversion")
            return None
        
        try:
            # Create OpenBabel molecule
            mol = ob.OBMol()
            conv = ob.OBConversion()
            
            # Set input format
            if not conv.SetInFormat(input_format):
                logger.error(f"Unsupported input format: {input_format}")
                return None
            
            # Read molecule
            if not conv.ReadString(mol, input_data):
                logger.error(f"Failed to read molecule in {input_format} format")
                return None
            
            # Set output format
            if not conv.SetOutFormat(output_format):
                logger.error(f"Unsupported output format: {output_format}")
                return None
            
            # Convert and return
            output_data = conv.WriteString(mol)
            logger.info(f"Successfully converted from {input_format} to {output_format}")
            return output_data
            
        except Exception as e:
            logger.error(f"Error converting molecular format: {e}")
            return None


# Global instance for easy access
molecular_utils = MolecularUtils()
