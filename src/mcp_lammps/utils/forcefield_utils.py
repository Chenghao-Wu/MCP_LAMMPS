"""
hybridForce Field Utilities - Comprehensive GAFF parameter management and assignment.

This module provides comprehensive GAFF/GAFF2 force field functionality using
the complete parameter database extracted from gaff.lt files.
"""

import logging
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, rdPartialCharges
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Force field assignment functionality will be limited.")
    RDKIT_AVAILABLE = False


class ForceFieldUtils:
    """
    Comprehensive utilities for GAFF force field parameter management and assignment.
    
    This class provides functionality for:
    - Complete GAFF2 parameter assignment using comprehensive database
    - Advanced atom type assignment for organic molecules
    - Topology file generation with full parameter coverage
    - Force field parameter validation and optimization
    """
    
    def __init__(self, parameter_file: Optional[Union[str, Path]] = None):
        """
        Initialize the force field utilities.
        
        Args:
            parameter_file: Path to GAFF parameter JSON file. If None, uses default.
        """
        self.rdkit_available = RDKIT_AVAILABLE
        
        # Load comprehensive GAFF parameters
        if parameter_file is None:
            # Use default parameter file
            current_dir = Path(__file__).parent.parent
            parameter_file = current_dir / "data" / "gaff_parameters.json"
        
        self.parameter_file = Path(parameter_file)
        self.parameters = self._load_parameters()
        
        # Extract parameter sections for easy access
        self.atom_types = self.parameters.get("atom_types", {})
        self.atom_masses = self.parameters.get("atom_masses", {})
        self.atom_descriptions = self.parameters.get("atom_descriptions", {})
        self.pair_coeffs = self.parameters.get("pair_coefficients", {})
        self.bond_coeffs = self.parameters.get("bond_coefficients", {})
        self.angle_coeffs = self.parameters.get("angle_coefficients", {})
        self.dihedral_coeffs = self.parameters.get("dihedral_coefficients", {})
        self.improper_coeffs = self.parameters.get("improper_coefficients", {})
        self.ff_settings = self.parameters.get("force_field_settings", {})
        
        # Enhanced GAFF atom type patterns for comprehensive assignment
        self._initialize_gaff_patterns()
        
        logger.info(f"Loaded GAFF parameters: {self.parameters.get('statistics', {})}")
    
    def _load_parameters(self) -> Dict[str, Any]:
        """Load GAFF parameters from JSON file."""
        if not self.parameter_file.exists():
            logger.error(f"GAFF parameter file not found: {self.parameter_file}")
            return {}
        
        try:
            with open(self.parameter_file, 'r') as f:
                parameters = json.load(f)
            logger.debug(f"Loaded GAFF parameters from: {self.parameter_file}")
            return parameters
        except Exception as e:
            logger.error(f"Failed to load GAFF parameters: {e}")
            return {}
    
    def _initialize_gaff_patterns(self) -> None:
        """Initialize comprehensive GAFF atom type patterns ordered by specificity."""
        # Use OrderedDict to ensure specific patterns are matched first
        from collections import OrderedDict
        
        self.gaff_patterns = OrderedDict([
            # Most specific oxygen patterns first (critical for correct assignment)
            ("oh", "[OX2H1]"),  # hydroxyl oxygen - MUST come before general oxygen patterns
            ("os", "[OX2]([CX4,CX3])[CX4,CX3]"),  # ether oxygen
            ("op", "[OX1]=[PX4]"),  # oxygen in phosphate
            ("oq", "[OX1]=[PX4]"),  # oxygen in phosphate (identical to op)
            ("o", "[OX1]=[CX3,SX4,PX4]"),  # carbonyl oxygen
            
            # Most specific hydrogen patterns first (critical for correct assignment)
            ("ho", "[H][OX2]"),  # H on oxygen (hydroxyl) - MUST come after oh pattern
            ("hw", "[H][OX2H2]"),  # H in water
            ("hn", "[H][NX3,NX2]"),  # H on nitrogen
            ("hp", "[H][PX4]"),  # H on phosphorus
            ("hs", "[H][SX2]"),  # H on sulfur
            ("hx", "[H][CX4]([CX4,H])([CX4,H])[NX4+]"),  # H on C next to positively charged group
            ("h3", "[H][CX4]([OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])([OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1]"),  # H on C with 3 electronegative
            ("h2", "[H][CX4]([CX4,H])([OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1]"),  # H on C with 2 electronegative
            ("h1", "[H][CX4]([CX4,H])([CX4,H])[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1]"),  # H on C with 1 electronegative
            ("h4", "[H][CX3]=[CX3,NX2,OX1,SX1]"),  # H on non-sp3 carbon with 1 electronegative
            ("h5", "[H][CX3]"),  # H on non-sp3 carbon with 2 electronegative
            ("ha", "[H][cX3]"),  # H on aromatic carbon
            ("hc", "[H][CX4]"),  # H on aliphatic carbon without electronegative groups - most general
            
            # Specific carbon patterns first
            ("cz", "[CX3](=[NX2])[NX3]"),  # sp2 carbon in guanidine group
            ("cs", "[CX3](=[SX1])"),  # thiocarbonyl carbon
            ("c", "[CX3](=[OX1])"),  # carbonyl carbon
            ("cx", "[CX4]1[CX4][CX4]1"),  # sp3 carbon in triangle system
            ("cy", "[CX4]1[CX4][CX4][CX4]1"),  # sp3 carbon in square system
            ("cu", "[cX3]1:[cX3]:[cX3]:1"),  # sp2 carbon in triangle system
            ("cv", "[cX3]1:[cX3]:[cX3]:[cX3]:1"),  # sp2 carbon in square system
            ("cc", "[cX3]1:[cX3]:[nX2]:[cX3]:[cX3]:1"),  # aromatic carbon in 5-ring with N
            ("cd", "[cX3]1:[cX3]:[cX3]:[nX2]:[cX3]:1"),  # aromatic carbon in 5-ring with N
            ("cp", "[cX3]1:[cX3]:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # head sp2 carbon connecting rings
            ("cq", "[cX3]1:[cX3]:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # head sp2 carbon connecting rings
            ("ca", "[cX3]1:[cX3]:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # aromatic carbon
            ("ce", "[CX3]=[CX3]"),  # inner sp2 carbon in conjugated system
            ("cf", "[CX3]=[CX3]"),  # inner sp2 carbon in conjugated system (identical to ce)
            ("cg", "[CX2]#[CX2]"),  # inner sp carbon in conjugated system
            ("ch", "[CX2]#[CX2]"),  # inner sp carbon in conjugated system (identical to cg)
            ("c1", "[CX2]#[CX2,NX1]"),  # sp carbon in alkyne/nitrile
            ("c2", "[CX3]=[CX3,NX2,OX1,SX1]"),  # sp2 carbon in alkene/carbonyl
            ("c3", "[CX4]"),  # sp3 carbon - most general
            
            # Nitrogen atom types - specific to general
            ("n+", "[NX4+]"),  # positively charged nitrogen
            ("n4", "[NX4+]"),  # sp3 nitrogen with four connections (charged)
            ("no", "[NX3](=[OX1])[OX1-]"),  # nitro nitrogen
            ("nh", "[NX3H1,NX3H2]"),  # amine nitrogen
            ("na", "[nX2]1:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # sp2 nitrogen in aromatic ring with three connections
            ("nb", "[nX2]1:[cX3]:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # sp2 nitrogen in pure aromatic systems
            ("nc", "[nX2]1:[cX3]:[cX3]:[nX2]:[cX3]:1"),  # sp2 nitrogen in non-pure aromatic systems
            ("nd", "[nX2]1:[cX3]:[cX3]:[nX2]:[cX3]:1"),  # sp2 nitrogen in non-pure aromatic systems (identical to nc)
            ("ne", "[NX2]=[CX3][CX3]=[CX3]"),  # inner sp2 nitrogen in conjugated systems
            ("nf", "[NX2]=[CX3][CX3]=[CX3]"),  # inner sp2 nitrogen in conjugated systems (identical to ne)
            ("n1", "[NX1]#[CX2]"),  # sp nitrogen
            ("n2", "[NX2]=[CX3,CX2]"),  # aliphatic sp2 nitrogen with two connections
            ("n3", "[NX3]([CX4,H])([CX4,H])[CX4,H]"),  # sp3 nitrogen with three connections
            ("n", "[NX3]([CX3]=[OX1])"),  # sp2 nitrogen in amide
            
            # Sulfur atom types - specific to general
            ("s6", "[SX4](=[OX1])(=[OX1])"),  # sulfone sulfur
            ("s4", "[SX3](=[OX1])"),  # sulfoxide sulfur
            ("sh", "[SX2H1]"),  # thiol sulfur
            ("ss", "[SX2]([CX4,CX3])[CX4,CX3]"),  # disulfide sulfur
            ("sx", "[SX2]1[CX4][CX4]1"),  # sulfur in triangle system
            ("sy", "[SX2]1[CX4][CX4][CX4]1"),  # sulfur in square system
            ("s2", "[SX2]"),  # sulfur with two connections (identical to s)
            ("s", "[SX2]"),  # sulfur with two connections
            
            # Phosphorus atom types - specific to general
            ("p5", "[PX4](=[OX1])"),  # phosphorus in phosphate
            ("pb", "[pX3]1:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # aromatic phosphorus
            ("pc", "[pX3]1:[cX3]:[cX3]:[nX2]:[cX3]:1"),  # aromatic phosphorus in 5-ring
            ("pd", "[pX3]1:[cX3]:[cX3]:[nX2]:[cX3]:1"),  # aromatic phosphorus in 5-ring (identical to pc)
            ("pe", "[PX3]=[CX3]"),  # inner sp2 phosphorus in conjugated systems
            ("pf", "[PX3]=[CX3]"),  # inner sp2 phosphorus in conjugated systems (identical to pe)
            ("px", "[PX3]1[CX4][CX4]1"),  # phosphorus in triangle system
            ("py", "[PX3]1[CX4][CX4][CX4]1"),  # phosphorus in square system
            ("p2", "[PX2]"),  # phosphorus with two connections
            ("p3", "[PX3]"),  # phosphorus with three connections
            ("p4", "[PX4]"),  # phosphorus with four connections
            
            # Halogen atom types
            ("f", "[FX1]"),  # fluorine
            ("cl", "[ClX1]"),  # chlorine
            ("br", "[BrX1]"),  # bromine
            ("i", "[IX1]"),  # iodine
            
            # Metal atom types (basic)
            ("Li", "[Li+]"),  # lithium
            ("Na", "[Na+]"),  # sodium
            ("K", "[K+]"),   # potassium
            ("Rb", "[Rb+]"), # rubidium
            ("Cs", "[Cs+]"), # cesium
            ("Mg", "[Mg+2]"), # magnesium
            ("Ca", "[Ca+2]"), # calcium
            ("Zn", "[Zn+2]"), # zinc
            ("Cu", "[Cu+,Cu+2]"), # copper
        ])
    
    def assign_gaff_atom_types(self, mol: Chem.Mol) -> Dict[int, str]:
        """
        Assign GAFF atom types to a molecule using comprehensive pattern matching.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary mapping atom indices to GAFF atom types
        """
        if not self.rdkit_available or mol is None:
            logger.error("RDKit not available or invalid molecule")
            return {}
        
        try:
            atom_types = {}
            
            for atom_idx in range(mol.GetNumAtoms()):
                atom_type = self._get_gaff_atom_type(mol, atom_idx)
                atom_types[atom_idx] = atom_type
            
            logger.info(f"Assigned GAFF atom types to {len(atom_types)} atoms")
            return atom_types
            
        except Exception as e:
            logger.error(f"Error assigning GAFF atom types: {e}")
            return {}
    
    def _get_gaff_atom_type(self, mol: Chem.Mol, atom_idx: int) -> str:
        """
        Get GAFF atom type for a specific atom using comprehensive pattern matching.
        
        Args:
            mol: RDKit molecule object
            atom_idx: Index of the atom
            
        Returns:
            GAFF atom type string
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Try to match against comprehensive GAFF patterns with position and element validation
        for gaff_type, smarts_pattern in self.gaff_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts_pattern)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        # Check if atom_idx is in the match and at the correct position
                        if atom_idx in match:
                            atom_position = match.index(atom_idx)
                            
                            # Validate that the atom type is appropriate for this element and position
                            if self._validate_pattern_assignment(mol, atom_idx, gaff_type, smarts_pattern, atom_position):
                                return gaff_type
            except Exception as e:
                logger.debug(f"Pattern matching failed for {gaff_type}: {e}")
                continue
        
        # Enhanced fallback based on element, hybridization, and environment
        element = atom.GetSymbol()
        return element.lower()
    
    def _validate_pattern_assignment(self, mol: Chem.Mol, atom_idx: int, gaff_type: str, smarts_pattern: str, atom_position: int) -> bool:
        """
        Validate that a GAFF type assignment is appropriate for the specific atom.
        
        Args:
            mol: RDKit molecule object
            atom_idx: Index of the atom being assigned
            gaff_type: Proposed GAFF atom type
            smarts_pattern: SMARTS pattern that matched
            atom_position: Position of the atom in the SMARTS match
            
        Returns:
            True if the assignment is valid, False otherwise
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        element = atom.GetSymbol()
        
        # Define element requirements for each GAFF type category
        gaff_element_map = {
            # Carbon types
            'c': 'C', 'c1': 'C', 'c2': 'C', 'c3': 'C', 'ca': 'C', 'cc': 'C', 'cd': 'C',
            'ce': 'C', 'cf': 'C', 'cg': 'C', 'ch': 'C', 'cp': 'C', 'cq': 'C', 'cs': 'C',
            'cu': 'C', 'cv': 'C', 'cx': 'C', 'cy': 'C', 'cz': 'C',
            
            # Hydrogen types
            'h1': 'H', 'h2': 'H', 'h3': 'H', 'h4': 'H', 'h5': 'H', 'ha': 'H', 'hc': 'H',
            'hn': 'H', 'ho': 'H', 'hp': 'H', 'hs': 'H', 'hw': 'H', 'hx': 'H',
            
            # Oxygen types
            'o': 'O', 'oh': 'O', 'os': 'O', 'op': 'O', 'oq': 'O',
            
            # Nitrogen types
            'n': 'N', 'n1': 'N', 'n2': 'N', 'n3': 'N', 'n4': 'N', 'na': 'N', 'nb': 'N',
            'nc': 'N', 'nd': 'N', 'ne': 'N', 'nf': 'N', 'nh': 'N', 'no': 'N', 'n+': 'N',
            
            # Sulfur types
            's': 'S', 's2': 'S', 's4': 'S', 's6': 'S', 'sh': 'S', 'ss': 'S', 'sx': 'S', 'sy': 'S',
            
            # Phosphorus types
            'p2': 'P', 'p3': 'P', 'p4': 'P', 'p5': 'P', 'pb': 'P', 'pc': 'P', 'pd': 'P',
            'pe': 'P', 'pf': 'P', 'px': 'P', 'py': 'P',
            
            # Halogen types
            'f': 'F', 'cl': 'Cl', 'br': 'Br', 'i': 'I',
            
            # Metal types
            'Li': 'Li', 'Na': 'Na', 'K': 'K', 'Rb': 'Rb', 'Cs': 'Cs',
            'Mg': 'Mg', 'Ca': 'Ca', 'Zn': 'Zn', 'Cu': 'Cu'
        }
        
        # Check if the GAFF type is appropriate for this element
        expected_element = gaff_element_map.get(gaff_type)
        if expected_element and element != expected_element:
            return False
        
        # For SMARTS patterns with multiple atoms, validate position-specific requirements
        # This is crucial for patterns like [H][OX2] where H should be at position 0, O at position 1
        
        # Parse the SMARTS pattern to understand atom positions
        if '[H]' in smarts_pattern and gaff_type.startswith('h'):
            # Hydrogen types: the atom should be hydrogen and typically at position 0
            if element != 'H':
                return False
            # For hydrogen patterns, the hydrogen is usually the first atom in the pattern
            if atom_position != 0 and '[H]' == smarts_pattern.split('[')[1].split(']')[0]:
                return False
                
        elif '[O' in smarts_pattern and gaff_type.startswith('o'):
            # Oxygen types: the atom should be oxygen
            if element != 'O':
                return False
            # For oxygen patterns like [OX2H1], oxygen should be the matched atom
            if '[OX' in smarts_pattern and atom_position == 0:
                return True
            # For patterns like [H][OX2], oxygen should be at position 1
            elif smarts_pattern.startswith('[H][O'):
                return atom_position == 1
                
        elif '[C' in smarts_pattern and gaff_type.startswith('c'):
            # Carbon types: the atom should be carbon
            if element != 'C':
                return False
                
        elif '[N' in smarts_pattern and gaff_type.startswith('n'):
            # Nitrogen types: the atom should be nitrogen
            if element != 'N':
                return False
                
        elif '[S' in smarts_pattern and gaff_type.startswith('s'):
            # Sulfur types: the atom should be sulfur
            if element != 'S':
                return False
                
        elif '[P' in smarts_pattern and gaff_type.startswith('p'):
            # Phosphorus types: the atom should be phosphorus
            if element != 'P':
                return False
        
        # Additional validation for specific problematic patterns
        if gaff_type == 'ho' and element != 'H':
            # 'ho' type should only be assigned to hydrogen atoms
            return False
        elif gaff_type == 'oh' and element != 'O':
            # 'oh' type should only be assigned to oxygen atoms
            return False
        
        return True
        hybridization = atom.GetHybridization()
        formal_charge = atom.GetFormalCharge()
        is_aromatic = atom.GetIsAromatic()
        
        # Carbon atom type assignment
        if element == "C":
            if hybridization == Chem.HybridizationType.SP3:
                # Check for special environments
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if len([n for n in neighbors if n in ['O', 'N', 'S', 'F', 'Cl', 'Br', 'I']]) >= 3:
                    return "c3"  # Highly substituted
                elif len([n for n in neighbors if n in ['O', 'N', 'S', 'F', 'Cl', 'Br', 'I']]) >= 1:
                    return "c3"  # Substituted
                else:
                    return "c3"  # Standard sp3 carbon
            elif hybridization == Chem.HybridizationType.SP2:
                if is_aromatic:
                    return "ca"  # Aromatic carbon
                else:
                    # Check if it's carbonyl
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == "O" and neighbor.GetTotalNumHs() == 0:
                            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                            if bond.GetBondType() == Chem.BondType.DOUBLE:
                                return "c"  # Carbonyl carbon
                    return "c2"  # Alkene carbon
            elif hybridization == Chem.HybridizationType.SP:
                return "c1"  # Alkyne carbon
            else:
                return "c3"  # Default to sp3
        
        # Hydrogen atom type assignment
        elif element == "H":
            neighbors = atom.GetNeighbors()
            if neighbors:
                neighbor = neighbors[0]
                neighbor_element = neighbor.GetSymbol()
                
                if neighbor_element == "C":
                    if neighbor.GetIsAromatic():
                        return "ha"  # Aromatic hydrogen
                    elif neighbor.GetHybridization() == Chem.HybridizationType.SP3:
                        # Count electronegative neighbors of the carbon
                        carbon_neighbors = [n.GetSymbol() for n in neighbor.GetNeighbors() if n.GetIdx() != atom_idx]
                        electronegative_count = len([n for n in carbon_neighbors if n in ['O', 'N', 'S', 'F', 'Cl', 'Br', 'I']])
                        
                        if electronegative_count == 0:
                            return "hc"
                        elif electronegative_count == 1:
                            return "h1"
                        elif electronegative_count == 2:
                            return "h2"
                        else:
                            return "h3"
                    else:
                        return "h4"  # H on sp2/sp carbon
                elif neighbor_element == "O":
                    return "ho"  # Hydroxyl hydrogen
                elif neighbor_element == "N":
                    return "hn"  # Amine hydrogen
                elif neighbor_element == "S":
                    return "hs"  # Thiol hydrogen
                else:
                    return "hc"  # Default
            else:
                return "hc"  # Default
        
        # Oxygen atom type assignment
        elif element == "O":
            if atom.GetTotalNumHs() > 0:
                return "oh"  # Hydroxyl oxygen
            elif atom.GetDegree() == 1:
                return "o"   # Carbonyl oxygen
            elif atom.GetDegree() == 2:
                return "os"  # Ether oxygen
            else:
                return "o"   # Default
        
        # Nitrogen atom type assignment
        elif element == "N":
            if formal_charge > 0:
                return "n4" if atom.GetDegree() == 4 else "n+"
            elif hybridization == Chem.HybridizationType.SP3:
                return "n3"
            elif hybridization == Chem.HybridizationType.SP2:
                if is_aromatic:
                    return "nb" if atom.GetDegree() == 2 else "na"
                else:
                    return "n2"
            elif hybridization == Chem.HybridizationType.SP:
                return "n1"
            else:
                return "n3"  # Default
        
        # Sulfur atom type assignment
        elif element == "S":
            if atom.GetTotalNumHs() > 0:
                return "sh"  # Thiol sulfur
            elif atom.GetDegree() == 2:
                return "ss"  # Disulfide sulfur
            elif atom.GetDegree() == 3:
                return "s4"  # Sulfoxide
            elif atom.GetDegree() == 4:
                return "s6"  # Sulfone
            else:
                return "s"   # Default
        
        # Phosphorus atom type assignment
        elif element == "P":
            if atom.GetDegree() == 2:
                return "p2"
            elif atom.GetDegree() == 3:
                return "p3"
            elif atom.GetDegree() == 4:
                return "p4"
            else:
                return "p5"  # Default
        
        # Halogen atom types
        elif element == "F":
            return "f"
        elif element == "Cl":
            return "cl"
        elif element == "Br":
            return "br"
        elif element == "I":
            return "i"
        
        # Default fallback
        else:
            logger.warning(f"Unknown element {element}, using generic type")
            return element.lower()
    
    def get_pair_parameters(self, atom_type: str) -> Optional[Dict[str, float]]:
        """
        Get Lennard-Jones pair parameters for an atom type.
        
        Args:
            atom_type: GAFF atom type
            
        Returns:
            Dictionary with epsilon and sigma parameters, or None if not found
        """
        return self.pair_coeffs.get(atom_type)
    
    def get_bond_parameters(self, atom_type1: str, atom_type2: str) -> Optional[Dict[str, Any]]:
        """
        Get bond parameters for a pair of atom types.
        
        Args:
            atom_type1: First GAFF atom type
            atom_type2: Second GAFF atom type
            
        Returns:
            Dictionary with bond parameters, or None if not found
        """
        # Try both orders
        bond_key1 = f"{atom_type1}-{atom_type2}"
        bond_key2 = f"{atom_type2}-{atom_type1}"
        
        return self.bond_coeffs.get(bond_key1) or self.bond_coeffs.get(bond_key2)
    
    def get_angle_parameters(self, atom_type1: str, atom_type2: str, atom_type3: str) -> Optional[Dict[str, Any]]:
        """
        Get angle parameters for a triplet of atom types.
        
        Args:
            atom_type1: First GAFF atom type
            atom_type2: Central GAFF atom type
            atom_type3: Third GAFF atom type
            
        Returns:
            Dictionary with angle parameters, or None if not found
        """
        # Try both orders
        angle_key1 = f"{atom_type1}-{atom_type2}-{atom_type3}"
        angle_key2 = f"{atom_type3}-{atom_type2}-{atom_type1}"
        
        return self.angle_coeffs.get(angle_key1) or self.angle_coeffs.get(angle_key2)
    
    def get_dihedral_parameters(self, atom_types: Tuple[str, str, str, str]) -> Optional[List[Dict[str, Any]]]:
        """
        Get dihedral parameters for a quartet of atom types.

        Args:
            atom_types: Tuple of four GAFF atom types

        Returns:
            List of dictionaries with dihedral parameters, or None if not found
        """
        # Try both orders and wildcard patterns
        dihedral_key1 = "-".join(atom_types)
        dihedral_key2 = "-".join(reversed(atom_types))

        # Try exact matches first
        params = self.dihedral_coeffs.get(dihedral_key1) or self.dihedral_coeffs.get(dihedral_key2)
        if params:
            return [params]

        # Try wildcard patterns (X-type1-type2-X, etc.)
        wildcard_patterns = [
            f"X-{atom_types[1]}-{atom_types[2]}-X",
            f"X-{atom_types[2]}-{atom_types[1]}-X",
            f"{atom_types[0]}-{atom_types[1]}-{atom_types[2]}-X",
            f"X-{atom_types[1]}-{atom_types[2]}-{atom_types[3]}",
        ]

        for pattern in wildcard_patterns:
            params = self.dihedral_coeffs.get(pattern)
            if params:
                return [params]

        return None
    
    def generate_topology(self, mol: Chem.Mol, atom_types: Dict[int, str]) -> Dict[str, Any]:
        """
        Generate molecular topology with comprehensive parameter assignment.
        
        Args:
            mol: RDKit molecule object
            atom_types: Dictionary mapping atom indices to GAFF atom types
            
        Returns:
            Dictionary containing topology information
        """
        if not self.rdkit_available or mol is None:
            logger.error("RDKit not available or invalid molecule")
            return {}
        
        try:
            topology = {
                "atoms": [],
                "bonds": [],
                "angles": [],
                "dihedrals": [],
                "impropers": []
            }
            
            # Generate atom information
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                atom_type = atom_types.get(atom_idx, "unk")
                
                atom_info = {
                    "index": atom_idx,
                    "element": atom.GetSymbol(),
                    "atom_type": atom_type,
                    "mass": self.atom_masses.get(atom_type, atom.GetMass()),
                    "charge": 0.0  # Will be calculated separately
                }
                topology["atoms"].append(atom_info)
            
            # Generate bonds
            for bond in mol.GetBonds():
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                atom1_type = atom_types.get(atom1_idx, "unk")
                atom2_type = atom_types.get(atom2_idx, "unk")
                
                bond_params = self.get_bond_parameters(atom1_type, atom2_type)
                
                bond_info = {
                    "atoms": [atom1_idx, atom2_idx],
                    "types": [atom1_type, atom2_type],
                    "bond_type": bond.GetBondType(),
                    "parameters": bond_params
                }
                topology["bonds"].append(bond_info)
            
            # Generate angles
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                neighbors = list(atom.GetNeighbors())
                
                # Generate all angle combinations with this atom as center
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        atom1_idx = neighbors[i].GetIdx()
                        atom2_idx = atom_idx  # Center atom
                        atom3_idx = neighbors[j].GetIdx()
                        
                        atom1_type = atom_types.get(atom1_idx, "unk")
                        atom2_type = atom_types.get(atom2_idx, "unk")
                        atom3_type = atom_types.get(atom3_idx, "unk")
                        
                        angle_params = self.get_angle_parameters(atom1_type, atom2_type, atom3_type)
                        
                        angle_info = {
                            "atoms": [atom1_idx, atom2_idx, atom3_idx],
                            "types": [atom1_type, atom2_type, atom3_type],
                            "parameters": angle_params
                        }
                        topology["angles"].append(angle_info)
            
            # Generate dihedrals (torsions)
            for bond in mol.GetBonds():
                atom2_idx = bond.GetBeginAtomIdx()
                atom3_idx = bond.GetEndAtomIdx()
                
                atom2 = mol.GetAtomWithIdx(atom2_idx)
                atom3 = mol.GetAtomWithIdx(atom3_idx)
                
                # Get neighbors for dihedral generation
                atom2_neighbors = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != atom3_idx]
                atom3_neighbors = [n.GetIdx() for n in atom3.GetNeighbors() if n.GetIdx() != atom2_idx]
                
                # Generate all dihedral combinations
                for atom1_idx in atom2_neighbors:
                    for atom4_idx in atom3_neighbors:
                        atom1_type = atom_types.get(atom1_idx, "unk")
                        atom2_type = atom_types.get(atom2_idx, "unk")
                        atom3_type = atom_types.get(atom3_idx, "unk")
                        atom4_type = atom_types.get(atom4_idx, "unk")
                        
                        dihedral_params = self.get_dihedral_parameters((atom1_type, atom2_type, atom3_type, atom4_type))
                        
                        dihedral_info = {
                            "atoms": [atom1_idx, atom2_idx, atom3_idx, atom4_idx],
                            "types": [atom1_type, atom2_type, atom3_type, atom4_type],
                            "parameters": dihedral_params
                        }
                        topology["dihedrals"].append(dihedral_info)
            
            logger.info(f"Generated topology: {len(topology['atoms'])} atoms, "
                       f"{len(topology['bonds'])} bonds, {len(topology['angles'])} angles, "
                       f"{len(topology['dihedrals'])} dihedrals")
            
            return topology
            
        except Exception as e:
            logger.error(f"Error generating topology: {e}")
            return {}
    
    def calculate_partial_charges(self, mol: Chem.Mol, method: str = "gasteiger") -> Dict[int, float]:
        """
        Calculate partial charges for atoms in a molecule.
        
        Args:
            mol: RDKit molecule object
            method: Charge calculation method ('gasteiger', 'mmff94', 'formal')
            
        Returns:
            Dictionary mapping atom indices to partial charges
        """
        if not self.rdkit_available or mol is None:
            logger.error("RDKit not available or invalid molecule")
            return {}
        
        try:
            charges = {}
            
            if method.lower() == "gasteiger":
                # Use Gasteiger charges
                rdPartialCharges.ComputeGasteigerCharges(mol)
                for atom_idx in range(mol.GetNumAtoms()):
                    atom = mol.GetAtomWithIdx(atom_idx)
                    charges[atom_idx] = float(atom.GetProp('_GasteigerCharge'))
            
            elif method.lower() == "mmff94":
                # Use MMFF94 charges
                try:
                    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                    if mmff_props:
                        for atom_idx in range(mol.GetNumAtoms()):
                            charges[atom_idx] = mmff_props.GetMMFFPartialCharge(atom_idx)
                    else:
                        logger.warning("MMFF94 properties not available, falling back to Gasteiger")
                        return self.calculate_partial_charges(mol, "gasteiger")
                except Exception as e:
                    logger.warning(f"MMFF94 charge calculation failed: {e}, falling back to Gasteiger")
                    return self.calculate_partial_charges(mol, "gasteiger")
            
            elif method.lower() == "formal":
                # Use formal charges
                for atom_idx in range(mol.GetNumAtoms()):
                    atom = mol.GetAtomWithIdx(atom_idx)
                    charges[atom_idx] = float(atom.GetFormalCharge())
            
            else:
                logger.error(f"Unknown charge method: {method}")
                return {}
            
            logger.info(f"Calculated {method} charges for {len(charges)} atoms")
            return charges
            
        except Exception as e:
            logger.error(f"Error calculating partial charges: {e}")
            return {}
    
    def validate_parameters(self, atom_types: Dict[int, str], topology: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that all required force field parameters are available.
        
        Args:
            atom_types: Dictionary mapping atom indices to GAFF atom types
            topology: Topology information
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check atom type parameters
        for atom_idx, atom_type in atom_types.items():
            if atom_type not in self.pair_coeffs:
                issues.append(f"Missing pair parameters for atom type: {atom_type}")
            if atom_type not in self.atom_masses:
                issues.append(f"Missing mass for atom type: {atom_type}")
        
        # Check bond parameters
        for bond in topology.get("bonds", []):
            if bond.get("parameters") is None:
                bond_types = bond.get("types", [])
                if len(bond_types) == 2:
                    issues.append(f"Missing bond parameters for: {bond_types[0]}-{bond_types[1]}")
        
        # Check angle parameters
        for angle in topology.get("angles", []):
            if angle.get("parameters") is None:
                angle_types = angle.get("types", [])
                if len(angle_types) == 3:
                    issues.append(f"Missing angle parameters for: {'-'.join(angle_types)}")
        
        # Check dihedral parameters
        missing_dihedrals = 0
        for dihedral in topology.get("dihedrals", []):
            if dihedral.get("parameters") is None:
                missing_dihedrals += 1
        
        if missing_dihedrals > 0:
            issues.append(f"Missing dihedral parameters for {missing_dihedrals} dihedrals")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Parameter validation failed with {len(issues)} issues")
        else:
            logger.info("All parameters validated successfully")
        
        return is_valid, issues

    def get_gaff_charge_estimates(self, mol: Chem.Mol, atom_types: Dict[int, str]) -> Dict[int, float]:
        """
        Get GAFF charge estimates for atoms based on atom types.

        Args:
            mol: RDKit molecule object
            atom_types: Dictionary mapping atom indices to GAFF atom types

        Returns:
            Dictionary mapping atom indices to estimated charges
        """
        try:
            charges = {}

            # GAFF charge estimates based on atom types (rough estimates)
            charge_estimates = {
                'c': 0.0, 'c1': 0.0, 'c2': 0.0, 'c3': 0.0,
                'ca': 0.0, 'cb': 0.0, 'cc': 0.0, 'cd': 0.0, 'ce': 0.0, 'cf': 0.0,
                'cg': 0.0, 'ch': 0.0, 'ci': 0.0, 'cj': 0.0, 'ck': 0.0, 'cl': 0.0,
                'cm': 0.0, 'cn': 0.0, 'co': 0.0, 'cp': 0.0, 'cq': 0.0, 'cr': 0.0,
                'cs': 0.0, 'ct': 0.0, 'cu': 0.0, 'cv': 0.0, 'cw': 0.0, 'cx': 0.0,
                'cy': 0.0, 'cz': 0.0,
                'h': 0.0, 'h1': 0.0, 'h2': 0.0, 'h3': 0.0, 'h4': 0.0, 'h5': 0.0,
                'ha': 0.0, 'hc': 0.0, 'hn': 0.0, 'ho': 0.0, 'hp': 0.0, 'hs': 0.0,
                'hw': 0.0,
                'n': 0.0, 'n1': 0.0, 'n2': 0.0, 'n3': 0.0, 'n4': 0.0,
                'na': 0.0, 'nb': 0.0, 'nc': 0.0, 'nd': 0.0, 'ne': 0.0, 'nf': 0.0,
                'nh': 0.0, 'ni': 0.0, 'nj': 0.0, 'nk': 0.0, 'nl': 0.0, 'nm': 0.0,
                'nn': 0.0, 'no': 0.0, 'np': 0.0,
                'o': 0.0, 'o1': 0.0, 'o2': 0.0, 'oh': 0.0, 'os': 0.0,
                'p': 0.0, 'p2': 0.0, 'p3': 0.0, 'p4': 0.0, 'p5': 0.0,
                'pb': 0.0, 'pc': 0.0, 'pd': 0.0, 'pe': 0.0, 'pf': 0.0,
                'px': 0.0, 'py': 0.0,
                's': 0.0, 's2': 0.0, 's4': 0.0, 's6': 0.0, 'sh': 0.0, 'ss': 0.0, 'sx': 0.0,
                'f': 0.0, 'cl': 0.0, 'br': 0.0, 'i': 0.0
            }

            for atom_idx in range(mol.GetNumAtoms()):
                atom_type = atom_types.get(atom_idx, 'c')
                charges[atom_idx] = charge_estimates.get(atom_type, 0.0)

            return charges

        except Exception as e:
            logger.error(f"Failed to get GAFF charge estimates: {e}")
            return {}

    def validate_charges(self, charges: Dict[int, float], mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """
        Validate partial charges for a molecule.

        Args:
            charges: Dictionary mapping atom indices to charges
            mol: RDKit molecule object

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            num_atoms = mol.GetNumAtoms()

            # Check that all atoms have charges
            for atom_idx in range(num_atoms):
                if atom_idx not in charges:
                    issues.append(f"Missing charge for atom {atom_idx}")

            # Check charge conservation (should be close to zero for neutral molecules)
            total_charge = sum(charges.values())
            if abs(total_charge) > 0.1:  # Allow some tolerance
                issues.append(f"Total charge ({total_charge:.3f}) far from neutral")

            # Check for unreasonable charge values
            for atom_idx, charge in charges.items():
                if abs(charge) > 2.0:  # Unreasonably large charge
                    issues.append(f"Atom {atom_idx} has unreasonable charge: {charge}")

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Validation failed: {e}")
            return False, issues

    def get_force_field_settings(self) -> Dict[str, str]:
        """
        Get LAMMPS force field settings and styles.
        
        Returns:
            Dictionary of force field settings
        """
        return self.ff_settings.copy()
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about loaded parameters.

        Returns:
            Dictionary with parameter counts
        """
        return self.parameters.get("statistics", {})

    def create_lammps_data_file(
        self,
        mol: Chem.Mol,
        atom_types: Dict[int, str],
        topology: Dict[str, Any],
        charges: Optional[Dict[int, float]] = None
    ) -> str:
        """
        Create LAMMPS data file content from molecular structure and topology.

        Args:
            mol: RDKit molecule object
            atom_types: Dictionary mapping atom indices to GAFF atom types
            topology: Molecular topology dictionary
            charges: Dictionary mapping atom indices to partial charges

        Returns:
            LAMMPS data file content as string
        """
        try:
            import numpy as np

            # Get molecular properties
            from .molecular_utils import molecular_utils
            mol_props = molecular_utils.calculate_molecular_properties(mol)
            num_atoms = mol.GetNumAtoms()

            # Get unique atom types
            unique_atom_types = list(set(atom_types.values()))
            unique_atom_types.sort()

            # Create atom type mapping
            atom_type_map = {atype: i+1 for i, atype in enumerate(unique_atom_types)}

            # Create unique bond, angle, dihedral types based on parameters
            bond_types = {}
            angle_types = {}
            dihedral_types = {}

            bond_type_counter = 1
            for bond in topology['bonds']:
                param_key = str(sorted(bond['parameters'].items())) if bond.get('parameters') else str(bond['types'])
                if param_key not in bond_types:
                    bond_types[param_key] = bond_type_counter
                    bond_type_counter += 1
                bond['type'] = bond_types[param_key]

            angle_type_counter = 1
            for angle in topology['angles']:
                param_key = str(sorted(angle['parameters'].items())) if angle.get('parameters') else str(angle['types'])
                if param_key not in angle_types:
                    angle_types[param_key] = angle_type_counter
                    angle_type_counter += 1
                angle['type'] = angle_types[param_key]

            dihedral_type_counter = 1
            for dihedral in topology['dihedrals']:
                # Handle both dict and list parameter formats
                if dihedral.get('parameters'):
                    if isinstance(dihedral['parameters'], list):
                        # Parameters is a list of dicts
                        param_key = str(sorted(dihedral['parameters'][0].items())) if dihedral['parameters'] else str(dihedral['types'])
                    else:
                        # Parameters is a dict
                        param_key = str(sorted(dihedral['parameters'].items()))
                else:
                    param_key = str(dihedral['types'])

                if param_key not in dihedral_types:
                    dihedral_types[param_key] = dihedral_type_counter
                    dihedral_type_counter += 1
                dihedral['type'] = dihedral_types[param_key]

            # Prepare data sections
            lines = []

            # Header
            lines.append(f"# LAMMPS data file generated from RDKit molecule")
            lines.append(f"# Molecular weight: {mol_props.get('molecular_weight', 0.0):.2f} g/mol")
            lines.append("")

            lines.append(f"{num_atoms} atoms")
            lines.append(f"{len(topology['bonds'])} bonds")
            lines.append(f"{len(topology['angles'])} angles")
            lines.append(f"{len(topology['dihedrals'])} dihedrals")
            lines.append("")

            lines.append(f"{len(unique_atom_types)} atom types")
            lines.append(f"{len(bond_types)} bond types")
            lines.append(f"{len(angle_types)} angle types")
            lines.append(f"{len(dihedral_types)} dihedral types")
            lines.append("")

            # Box dimensions (centered around origin, reasonable size)
            box_size = 20.0  # Angstroms
            half_box = box_size / 2.0
            lines.append(f"{-half_box} {half_box} xlo xhi")
            lines.append(f"{-half_box} {half_box} ylo yhi")
            lines.append(f"{-half_box} {half_box} zlo zhi")
            lines.append("")

            # Masses section
            lines.append("Masses")
            lines.append("")
            for atype, type_id in atom_type_map.items():
                # Get atomic mass from GAFF parameters or periodic table
                mass = self._get_atomic_mass_from_gaff(atype)
                lines.append(f"{type_id} {mass}")
            lines.append("")

            # Atoms section
            lines.append("Atoms")
            lines.append("")
            conformer = mol.GetConformer()
            for i in range(num_atoms):
                atom = mol.GetAtomWithIdx(i)
                pos = conformer.GetAtomPosition(i)
                atom_type_str = atom_types[i]
                atom_type_id = atom_type_map[atom_type_str]
                charge = charges.get(i, 0.0) if charges else 0.0

                lines.append(f"{i+1} 1 {atom_type_id} {charge} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
            lines.append("")

            # Bonds section
            if topology['bonds']:
                lines.append("Bonds")
                lines.append("")
                for i, bond in enumerate(topology['bonds'], 1):
                    lines.append(f"{i} {bond['type']} {bond['atoms'][0]+1} {bond['atoms'][1]+1}")
                lines.append("")

            # Angles section
            if topology['angles']:
                lines.append("Angles")
                lines.append("")
                for i, angle in enumerate(topology['angles'], 1):
                    lines.append(f"{i} {angle['type']} {angle['atoms'][0]+1} {angle['atoms'][1]+1} {angle['atoms'][2]+1}")
                lines.append("")

            # Dihedrals section
            if topology['dihedrals']:
                lines.append("Dihedrals")
                lines.append("")
                for i, dihedral in enumerate(topology['dihedrals'], 1):
                    lines.append(f"{i} {dihedral['type']} {dihedral['atoms'][0]+1} {dihedral['atoms'][1]+1} {dihedral['atoms'][2]+1} {dihedral['atoms'][3]+1}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Failed to create LAMMPS data file: {e}")
            return ""

    def _get_atomic_mass_from_gaff(self, gaff_type: str) -> float:
        """
        Get atomic mass from GAFF atom type.

        Args:
            gaff_type: GAFF atom type string

        Returns:
            Atomic mass in g/mol
        """
        # Simple mapping based on common GAFF types
        mass_map = {
            'c': 12.01, 'c1': 12.01, 'c2': 12.01, 'c3': 12.01,
            'h': 1.008, 'h1': 1.008, 'h2': 1.008, 'h3': 1.008,
            'o': 16.00, 'o1': 16.00, 'o2': 16.00,
            'n': 14.01, 'n1': 14.01, 'n2': 14.01, 'n3': 14.01,
            's': 32.06, 'p': 30.97, 'f': 19.00, 'cl': 35.45, 'br': 79.90, 'i': 126.90
        }

        # Extract element from GAFF type (first character, lowercase)
        element = gaff_type[0].lower()

        return mass_map.get(gaff_type, mass_map.get(element, 12.01))

    def _matches_wildcard_pattern(self, dihedral_type: Union[str, Tuple[str, ...]], pattern: Union[str, Tuple[str, ...]]) -> bool:
        """
        Check if a dihedral type matches a wildcard pattern.

        Args:
            dihedral_type: The actual dihedral type (e.g., "c3-c3-c3-hc" or ("c3", "c3", "c3", "c3"))
            pattern: Pattern with wildcards (e.g., "X-c3-c3-X" or ("X", "c3", "c3", "X"))

        Returns:
            True if the pattern matches
        """
        # Convert to lists if they're tuples or strings
        if isinstance(dihedral_type, str):
            type_parts = dihedral_type.split('-')
        else:
            type_parts = list(dihedral_type)

        if isinstance(pattern, str):
            pattern_parts = pattern.split('-')
        else:
            pattern_parts = list(pattern)

        if len(pattern_parts) != len(type_parts):
            return False

        for pattern_part, type_part in zip(pattern_parts, type_parts):
            if pattern_part != 'X' and pattern_part != type_part:
                return False

        return True

    def _get_atomic_mass(self, atom_type: str) -> float:
        """
        Get atomic mass for a GAFF atom type.

        Args:
            atom_type: GAFF atom type string

        Returns:
            Atomic mass in g/mol
        """
        return self._get_atomic_mass_from_gaff(atom_type)


# Create a global instance for easy access
forcefield_utils = ForceFieldUtils()