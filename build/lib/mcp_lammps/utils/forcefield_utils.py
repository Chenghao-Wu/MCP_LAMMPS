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
            # CRITICAL ORDER: hw (water) must come before ho (hydroxyl) to avoid misassignment
            ("hw", "[H][OX2H2]"),  # H in water - most specific, match first
            ("ho", "[H][OX2H1]"),  # H on hydroxyl oxygen - match OX2H1 specifically
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
            ("ca", "[cX3]1:[cX3]:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # aromatic carbon
            ("cp", "[cX3]1:[cX3]:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # head sp2 carbon connecting rings
            ("cq", "[cX3]1:[cX3]:[cX3]:[cX3]:[cX3]:[cX3]:1"),  # head sp2 carbon connecting rings
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
            fallback_count = 0
            element_counts = {}
            
            for atom_idx in range(mol.GetNumAtoms()):
                atom_type = self._get_gaff_atom_type(mol, atom_idx)
                atom_types[atom_idx] = atom_type
                
                # Track statistics
                atom = mol.GetAtomWithIdx(atom_idx)
                element = atom.GetSymbol()
                
                # Check if this is a fallback assignment (generic element type)
                if atom_type == element.lower():
                    fallback_count += 1
                
                # Count atom types by element
                if element not in element_counts:
                    element_counts[element] = []
                element_counts[element].append(atom_type)
            
            # Enhanced logging with statistics
            logger.info(f"Assigned GAFF atom types to {len(atom_types)} atoms")
            
            # Log type distribution by element
            for element, types in element_counts.items():
                unique_types = set(types)
                logger.debug(f"  {element}: {len(types)} atoms with {len(unique_types)} type(s): {', '.join(sorted(unique_types))}")
            
            # Warn if many fallbacks were used
            if fallback_count > 0:
                fallback_pct = (fallback_count / len(atom_types)) * 100
                logger.warning(
                    f"Atom type assignment used fallback for {fallback_count}/{len(atom_types)} atoms ({fallback_pct:.1f}%).\n"
                    f"This may indicate:\n"
                    f"  - Unusual chemical structures not in GAFF patterns\n"
                    f"  - Missing or incomplete GAFF atom type definitions\n"
                    f"  - Metal atoms or non-standard elements\n"
                    f"Recommendation: Review fallback assignments in detailed logs (set log level to DEBUG)"
                )
            
            return atom_types
            
        except Exception as e:
            logger.error(
                f"Critical error in GAFF atom type assignment:\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Molecule info: {mol.GetNumAtoms()} atoms\n"
                f"  Recommendation: Check molecule structure and RDKit availability"
            )
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
        element = atom.GetSymbol()
        hybridization = atom.GetHybridization()
        
        # Track attempted patterns for debugging
        attempted_patterns = []
        matched_but_rejected = []
        
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
                            attempted_patterns.append(gaff_type)
                            
                            # Validate that the atom type is appropriate for this element and position
                            if self._validate_pattern_assignment(mol, atom_idx, gaff_type, smarts_pattern, atom_position):
                                logger.debug(f"Assigned atom {atom_idx} ({element}) type '{gaff_type}' using pattern {smarts_pattern}")
                                return gaff_type
                            else:
                                matched_but_rejected.append((gaff_type, smarts_pattern))
            except Exception as e:
                logger.debug(f"Pattern matching exception for {gaff_type} on atom {atom_idx}: {e}")
                continue
        
        # Enhanced fallback based on element, hybridization, and environment
        fallback_type = element.lower()
        
        # Log detailed fallback information
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
        neighbor_info = f"{len(neighbors)} neighbors: {', '.join(neighbors) if neighbors else 'none'}"
        
        logger.warning(
            f"Atom type assignment fallback for atom {atom_idx}:\n"
            f"  Element: {element}\n"
            f"  Hybridization: {hybridization}\n"
            f"  Environment: {neighbor_info}\n"
            f"  Attempted {len(attempted_patterns)} GAFF patterns\n"
            f"  Matched but rejected: {len(matched_but_rejected)} patterns\n"
            f"  Using fallback type: '{fallback_type}'\n"
            f"  To debug: check if this atom type is unusual or if GAFF patterns need expansion"
        )
        
        # Log some of the rejected patterns for debugging
        if matched_but_rejected:
            for gaff_type, pattern in matched_but_rejected[:3]:  # Show first 3
                logger.debug(f"  Rejected: {gaff_type} (pattern: {pattern})")
        
        return fallback_type
    
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
        elif gaff_type == 'hw' and element != 'H':
            # 'hw' type should only be assigned to hydrogen atoms
            return False
        elif gaff_type == 'oh' and element != 'O':
            # 'oh' type should only be assigned to oxygen atoms
            return False
        
        # Additional validation for hw vs ho distinction
        if gaff_type == 'hw' and element == 'H':
            # hw should only match H in water (H-O-H where O has exactly 2 H)
            # The pattern [H][OX2H2] ensures this, but double-check
            neighbors = atom.GetNeighbors()
            if neighbors:
                neighbor = neighbors[0]
                if neighbor.GetSymbol() == 'O':
                    # Check if oxygen has exactly 2 hydrogens
                    h_count = sum(1 for n in neighbor.GetNeighbors() if n.GetSymbol() == 'H')
                    if h_count != 2:
                        return False
        elif gaff_type == 'ho' and element == 'H':
            # ho should match H on hydroxyl (H-O-R where O has 1 H)
            neighbors = atom.GetNeighbors()
            if neighbors:
                neighbor = neighbors[0]
                if neighbor.GetSymbol() == 'O':
                    # Check if oxygen has exactly 1 hydrogen
                    h_count = sum(1 for n in neighbor.GetNeighbors() if n.GetSymbol() == 'H')
                    if h_count != 1:
                        return False
        
        return True
    
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
        Get dihedral parameters for a quartet of atom types with comprehensive fallback.

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

        # Try wildcard patterns (X-type1-type2-X, etc.) - most specific first
        wildcard_patterns = [
            f"{atom_types[0]}-{atom_types[1]}-{atom_types[2]}-X",
            f"X-{atom_types[1]}-{atom_types[2]}-{atom_types[3]}",
            f"X-{atom_types[2]}-{atom_types[1]}-X",  # Reversed center atoms
            f"{atom_types[3]}-{atom_types[2]}-{atom_types[1]}-X",
            f"X-{atom_types[1]}-{atom_types[2]}-X",
        ]

        for pattern in wildcard_patterns:
            params = self.dihedral_coeffs.get(pattern)
            if params:
                return [params]
        
        # Try generic patterns based on central atom types
        generic_patterns = self._get_generic_dihedral_patterns(atom_types)
        for pattern in generic_patterns:
            params = self.dihedral_coeffs.get(pattern)
            if params:
                logger.debug(f"Using generic dihedral pattern {pattern} for {dihedral_key1}")
                return [params]
        
        # Ultimate fallback: use generic parameters based on bond types
        fallback_params = self._get_fallback_dihedral_parameters(atom_types)
        if fallback_params:
            logger.debug(f"Using fallback dihedral parameters for {dihedral_key1}")
            return [fallback_params]

        return None
    
    def _get_generic_dihedral_patterns(self, atom_types: Tuple[str, str, str, str]) -> List[str]:
        """
        Generate generic dihedral patterns for fallback matching.
        
        Args:
            atom_types: Tuple of four GAFF atom types
            
        Returns:
            List of generic pattern strings to try
        """
        patterns = []
        
        # Get element symbols from atom types (first character)
        elements = [atype[0] for atype in atom_types]
        
        # Try patterns with element-based wildcards
        # e.g., c3-c3-c3-c3 can fall back to X-c-c-X (generic carbon-carbon)
        center1 = atom_types[1]
        center2 = atom_types[2]
        
        # Try with generic element types
        if center1[0] == center2[0]:  # Same element type in center
            elem = center1[0]
            patterns.append(f"X-{elem}-{elem}-X")
        
        # Try patterns based on hybridization
        # c3 (sp3), c2 (sp2), c1 (sp), ca (aromatic) -> try generic 'c'
        center1_generic = self._get_generic_type(center1)
        center2_generic = self._get_generic_type(center2)
        
        if center1_generic != center1 or center2_generic != center2:
            patterns.append(f"X-{center1_generic}-{center2_generic}-X")
        
        return patterns
    
    def _get_generic_type(self, atom_type: str) -> str:
        """
        Get a more generic version of an atom type for fallback matching.
        
        Args:
            atom_type: Specific GAFF atom type (e.g., 'c3', 'ca', 'oh')
            
        Returns:
            Generic atom type (e.g., 'c', 'o', 'h')
        """
        # Map specific types to generic types
        generic_map = {
            # Carbon types -> generic 'c'
            'c1': 'c', 'c2': 'c', 'c3': 'c', 'ca': 'c', 'cc': 'c', 'cd': 'c',
            'ce': 'c', 'cf': 'c', 'cg': 'c', 'ch': 'c', 'cp': 'c', 'cq': 'c',
            'cs': 'c', 'cu': 'c', 'cv': 'c', 'cx': 'c', 'cy': 'c', 'cz': 'c',
            
            # Hydrogen types -> generic 'h'
            'h1': 'h', 'h2': 'h', 'h3': 'h', 'h4': 'h', 'h5': 'h', 'ha': 'h',
            'hc': 'h', 'hn': 'h', 'ho': 'h', 'hp': 'h', 'hs': 'h', 'hw': 'h', 'hx': 'h',
            
            # Oxygen types -> generic 'o'
            'oh': 'o', 'os': 'o', 'op': 'o', 'oq': 'o',
            
            # Nitrogen types -> generic 'n'
            'n1': 'n', 'n2': 'n', 'n3': 'n', 'n4': 'n', 'na': 'n', 'nb': 'n',
            'nc': 'n', 'nd': 'n', 'ne': 'n', 'nf': 'n', 'nh': 'n', 'no': 'n', 'n+': 'n',
            
            # Sulfur types -> generic 's'
            's2': 's', 's4': 's', 's6': 's', 'sh': 's', 'ss': 's', 'sx': 's', 'sy': 's',
            
            # Phosphorus types -> generic 'p'
            'p2': 'p', 'p3': 'p', 'p4': 'p', 'p5': 'p', 'pb': 'p', 'pc': 'p',
            'pd': 'p', 'pe': 'p', 'pf': 'p', 'px': 'p', 'py': 'p',
        }
        
        return generic_map.get(atom_type, atom_type)
    
    def _get_fallback_dihedral_parameters(self, atom_types: Tuple[str, str, str, str]) -> Optional[Dict[str, Any]]:
        """
        Provide reasonable fallback dihedral parameters based on chemical intuition.
        
        Args:
            atom_types: Tuple of four GAFF atom types
            
        Returns:
            Dictionary with fallback dihedral parameters, or None
        """
        # Extract element types
        elements = [atype[0] for atype in atom_types]
        center_elements = (elements[1], elements[2])
        
        # Define generic fallback parameters for common dihedral types
        # These are conservative parameters that allow some rotation
        fallback_params = {
            ('c', 'c'): {'style': 'fourier', 'n_terms': 1, 'k': 0.1556, 'n': 3, 'phase': 0.0},  # C-C rotation (like GAFF)
            ('c', 'n'): {'style': 'fourier', 'n_terms': 1, 'k': 0.0, 'n': 2, 'phase': 180.0},  # C-N partial double bond
            ('c', 'o'): {'style': 'fourier', 'n_terms': 1, 'k': 0.0, 'n': 2, 'phase': 180.0},  # C-O ester/ether
            ('n', 'c'): {'style': 'fourier', 'n_terms': 1, 'k': 0.0, 'n': 2, 'phase': 180.0},  # N-C
            ('o', 'c'): {'style': 'fourier', 'n_terms': 1, 'k': 0.0, 'n': 2, 'phase': 180.0},  # O-C
            ('c', 's'): {'style': 'fourier', 'n_terms': 1, 'k': 0.1556, 'n': 3, 'phase': 0.0},  # C-S similar to C-C
            ('s', 'c'): {'style': 'fourier', 'n_terms': 1, 'k': 0.1556, 'n': 3, 'phase': 0.0},  # S-C
            ('c', 'p'): {'style': 'fourier', 'n_terms': 1, 'k': 0.1556, 'n': 3, 'phase': 0.0},  # C-P
            ('p', 'c'): {'style': 'fourier', 'n_terms': 1, 'k': 0.1556, 'n': 3, 'phase': 0.0},  # P-C
        }
        
        # Try to find matching fallback parameters
        params = fallback_params.get(center_elements)
        if params:
            return params
        
        # Try reversed
        params = fallback_params.get((center_elements[1], center_elements[0]))
        if params:
            return params
        
        # Ultimate fallback: very weak dihedral (allows rotation)
        logger.warning(f"Using ultimate fallback dihedral for {'-'.join(atom_types)}")
        return {'style': 'fourier', 'n_terms': 1, 'k': 0.05, 'n': 3, 'phase': 0.0}
    
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
            
            # Generate improper dihedrals for sp2 centers and aromatic systems
            impropers = self._generate_improper_dihedrals(mol, atom_types)
            topology["impropers"] = impropers
            
            logger.info(f"Generated topology: {len(topology['atoms'])} atoms, "
                       f"{len(topology['bonds'])} bonds, {len(topology['angles'])} angles, "
                       f"{len(topology['dihedrals'])} dihedrals, {len(topology['impropers'])} impropers")
            
            return topology
            
        except Exception as e:
            logger.error(f"Error generating topology: {e}")
            return {}
    
    def _generate_improper_dihedrals(self, mol: Chem.Mol, atom_types: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Generate improper dihedrals for sp2 centers and aromatic systems.
        
        Improper dihedrals maintain planarity in:
        - Aromatic rings
        - Carbonyl groups
        - sp2 carbons (alkenes, imines, etc.)
        - Amide bonds
        
        Args:
            mol: RDKit molecule object
            atom_types: Dictionary mapping atom indices to GAFF atom types
            
        Returns:
            List of improper dihedral dictionaries
        """
        impropers = []
        
        if not self.rdkit_available or mol is None:
            return impropers
        
        try:
            # Iterate through all atoms to find sp2 centers
            for center_idx in range(mol.GetNumAtoms()):
                center_atom = mol.GetAtomWithIdx(center_idx)
                center_type = atom_types.get(center_idx, "unk")
                
                # Check if this atom needs improper dihedrals
                # sp2 carbons, aromatic atoms, sp2 nitrogens, etc.
                needs_improper = False
                
                # Check hybridization
                if center_atom.GetHybridization() == Chem.HybridizationType.SP2:
                    needs_improper = True
                elif center_atom.GetIsAromatic():
                    needs_improper = True
                # Check for specific GAFF types that need impropers
                elif center_type in ['c', 'c2', 'ca', 'cc', 'cd', 'ce', 'cf', 'n', 'n2', 'na', 'nb', 'nc', 'nd']:
                    needs_improper = True
                
                if not needs_improper:
                    continue
                
                # Get neighbors
                neighbors = list(center_atom.GetNeighbors())
                
                # Need exactly 3 neighbors for a standard improper
                if len(neighbors) != 3:
                    continue
                
                neighbor_indices = [n.GetIdx() for n in neighbors]
                neighbor_types = [atom_types.get(idx, "unk") for idx in neighbor_indices]
                
                # Create improper dihedral
                # Convention: center atom is third in the list (atom3)
                # Improper: atom1-atom2-center-atom4
                improper_info = {
                    "atoms": [neighbor_indices[0], neighbor_indices[1], center_idx, neighbor_indices[2]],
                    "types": [neighbor_types[0], neighbor_types[1], center_type, neighbor_types[2]],
                    "parameters": self._get_improper_parameters(
                        neighbor_types[0], neighbor_types[1], center_type, neighbor_types[2]
                    )
                }
                impropers.append(improper_info)
            
            return impropers
            
        except Exception as e:
            logger.error(f"Error generating improper dihedrals: {e}")
            return []
    
    def _get_improper_parameters(self, type1: str, type2: str, type3: str, type4: str) -> Optional[Dict[str, Any]]:
        """
        Get improper dihedral parameters.
        
        Args:
            type1, type2, type3, type4: GAFF atom types
            
        Returns:
            Dictionary with improper parameters, or None if not found
        """
        # Try to find in improper coefficients
        improper_key = f"{type1}-{type2}-{type3}-{type4}"
        params = self.improper_coeffs.get(improper_key)
        if params:
            return params
        
        # Try reversed
        improper_key_rev = f"{type4}-{type3}-{type2}-{type1}"
        params = self.improper_coeffs.get(improper_key_rev)
        if params:
            return params
        
        # Fallback: use generic improper parameters based on center atom
        # CVFF style improper: K * (1 + d * cos(n * phi))
        # For planarity, typically use n=2 (180° out of phase)
        center_type = type3
        
        # Generic improper parameters for common sp2 centers
        generic_impropers = {
            'c': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Carbonyl carbon
            'c2': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # sp2 carbon
            'ca': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Aromatic carbon
            'cc': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Aromatic 5-ring carbon
            'cd': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Aromatic 5-ring carbon
            'ce': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Conjugated sp2 carbon
            'cf': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Conjugated sp2 carbon
            'n': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},   # sp2 nitrogen (amide)
            'n2': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # sp2 nitrogen
            'na': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Aromatic nitrogen
            'nb': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Aromatic nitrogen
            'nc': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Aromatic nitrogen
            'nd': {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2},  # Aromatic nitrogen
        }
        
        params = generic_impropers.get(center_type)
        if params:
            logger.debug(f"Using generic improper parameters for center type {center_type}")
            return params
        
        # Ultimate fallback
        logger.debug(f"Using default improper parameters for {improper_key}")
        return {'style': 'cvff', 'k': 1.1, 'd': -1, 'n': 2}
    
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

            # Use unified topology mapping for consistent type assignment
            unified_topology = self.create_unified_topology_mapping(topology)
            bond_type_mapping = unified_topology["bond_type_mapping"]
            angle_type_mapping = unified_topology["angle_type_mapping"]
            dihedral_type_mapping = unified_topology["dihedral_type_mapping"]

            # Assign type IDs to topology elements using unified mapping (now using string keys)
            for bond in topology['bonds']:
                bond_type_key = "-".join(sorted(bond['types'][:2]))
                bond['type'] = bond_type_mapping.get(bond_type_key, 1)

            for angle in topology['angles']:
                angle_type_key = "-".join(angle['types'][:3])
                angle['type'] = angle_type_mapping.get(angle_type_key, 1)

            for dihedral in topology['dihedrals']:
                dihedral_type_key = "-".join(dihedral['types'][:4])
                dihedral['type'] = dihedral_type_mapping.get(dihedral_type_key, 1)

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
            lines.append(f"{len(unified_topology['unique_bond_types'])} bond types")
            lines.append(f"{len(unified_topology['unique_angle_types'])} angle types")
            lines.append(f"{len(unified_topology['unique_dihedral_types'])} dihedral types")
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

    def create_unified_topology_mapping(self, topology_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create consistent type mappings for all topology elements.
        
        This function ensures that both data file generation and script generation
        use identical sorting algorithms for type assignment, preventing inconsistencies.
        
        Args:
            topology_data: Dictionary containing bonds, angles, dihedrals lists
            
        Returns:
            Dictionary with consistent type mappings for bonds, angles, dihedrals
        """
        try:
            # Use deterministic sorting for all type collections
            # For bonds: sort atom types within each bond, then sort bonds
            bond_types = set()
            for bond in topology_data.get("bonds", []):
                if "types" in bond and len(bond["types"]) >= 2:
                    # Sort atom types within bond for consistent ordering (c3-hc == hc-c3)
                    bond_type = tuple(sorted(bond["types"][:2]))
                    bond_types.add(bond_type)
            
            # For angles: maintain order (center atom is important)
            angle_types = set()
            for angle in topology_data.get("angles", []):
                if "types" in angle and len(angle["types"]) >= 3:
                    # Don't sort angles - order matters (atom1-center-atom3)
                    angle_type = tuple(angle["types"][:3])
                    angle_types.add(angle_type)
            
            # For dihedrals: maintain order (torsion direction matters)
            dihedral_types = set()
            for dihedral in topology_data.get("dihedrals", []):
                if "types" in dihedral and len(dihedral["types"]) >= 4:
                    # Don't sort dihedrals - order matters for torsion
                    dihedral_type = tuple(dihedral["types"][:4])
                    dihedral_types.add(dihedral_type)
            
            # Sort the collected types for deterministic ordering
            sorted_bond_types = sorted(list(bond_types))
            sorted_angle_types = sorted(list(angle_types))
            sorted_dihedral_types = sorted(list(dihedral_types))
            
            # Create mappings with 1-based indexing using STRING KEYS (not tuples)
            # This eliminates JSON serialization issues with the MCP framework
            bond_type_mapping = {"-".join(bt): i+1 for i, bt in enumerate(sorted_bond_types)}
            angle_type_mapping = {"-".join(at): i+1 for i, at in enumerate(sorted_angle_types)}
            dihedral_type_mapping = {"-".join(dt): i+1 for i, dt in enumerate(sorted_dihedral_types)}
            
            logger.info(f"Created unified topology mapping: {len(bond_type_mapping)} bond types, "
                       f"{len(angle_type_mapping)} angle types, {len(dihedral_type_mapping)} dihedral types")
            
            return {
                "unique_bond_types": [list(bt) for bt in sorted_bond_types],  # Convert tuples to lists for JSON
                "unique_angle_types": [list(at) for at in sorted_angle_types],
                "unique_dihedral_types": [list(dt) for dt in sorted_dihedral_types],
                "bond_type_mapping": bond_type_mapping,
                "angle_type_mapping": angle_type_mapping,
                "dihedral_type_mapping": dihedral_type_mapping
            }
            
        except Exception as e:
            logger.error(f"Error creating unified topology mapping: {e}")
            return {
                "unique_bond_types": [],
                "unique_angle_types": [],
                "unique_dihedral_types": [],
                "bond_type_mapping": {},
                "angle_type_mapping": {},
                "dihedral_type_mapping": {}
            }

    def create_unified_topology_mapping_multi(self, processed_molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create unified topology mapping across multiple molecule types.
        
        Args:
            processed_molecules: List of processed molecule dictionaries
            
        Returns:
            Dictionary containing unified type mappings across all molecules
        """
        try:
            # Collect all topology data from all molecules
            all_bonds = []
            all_angles = []
            all_dihedrals = []
            
            for mol_info in processed_molecules:
                topology = mol_info.get('topology', {})
                all_bonds.extend(topology.get("bonds", []))
                all_angles.extend(topology.get("angles", []))
                all_dihedrals.extend(topology.get("dihedrals", []))
            
            # Create combined topology data
            combined_topology = {
                "bonds": all_bonds,
                "angles": all_angles,
                "dihedrals": all_dihedrals
            }
            
            # Use the unified mapping function
            return self.create_unified_topology_mapping(combined_topology)
            
        except Exception as e:
            logger.error(f"Error creating unified topology mapping for multiple molecules: {e}")
            return {
                "unique_bond_types": [],
                "unique_angle_types": [],
                "unique_dihedral_types": [],
                "bond_type_mapping": {},
                "angle_type_mapping": {},
                "dihedral_type_mapping": {}
            }

    def validate_type_consistency(
        self,
        data_file_path: Path,
        script_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate type consistency between LAMMPS data file and input script.
        
        Args:
            data_file_path: Path to LAMMPS data file
            script_content: Content of LAMMPS input script
            metadata: Optional metadata with type mappings
            
        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        issues = []
        
        try:
            # Parse data file for type counts
            data_file_types = self._parse_data_file_types(data_file_path)
            
            # Parse script for coefficient counts
            script_coeffs = self._parse_script_coefficients(script_content)
            
            # Check atom types
            if data_file_types.get('atom_types', 0) != len(script_coeffs.get('pair_coeffs', [])):
                issues.append(f"Atom type mismatch: data file has {data_file_types.get('atom_types', 0)} types, "
                            f"script has {len(script_coeffs.get('pair_coeffs', []))} pair coefficients")
            
            # Check bond types
            if data_file_types.get('bond_types', 0) != len(script_coeffs.get('bond_coeffs', [])):
                issues.append(f"Bond type mismatch: data file has {data_file_types.get('bond_types', 0)} types, "
                            f"script has {len(script_coeffs.get('bond_coeffs', []))} bond coefficients")
            
            # Check angle types
            if data_file_types.get('angle_types', 0) != len(script_coeffs.get('angle_coeffs', [])):
                issues.append(f"Angle type mismatch: data file has {data_file_types.get('angle_types', 0)} types, "
                            f"script has {len(script_coeffs.get('angle_coeffs', []))} angle coefficients")
            
            # Check dihedral types
            if data_file_types.get('dihedral_types', 0) != len(script_coeffs.get('dihedral_coeffs', [])):
                issues.append(f"Dihedral type mismatch: data file has {data_file_types.get('dihedral_types', 0)} types, "
                            f"script has {len(script_coeffs.get('dihedral_coeffs', []))} dihedral coefficients")
            
            is_consistent = len(issues) == 0
            
            if is_consistent:
                logger.info("Type consistency validation passed")
            else:
                logger.warning(f"Type consistency validation failed: {issues}")
            
            return is_consistent, issues
            
        except Exception as e:
            logger.error(f"Error validating type consistency: {e}")
            return False, [f"Validation error: {e}"]

    def _parse_data_file_types(self, data_file_path: Path) -> Dict[str, int]:
        """Parse LAMMPS data file to extract type counts."""
        type_counts = {}
        
        try:
            with open(data_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if 'atom types' in line:
                        type_counts['atom_types'] = int(line.split()[0])
                    elif 'bond types' in line:
                        type_counts['bond_types'] = int(line.split()[0])
                    elif 'angle types' in line:
                        type_counts['angle_types'] = int(line.split()[0])
                    elif 'dihedral types' in line:
                        type_counts['dihedral_types'] = int(line.split()[0])
                    elif 'improper types' in line:
                        type_counts['improper_types'] = int(line.split()[0])
        except Exception as e:
            logger.error(f"Error parsing data file {data_file_path}: {e}")
        
        return type_counts

    def _parse_script_coefficients(self, script_content: str) -> Dict[str, List[int]]:
        """Parse LAMMPS script to extract coefficient type IDs."""
        coeffs = {
            'pair_coeffs': [],
            'bond_coeffs': [],
            'angle_coeffs': [],
            'dihedral_coeffs': [],
            'improper_coeffs': []
        }
        
        try:
            for line in script_content.split('\n'):
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                if line.startswith('pair_coeff'):
                    parts = line.split()
                    if len(parts) >= 3:
                        type_id = int(parts[1])
                        if type_id not in coeffs['pair_coeffs']:
                            coeffs['pair_coeffs'].append(type_id)
                
                elif line.startswith('bond_coeff'):
                    parts = line.split()
                    if len(parts) >= 3:
                        type_id = int(parts[1])
                        if type_id not in coeffs['bond_coeffs']:
                            coeffs['bond_coeffs'].append(type_id)
                
                elif line.startswith('angle_coeff'):
                    parts = line.split()
                    if len(parts) >= 3:
                        type_id = int(parts[1])
                        if type_id not in coeffs['angle_coeffs']:
                            coeffs['angle_coeffs'].append(type_id)
                
                elif line.startswith('dihedral_coeff'):
                    parts = line.split()
                    if len(parts) >= 3:
                        type_id = int(parts[1])
                        if type_id not in coeffs['dihedral_coeffs']:
                            coeffs['dihedral_coeffs'].append(type_id)
        
        except Exception as e:
            logger.error(f"Error parsing script coefficients: {e}")
        
        return coeffs


# Create a global instance for easy access
forcefield_utils = ForceFieldUtils()