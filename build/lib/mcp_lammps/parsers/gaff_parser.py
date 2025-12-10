"""
GAFF Parameter Parser - Extracts all force field parameters from gaff.lt files.

This module provides comprehensive parsing of GAFF/GAFF2 force field parameters
from moltemplate (.lt) files, extracting:
- Atom types and masses
- Pair coefficients (Lennard-Jones parameters)
- Bond coefficients
- Angle coefficients  
- Dihedral coefficients
- Improper coefficients
- Force field settings and styles
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

logger = logging.getLogger(__name__)


class GAFFParser:
    """
    Parser for GAFF force field parameters from moltemplate (.lt) files.
    
    This class extracts comprehensive GAFF2 parameters and organizes them
    into structured dictionaries for efficient lookup and usage.
    """
    
    def __init__(self):
        """Initialize the GAFF parser."""
        self.atom_types = {}
        self.atom_masses = {}
        self.pair_coeffs = {}
        self.bond_coeffs = {}
        self.angle_coeffs = {}
        self.dihedral_coeffs = {}
        self.improper_coeffs = {}
        self.force_field_settings = {}
        self.atom_descriptions = {}
        
    def parse_gaff_file(self, gaff_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a GAFF .lt file and extract all force field parameters.
        
        Args:
            gaff_file: Path to the GAFF .lt file
            
        Returns:
            Dictionary containing all parsed parameters
        """
        gaff_path = Path(gaff_file)
        if not gaff_path.exists():
            raise FileNotFoundError(f"GAFF file not found: {gaff_file}")
            
        logger.info(f"Parsing GAFF file: {gaff_path}")
        
        with open(gaff_path, 'r') as f:
            content = f.read()
            
        # Parse different sections
        self._parse_atom_descriptions(content)
        self._parse_atom_masses(content)
        self._parse_pair_coefficients(content)
        self._parse_bond_coefficients(content)
        self._parse_angle_coefficients(content)
        self._parse_dihedral_coefficients(content)
        self._parse_improper_coefficients(content)
        self._parse_force_field_settings(content)
        
        # Compile results
        parameters = {
            "atom_types": self.atom_types,
            "atom_masses": self.atom_masses,
            "atom_descriptions": self.atom_descriptions,
            "pair_coefficients": self.pair_coeffs,
            "bond_coefficients": self.bond_coeffs,
            "angle_coefficients": self.angle_coeffs,
            "dihedral_coefficients": self.dihedral_coeffs,
            "improper_coefficients": self.improper_coeffs,
            "force_field_settings": self.force_field_settings,
            "statistics": self._generate_statistics()
        }
        
        logger.info(f"Successfully parsed GAFF parameters: {self._generate_statistics()}")
        return parameters
    
    def _parse_atom_descriptions(self, content: str) -> None:
        """Parse atom type descriptions from comments."""
        logger.debug("Parsing atom type descriptions")
        
        # Find atom type description section
        pattern = r'#\s*@atom:(\w+)\s+(.+?)(?=\n|$)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for atom_type, description in matches:
            self.atom_descriptions[atom_type] = description.strip()
            
        logger.debug(f"Found {len(self.atom_descriptions)} atom type descriptions")
    
    def _parse_atom_masses(self, content: str) -> None:
        """Parse atom masses from the masses section."""
        logger.debug("Parsing atom masses")
        
        # Find masses section
        mass_pattern = r'@atom:(\w+)\s+([\d.]+)'
        matches = re.findall(mass_pattern, content)
        
        for atom_type, mass in matches:
            self.atom_masses[atom_type] = float(mass)
            self.atom_types[atom_type] = {
                "mass": float(mass),
                "description": self.atom_descriptions.get(atom_type, "")
            }
            
        logger.debug(f"Found {len(self.atom_masses)} atom masses")
    
    def _parse_pair_coefficients(self, content: str) -> None:
        """Parse pair coefficients (Lennard-Jones parameters)."""
        logger.debug("Parsing pair coefficients")
        
        # Pattern for pair_coeff lines
        pattern = r'pair_coeff\s+@atom:(\w+)\s+@atom:(\w+)\s+lj/charmm/coul/long\s+([\d.]+)\s+([\d.]+)'
        matches = re.findall(pattern, content)
        
        for atom1, atom2, epsilon, sigma in matches:
            if atom1 == atom2:  # Self interactions
                self.pair_coeffs[atom1] = {
                    "epsilon": float(epsilon),
                    "sigma": float(sigma),
                    "style": "lj/charmm/coul/long"
                }
                
        logger.debug(f"Found {len(self.pair_coeffs)} pair coefficients")
    
    def _parse_bond_coefficients(self, content: str) -> None:
        """Parse bond coefficients."""
        logger.debug("Parsing bond coefficients")
        
        # Pattern for bond_coeff lines
        pattern = r'bond_coeff\s+@bond:(\w+)-(\w+)\s+harmonic\s+([\d.]+)\s+([\d.]+)(?:\s+#\s*(.+?))?(?=\n|$)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for atom1, atom2, k, r0, comment in matches:
            bond_key = tuple(sorted([atom1, atom2]))
            self.bond_coeffs[bond_key] = {
                "k": float(k),
                "r0": float(r0),
                "style": "harmonic",
                "comment": comment.strip() if comment else ""
            }
            
        logger.debug(f"Found {len(self.bond_coeffs)} bond coefficients")
    
    def _parse_angle_coefficients(self, content: str) -> None:
        """Parse angle coefficients."""
        logger.debug("Parsing angle coefficients")
        
        # Pattern for angle_coeff lines
        pattern = r'angle_coeff\s+@angle:(\w+)-(\w+)-(\w+)\s+harmonic\s+([\d.]+)\s+([\d.]+)(?:\s+#\s*(.+?))?(?=\n|$)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for atom1, atom2, atom3, k, theta0, comment in matches:
            angle_key = (atom1, atom2, atom3)
            self.angle_coeffs[angle_key] = {
                "k": float(k),
                "theta0": float(theta0),
                "style": "harmonic",
                "comment": comment.strip() if comment else ""
            }
            
        logger.debug(f"Found {len(self.angle_coeffs)} angle coefficients")
    
    def _parse_dihedral_coefficients(self, content: str) -> None:
        """Parse dihedral coefficients."""
        logger.debug("Parsing dihedral coefficients")
        
        # Pattern for dihedral_coeff lines with fourier style
        pattern = r'dihedral_coeff\s+@dihedral:(.+?)\s+fourier\s+(\d+)\s+([\d.-]+)\s+(\d+)\s+([\d.-]+)(?:\s+#\s*(.+?))?(?=\n|$)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for dihedral_def, n_terms, k, n, phase, comment in matches:
            # Parse dihedral definition (e.g., "X-c-c-X" or "c3-c3-c3-c3")
            atoms = dihedral_def.split('-')
            if len(atoms) == 4:
                dihedral_key = tuple(atoms)
                self.dihedral_coeffs[dihedral_key] = {
                    "style": "fourier",
                    "n_terms": int(n_terms),
                    "k": float(k),
                    "n": int(n),
                    "phase": float(phase),
                    "comment": comment.strip() if comment else ""
                }
                
        logger.debug(f"Found {len(self.dihedral_coeffs)} dihedral coefficients")
    
    def _parse_improper_coefficients(self, content: str) -> None:
        """Parse improper coefficients."""
        logger.debug("Parsing improper coefficients")
        
        # Find improper definitions in the "Data Impropers By Type" section
        improper_section = re.search(
            r'write_once\("Data Impropers By Type.*?\"\)\s*\{(.*?)\}',
            content, re.DOTALL
        )
        
        if improper_section:
            improper_content = improper_section.group(1)
            # Pattern for improper definitions
            pattern = r'@improper:(\w+)-(\w+)-(\w+)-(\w+)\s+@atom:(\w+)\s+@atom:(\w+)\s+@atom:(\w+)\s+@atom:(\w+)'
            matches = re.findall(pattern, improper_content)
            
            for match in matches:
                improper_key = tuple(match[:4])  # First 4 are the improper type definition
                atom_types = tuple(match[4:])    # Last 4 are the actual atom types
                self.improper_coeffs[improper_key] = {
                    "atom_types": atom_types,
                    "style": "cvff"  # Default style from gaff.lt
                }
                
        logger.debug(f"Found {len(self.improper_coeffs)} improper coefficients")
    
    def _parse_force_field_settings(self, content: str) -> None:
        """Parse force field settings and styles."""
        logger.debug("Parsing force field settings")
        
        # Find the "In Init" section
        init_section = re.search(
            r'write_once\("In Init"\)\s*\{(.*?)\}',
            content, re.DOTALL
        )
        
        if init_section:
            init_content = init_section.group(1)
            
            # Parse individual settings
            settings = {}
            
            # Parse style settings
            style_patterns = {
                'units': r'units\s+(\w+)',
                'atom_style': r'atom_style\s+(\w+)',
                'bond_style': r'bond_style\s+(.+?)(?=\n|$)',
                'angle_style': r'angle_style\s+(.+?)(?=\n|$)',
                'dihedral_style': r'dihedral_style\s+(.+?)(?=\n|$)',
                'improper_style': r'improper_style\s+(.+?)(?=\n|$)',
                'pair_style': r'pair_style\s+(.+?)(?=\n|$)',
                'kspace_style': r'kspace_style\s+(.+?)(?=\n|$)',
            }
            
            for setting, pattern in style_patterns.items():
                match = re.search(pattern, init_content, re.MULTILINE)
                if match:
                    settings[setting] = match.group(1).strip()
            
            # Parse other settings
            other_patterns = {
                'pair_modify': r'pair_modify\s+(.+?)(?=\n|$)',
                'special_bonds': r'special_bonds\s+(.+?)(?=\n|$)',
            }
            
            for setting, pattern in other_patterns.items():
                match = re.search(pattern, init_content, re.MULTILINE)
                if match:
                    settings[setting] = match.group(1).strip()
            
            self.force_field_settings = settings
            
        logger.debug(f"Found {len(self.force_field_settings)} force field settings")
    
    def _generate_statistics(self) -> Dict[str, int]:
        """Generate statistics about parsed parameters."""
        return {
            "atom_types": len(self.atom_types),
            "pair_coefficients": len(self.pair_coeffs),
            "bond_coefficients": len(self.bond_coeffs),
            "angle_coefficients": len(self.angle_coeffs),
            "dihedral_coefficients": len(self.dihedral_coeffs),
            "improper_coefficients": len(self.improper_coeffs),
            "force_field_settings": len(self.force_field_settings)
        }
    
    def save_parameters(self, output_file: Union[str, Path], parameters: Dict[str, Any]) -> None:
        """
        Save parsed parameters to a JSON file.
        
        Args:
            output_file: Path to output JSON file
            parameters: Parsed parameters dictionary
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tuple keys to strings for JSON serialization
        json_parameters = self._prepare_for_json(parameters)
        
        with open(output_path, 'w') as f:
            json.dump(json_parameters, f, indent=2, sort_keys=True)
            
        logger.info(f"Saved GAFF parameters to: {output_path}")
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization by converting tuples to strings."""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Convert tuple keys to string representation
                if isinstance(key, tuple):
                    key_str = "-".join(str(k) for k in key)
                else:
                    key_str = str(key)
                result[key_str] = self._prepare_for_json(value)
            return result
        elif isinstance(data, (list, tuple)):
            return [self._prepare_for_json(item) for item in data]
        else:
            return data
    
    @staticmethod
    def load_parameters(parameter_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load parameters from a JSON file.
        
        Args:
            parameter_file: Path to parameter JSON file
            
        Returns:
            Dictionary containing loaded parameters
        """
        parameter_path = Path(parameter_file)
        if not parameter_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {parameter_file}")
            
        with open(parameter_path, 'r') as f:
            parameters = json.load(f)
            
        logger.info(f"Loaded GAFF parameters from: {parameter_path}")
        return parameters


def main():
    """Main function for testing the parser."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python gaff_parser.py <gaff.lt> <output.json>")
        sys.exit(1)
    
    gaff_file = sys.argv[1]
    output_file = sys.argv[2]
    
    parser = GAFFParser()
    parameters = parser.parse_gaff_file(gaff_file)
    parser.save_parameters(output_file, parameters)
    
    print(f"Successfully parsed GAFF file: {gaff_file}")
    print(f"Parameters saved to: {output_file}")
    print(f"Statistics: {parameters['statistics']}")


if __name__ == "__main__":
    main()
