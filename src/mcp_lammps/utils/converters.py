"""
Converters - Utility functions for converting between different data formats.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def dict_to_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Convert dictionary to JSON file.
    
    Args:
        data: Dictionary to convert
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to convert dict to JSON: {e}")
        return False


def json_to_dict(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Convert JSON file to dictionary.
    
    Args:
        file_path: Input file path
        
    Returns:
        Dictionary or None if failed
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to convert JSON to dict: {e}")
        return None


def dataframe_to_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> bool:
    """
    Convert DataFrame to CSV file.
    
    Args:
        df: DataFrame to convert
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Failed to convert DataFrame to CSV: {e}")
        return False


def csv_to_dataframe(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Convert CSV file to DataFrame.
    
    Args:
        file_path: Input file path
        
    Returns:
        DataFrame or None if failed
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to convert CSV to DataFrame: {e}")
        return None


def array_to_npy(array: np.ndarray, file_path: Union[str, Path]) -> bool:
    """
    Convert numpy array to .npy file.
    
    Args:
        array: Array to convert
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        np.save(file_path, array)
        return True
    except Exception as e:
        logger.error(f"Failed to convert array to .npy: {e}")
        return False


def npy_to_array(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Convert .npy file to numpy array.
    
    Args:
        file_path: Input file path
        
    Returns:
        Array or None if failed
    """
    try:
        return np.load(file_path)
    except Exception as e:
        logger.error(f"Failed to convert .npy to array: {e}")
        return None


def lammps_data_to_dict(data_content: str) -> Dict[str, Any]:
    """
    Convert LAMMPS data file content to dictionary.
    
    Args:
        data_content: LAMMPS data file content
        
    Returns:
        Dictionary representation
    """
    result = {
        "atoms": [],
        "bonds": [],
        "angles": [],
        "dihedrals": [],
        "impropers": [],
        "masses": [],
        "pair_coeffs": [],
        "bond_coeffs": [],
        "angle_coeffs": [],
        "dihedral_coeffs": [],
        "improper_coeffs": [],
        "box": {},
        "header": {}
    }
    
    lines = data_content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Check for section headers
        if line.lower() in ['atoms', 'bonds', 'angles', 'dihedrals', 'impropers',
                           'masses', 'pair coeffs', 'bond coeffs', 'angle coeffs',
                           'dihedral coeffs', 'improper coeffs']:
            current_section = line.lower().replace(' ', '_')
            continue
        
        # Parse box dimensions
        if 'xlo' in line and 'xhi' in line:
            parts = line.split()
            result["box"]["xlo"] = float(parts[0])
            result["box"]["xhi"] = float(parts[1])
        elif 'ylo' in line and 'yhi' in line:
            parts = line.split()
            result["box"]["ylo"] = float(parts[0])
            result["box"]["yhi"] = float(parts[1])
        elif 'zlo' in line and 'zhi' in line:
            parts = line.split()
            result["box"]["zlo"] = float(parts[0])
            result["box"]["zhi"] = float(parts[1])
        
        # Parse header information
        elif 'atoms' in line and 'atom types' not in line:
            result["header"]["num_atoms"] = int(line.split()[0])
        elif 'atom types' in line:
            result["header"]["num_types"] = int(line.split()[0])
        elif 'bonds' in line and 'bond types' not in line:
            result["header"]["num_bonds"] = int(line.split()[0])
        elif 'bond types' in line:
            result["header"]["num_bond_types"] = int(line.split()[0])
        elif 'angles' in line and 'angle types' not in line:
            result["header"]["num_angles"] = int(line.split()[0])
        elif 'angle types' in line:
            result["header"]["num_angle_types"] = int(line.split()[0])
        
        # Parse data sections
        elif current_section and line and not line.startswith('#'):
            parts = line.split()
            if current_section == "atoms" and len(parts) >= 7:
                result["atoms"].append({
                    "id": int(parts[0]),
                    "type": int(parts[1]),
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "z": float(parts[4])
                })
            elif current_section == "bonds" and len(parts) >= 4:
                result["bonds"].append({
                    "id": int(parts[0]),
                    "type": int(parts[1]),
                    "atom1": int(parts[2]),
                    "atom2": int(parts[3])
                })
            elif current_section == "angles" and len(parts) >= 5:
                result["angles"].append({
                    "id": int(parts[0]),
                    "type": int(parts[1]),
                    "atom1": int(parts[2]),
                    "atom2": int(parts[3]),
                    "atom3": int(parts[4])
                })
            elif current_section == "masses" and len(parts) >= 2:
                result["masses"].append({
                    "type": int(parts[0]),
                    "mass": float(parts[1])
                })
    
    return result


def dict_to_lammps_data(data: Dict[str, Any]) -> str:
    """
    Convert dictionary to LAMMPS data file content.
    
    Args:
        data: Dictionary representation
        
    Returns:
        LAMMPS data file content
    """
    lines = []
    
    # Header
    lines.append("LAMMPS data file")
    lines.append("")
    
    # Counts
    header = data.get("header", {})
    if "num_atoms" in header:
        lines.append(f"{header['num_atoms']} atoms")
    if "num_types" in header:
        lines.append(f"{header['num_types']} atom types")
    if "num_bonds" in header:
        lines.append(f"{header['num_bonds']} bonds")
    if "num_bond_types" in header:
        lines.append(f"{header['num_bond_types']} bond types")
    if "num_angles" in header:
        lines.append(f"{header['num_angles']} angles")
    if "num_angle_types" in header:
        lines.append(f"{header['num_angle_types']} angle types")
    
    lines.append("")
    
    # Box dimensions
    box = data.get("box", {})
    if "xlo" in box and "xhi" in box:
        lines.append(f"{box['xlo']:.6f} {box['xhi']:.6f} xlo xhi")
    if "ylo" in box and "yhi" in box:
        lines.append(f"{box['ylo']:.6f} {box['yhi']:.6f} ylo yhi")
    if "zlo" in box and "zhi" in box:
        lines.append(f"{box['zlo']:.6f} {box['zhi']:.6f} zlo zhi")
    
    lines.append("")
    
    # Masses
    masses = data.get("masses", [])
    if masses:
        lines.append("Masses")
        lines.append("")
        for mass in masses:
            lines.append(f"{mass['type']} {mass['mass']:.6f}")
        lines.append("")
    
    # Atoms
    atoms = data.get("atoms", [])
    if atoms:
        lines.append("Atoms")
        lines.append("")
        for atom in atoms:
            lines.append(f"{atom['id']} {atom['type']} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}")
        lines.append("")
    
    # Bonds
    bonds = data.get("bonds", [])
    if bonds:
        lines.append("Bonds")
        lines.append("")
        for bond in bonds:
            lines.append(f"{bond['id']} {bond['type']} {bond['atom1']} {bond['atom2']}")
        lines.append("")
    
    # Angles
    angles = data.get("angles", [])
    if angles:
        lines.append("Angles")
        lines.append("")
        for angle in angles:
            lines.append(f"{angle['id']} {angle['type']} {angle['atom1']} {angle['atom2']} {angle['atom3']}")
        lines.append("")
    
    return "\n".join(lines)


def xyz_to_dict(xyz_content: str) -> Dict[str, Any]:
    """
    Convert XYZ file content to dictionary.
    
    Args:
        xyz_content: XYZ file content
        
    Returns:
        Dictionary representation
    """
    lines = xyz_content.strip().split('\n')
    
    if len(lines) < 2:
        return {"atoms": [], "comment": ""}
    
    num_atoms = int(lines[0])
    comment = lines[1] if len(lines) > 1 else ""
    atoms = []
    
    for i in range(2, min(2 + num_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 4:
            atoms.append({
                "element": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3])
            })
    
    return {
        "num_atoms": num_atoms,
        "comment": comment,
        "atoms": atoms
    }


def dict_to_xyz(data: Dict[str, Any]) -> str:
    """
    Convert dictionary to XYZ file content.
    
    Args:
        data: Dictionary representation
        
    Returns:
        XYZ file content
    """
    atoms = data.get("atoms", [])
    comment = data.get("comment", "")
    
    lines = [str(len(atoms)), comment]
    
    for atom in atoms:
        lines.append(f"{atom['element']} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}")
    
    return "\n".join(lines)


def thermo_data_to_dataframe(thermo_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert thermodynamic data list to DataFrame.
    
    Args:
        thermo_data: List of thermodynamic data dictionaries
        
    Returns:
        DataFrame
    """
    if not thermo_data:
        return pd.DataFrame()
    
    return pd.DataFrame(thermo_data)


def dataframe_to_thermo_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to thermodynamic data list.
    
    Args:
        df: DataFrame
        
    Returns:
        List of thermodynamic data dictionaries
    """
    if df.empty:
        return []
    
    return df.to_dict('records')


def trajectory_data_to_dict(trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert trajectory data to a more manageable dictionary format.
    
    Args:
        trajectory_data: Raw trajectory data
        
    Returns:
        Processed trajectory data
    """
    result = {
        "timesteps": trajectory_data.get("timesteps", []),
        "num_frames": len(trajectory_data.get("timesteps", [])),
        "num_atoms": 0,
        "box_data": trajectory_data.get("box", []),
        "atom_data": {}
    }
    
    # Process atom data
    positions = trajectory_data.get("positions", [])
    if positions:
        result["num_atoms"] = positions[0].shape[0] if positions else 0
        result["atom_data"]["positions"] = [pos.tolist() for pos in positions]
    
    velocities = trajectory_data.get("velocities", [])
    if velocities:
        result["atom_data"]["velocities"] = [vel.tolist() for vel in velocities]
    
    forces = trajectory_data.get("forces", [])
    if forces:
        result["atom_data"]["forces"] = [force.tolist() for force in forces]
    
    types = trajectory_data.get("types", [])
    if types:
        result["atom_data"]["types"] = [t.tolist() for t in types]
    
    return result


def dict_to_trajectory_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dictionary back to trajectory data format.
    
    Args:
        data: Processed trajectory data
        
    Returns:
        Raw trajectory data
    """
    result = {
        "timesteps": data.get("timesteps", []),
        "box": data.get("box_data", [])
    }
    
    # Convert atom data back to numpy arrays
    atom_data = data.get("atom_data", {})
    
    if "positions" in atom_data:
        result["positions"] = [np.array(pos) for pos in atom_data["positions"]]
    
    if "velocities" in atom_data:
        result["velocities"] = [np.array(vel) for vel in atom_data["velocities"]]
    
    if "forces" in atom_data:
        result["forces"] = [np.array(force) for force in atom_data["forces"]]
    
    if "types" in atom_data:
        result["types"] = [np.array(t) for t in atom_data["types"]]
    
    return result 