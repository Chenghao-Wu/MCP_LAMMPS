"""
Data Handler - Manages input/output files and data processing for LAMMPS simulations.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Handles data processing and file management for LAMMPS simulations.
    
    This class provides functionality for:
    - Reading and writing simulation data files
    - Processing trajectory and thermodynamic data
    - Converting between different file formats
    - Managing simulation input/output files
    """
    
    def __init__(self, work_dir: Path):
        """
        Initialize the data handler.
        
        Args:
            work_dir: Base working directory for data files
        """
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.input_dir = work_dir / "input"
        self.output_dir = work_dir / "output"
        self.temp_dir = work_dir / "temp"
        
        for dir_path in [self.input_dir, self.output_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Data handler initialized with work directory: {work_dir}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the data handler.
        
        Returns:
            Status information dictionary
        """
        return {
            "work_directory": str(self.work_dir),
            "input_files": len(list(self.input_dir.glob("*"))),
            "output_files": len(list(self.output_dir.glob("*"))),
            "temp_files": len(list(self.temp_dir.glob("*")))
        }
    
    def save_structure_file(
        self,
        filename: str,
        content: str,
        file_type: str = "data"
    ) -> Path:
        """
        Save a structure file to the input directory.
        
        Args:
            filename: Name of the file
            content: File content
            file_type: Type of structure file (data, xyz, pdb, etc.)
            
        Returns:
            Path to the saved file
        """
        file_path = self.input_dir / filename
        
        # Add appropriate extension if not present
        if not file_path.suffix:
            if file_type == "data":
                file_path = file_path.with_suffix(".data")
            elif file_type == "xyz":
                file_path = file_path.with_suffix(".xyz")
            elif file_type == "pdb":
                file_path = file_path.with_suffix(".pdb")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved structure file: {file_path}")
        return file_path
    
    def save_script_file(self, filename: str, content: str) -> Path:
        """
        Save a LAMMPS script file.
        
        Args:
            filename: Name of the script file
            content: Script content
            
        Returns:
            Path to the saved file
        """
        file_path = self.input_dir / filename
        if not file_path.suffix:
            file_path = file_path.with_suffix(".lmp")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved script file: {file_path}")
        return file_path
    
    def read_thermo_data(self, file_path: Path) -> pd.DataFrame:
        """
        Read thermodynamic data from a LAMMPS log file.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            DataFrame with thermodynamic data
        """
        try:
            # Read the log file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the thermo data section
            thermo_data = []
            in_thermo_section = False
            headers = []
            
            for line in lines:
                line = line.strip()
                
                # Look for thermo_style command to get headers
                if line.startswith("thermo_style"):
                    parts = line.split()
                    if len(parts) > 1:
                        headers = parts[1:]
                
                # Look for thermo data start
                if line.startswith("Step"):
                    in_thermo_section = True
                    if not headers:
                        headers = line.split()
                    continue
                
                # Look for thermo data end
                if in_thermo_section and (line.startswith("Loop") or not line):
                    in_thermo_section = False
                    continue
                
                # Parse thermo data lines
                if in_thermo_section and line and not line.startswith("#"):
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) == len(headers):
                            thermo_data.append(values)
                    except ValueError:
                        continue
            
            if thermo_data and headers:
                df = pd.DataFrame(thermo_data, columns=headers)
                return df
            else:
                logger.warning(f"No thermodynamic data found in {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to read thermo data from {file_path}: {e}")
            return pd.DataFrame()
    
    def read_trajectory_data(self, file_path: Path) -> Dict[str, Any]:
        """
        Read trajectory data from a LAMMPS dump file.
        
        Args:
            file_path: Path to the dump file
            
        Returns:
            Dictionary with trajectory data
        """
        try:
            trajectory_data = {
                'timesteps': [],
                'positions': [],
                'velocities': [],
                'forces': [],
                'types': [],
                'box': []
            }
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for ITEM: TIMESTEP
                if line == "ITEM: TIMESTEP":
                    timestep = int(lines[i + 1].strip())
                    trajectory_data['timesteps'].append(timestep)
                    i += 2
                    continue
                
                # Look for ITEM: NUMBER OF ATOMS
                if line == "ITEM: NUMBER OF ATOMS":
                    num_atoms = int(lines[i + 1].strip())
                    i += 2
                    continue
                
                # Look for ITEM: BOX BOUNDS
                if line.startswith("ITEM: BOX BOUNDS"):
                    box_data = []
                    for j in range(3):
                        bounds = [float(x) for x in lines[i + 1 + j].split()]
                        box_data.append(bounds)
                    trajectory_data['box'].append(box_data)
                    i += 4
                    continue
                
                # Look for ITEM: ATOMS
                if line.startswith("ITEM: ATOMS"):
                    # Parse atom data
                    atom_data = []
                    i += 1
                    
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("ITEM:"):
                        try:
                            values = lines[i].strip().split()
                            atom_data.append([float(x) for x in values])
                        except ValueError:
                            pass
                        i += 1
                    
                    if atom_data:
                        # Convert to numpy arrays
                        atom_array = np.array(atom_data)
                        
                        # Extract different properties based on available columns
                        # This is a simplified version - actual implementation would be more robust
                        if atom_array.shape[1] >= 3:  # At least id, type, x, y, z
                            trajectory_data['positions'].append(atom_array[:, 2:5])
                            trajectory_data['types'].append(atom_array[:, 1].astype(int))
                        
                        if atom_array.shape[1] >= 6:  # Including velocities
                            trajectory_data['velocities'].append(atom_array[:, 5:8])
                        
                        if atom_array.shape[1] >= 9:  # Including forces
                            trajectory_data['forces'].append(atom_array[:, 8:11])
                    
                    continue
                
                i += 1
            
            return trajectory_data
            
        except Exception as e:
            logger.error(f"Failed to read trajectory data from {file_path}: {e}")
            return {}
    
    def save_results(self, simulation_id: str, results: Dict[str, Any]) -> Path:
        """
        Save simulation results to a JSON file.
        
        Args:
            simulation_id: Simulation ID
            results: Results dictionary
            
        Returns:
            Path to the saved results file
        """
        results_file = self.output_dir / f"{simulation_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved results for simulation {simulation_id}: {results_file}")
        return results_file
    
    def load_results(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load simulation results from a JSON file.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Results dictionary or None if not found
        """
        results_file = self.output_dir / f"{simulation_id}_results.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            logger.error(f"Failed to load results for simulation {simulation_id}: {e}")
            return None
    
    def export_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
        filename: str,
        format: str = "csv"
    ) -> Path:
        """
        Export data in various formats.
        
        Args:
            data: Data to export
            filename: Output filename
            format: Export format (csv, json, npy, txt)
            
        Returns:
            Path to the exported file
        """
        export_file = self.output_dir / filename
        
        try:
            if format == "csv" and isinstance(data, pd.DataFrame):
                export_file = export_file.with_suffix(".csv")
                data.to_csv(export_file, index=False)
            
            elif format == "json":
                export_file = export_file.with_suffix(".json")
                if isinstance(data, pd.DataFrame):
                    data.to_json(export_file, orient='records', indent=2)
                else:
                    with open(export_file, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
            
            elif format == "npy" and isinstance(data, np.ndarray):
                export_file = export_file.with_suffix(".npy")
                np.save(export_file, data)
            
            elif format == "txt":
                export_file = export_file.with_suffix(".txt")
                if isinstance(data, pd.DataFrame):
                    data.to_csv(export_file, sep='\t', index=False)
                elif isinstance(data, np.ndarray):
                    np.savetxt(export_file, data)
                else:
                    with open(export_file, 'w') as f:
                        f.write(str(data))
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported data to: {export_file}")
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
    
    def create_water_molecule_file(self) -> Path:
        """
        Create a water molecule template file for LAMMPS.
        
        Returns:
            Path to the water molecule file
        """
        water_content = """3 atoms
2 bonds
1 angles

Coords

1 0.000000 0.000000 0.000000
2 0.957200 0.000000 0.000000
3 0.240000 0.927000 0.000000

Types

1 1
2 2
3 2

Bonds

1 1 1 2
2 1 1 3

Angles

1 1 2 1 3
"""
        
        water_file = self.input_dir / "H2O.txt"
        with open(water_file, 'w') as f:
            f.write(water_content)
        
        logger.info(f"Created water molecule file: {water_file}")
        return water_file
    
    def create_water_box_file(self, num_molecules: int = 100, box_size: float = 30.0) -> Path:
        """
        Create a water box data file for LAMMPS.
        
        Args:
            num_molecules: Number of water molecules
            box_size: Simulation box size (Angstroms)
            
        Returns:
            Path to the water box file
        """
        import random
        
        # TIP3P water molecule geometry (Angstroms)
        # O-H bond length: 0.9572, H-O-H angle: 104.52 degrees
        oh_distance = 0.9572
        hoh_angle_rad = 104.52 * 3.14159 / 180.0
        
        # Calculate H positions relative to O
        h1_x = oh_distance
        h1_y = 0.0
        h1_z = 0.0
        
        h2_x = oh_distance * 0.5
        h2_y = oh_distance * 0.866  # sin(60°)
        h2_z = 0.0
        
        # Create water box content
        total_atoms = num_molecules * 3
        total_bonds = num_molecules * 2
        total_angles = num_molecules
        
        content = f"""LAMMPS data file - Water box

{total_atoms} atoms
{total_bonds} bonds
{total_angles} angles

3 atom types
2 bond types
1 angle types

{-box_size/2:.6f} {box_size/2:.6f} xlo xhi
{-box_size/2:.6f} {box_size/2:.6f} ylo yhi
{-box_size/2:.6f} {box_size/2:.6f} zlo zhi

Masses

1 15.9994
2 1.008
3 1.008

Atoms

"""
        
        # Add water molecules
        atom_id = 1
        bond_id = 1
        angle_id = 1
        
        for mol in range(num_molecules):
            # Random position for oxygen
            ox = random.uniform(-box_size/2 + 2, box_size/2 - 2)
            oy = random.uniform(-box_size/2 + 2, box_size/2 - 2)
            oz = random.uniform(-box_size/2 + 2, box_size/2 - 2)
            
            # Add oxygen atom (format: atom-ID molecule-ID atom-type charge x y z)
            content += f"{atom_id} {mol+1} 1 -0.8476 {ox:.6f} {oy:.6f} {oz:.6f}\n"
            o_atom_id = atom_id
            atom_id += 1
            
            # Add hydrogen atoms
            h1x = ox + h1_x
            h1y = oy + h1_y
            h1z = oz + h1_z
            content += f"{atom_id} {mol+1} 2 0.4238 {h1x:.6f} {h1y:.6f} {h1z:.6f}\n"
            h1_atom_id = atom_id
            atom_id += 1
            
            h2x = ox + h2_x
            h2y = oy + h2_y
            h2z = oz + h2_z
            content += f"{atom_id} {mol+1} 2 0.4238 {h2x:.6f} {h2y:.6f} {h2z:.6f}\n"
            h2_atom_id = atom_id
            atom_id += 1
        
        # Add bonds
        content += "\nBonds\n\n"
        atom_id = 1
        for mol in range(num_molecules):
            o_atom_id = atom_id
            h1_atom_id = atom_id + 1
            h2_atom_id = atom_id + 2
            
            content += f"{bond_id} 1 {o_atom_id} {h1_atom_id}\n"
            bond_id += 1
            content += f"{bond_id} 1 {o_atom_id} {h2_atom_id}\n"
            bond_id += 1
            
            atom_id += 3
        
        # Add angles
        content += "\nAngles\n\n"
        atom_id = 1
        for mol in range(num_molecules):
            o_atom_id = atom_id
            h1_atom_id = atom_id + 1
            h2_atom_id = atom_id + 2
            
            content += f"{angle_id} 1 {h1_atom_id} {o_atom_id} {h2_atom_id}\n"
            angle_id += 1
            
            atom_id += 3
        
        water_box_file = self.input_dir / "water_box.data"
        with open(water_box_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Created water box file: {water_box_file}")
        return water_box_file
    
    def create_simple_structure(
        self,
        num_atoms: int = 100,
        box_size: float = 20.0,
        atom_type: int = 1
    ) -> Path:
        """
        Create a simple structure file for testing.
        
        Args:
            num_atoms: Number of atoms
            box_size: Simulation box size
            atom_type: Atom type
            
        Returns:
            Path to the structure file
        """
        import random
        
        structure_content = f"""LAMMPS data file - Simple structure

{num_atoms} atoms
1 atom types

{-box_size/2:.6f} {box_size/2:.6f} xlo xhi
{-box_size/2:.6f} {box_size/2:.6f} ylo yhi
{-box_size/2:.6f} {box_size/2:.6f} zlo zhi

Atoms

"""
        
        # Add atom coordinates
        for i in range(1, num_atoms + 1):
            x = random.uniform(-box_size/2, box_size/2)
            y = random.uniform(-box_size/2, box_size/2)
            z = random.uniform(-box_size/2, box_size/2)
            structure_content += f"{i} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n"
        
        structure_file = self.input_dir / "simple_structure.data"
        with open(structure_file, 'w') as f:
            f.write(structure_content)
        
        logger.info(f"Created simple structure file: {structure_file}")
        return structure_file
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary files.
        
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        for temp_file in self.temp_dir.iterdir():
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
        return cleaned_count
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information dictionary
        """
        if not file_path.exists():
            return {"error": "File not found"}
        
        try:
            stat = file_path.stat()
            info = {
                "name": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "type": file_path.suffix,
                "path": str(file_path)
            }
            
            # Add specific info based on file type
            if file_path.suffix == ".json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                info["json_keys"] = list(data.keys()) if isinstance(data, dict) else None
            
            elif file_path.suffix == ".csv":
                df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows
                info["csv_columns"] = list(df.columns)
                info["csv_rows"] = len(df)
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def import_smiles_structure(
        self,
        smiles: str,
        molecule_name: str,
        optimize_geometry: bool = True
    ) -> Path:
        """
        Import molecular structure from SMILES string.
        
        Args:
            smiles: SMILES string representation
            molecule_name: Name for the molecule
            optimize_geometry: Whether to optimize 3D geometry
            
        Returns:
            Path to the created structure file
        """
        try:
            from .utils.molecular_utils import molecular_utils
            from .utils.forcefield_utils import forcefield_utils
            
            # Convert SMILES to 3D structure
            mol = molecular_utils.smiles_to_3d(smiles, optimize=optimize_geometry)
            if mol is None:
                raise ValueError(f"Failed to convert SMILES to 3D structure: {smiles}")
            
            # Assign GAFF atom types
            atom_types = forcefield_utils.assign_gaff_atom_types(mol)
            if not atom_types:
                raise ValueError("Failed to assign GAFF atom types")
            
            # Calculate partial charges
            charges = forcefield_utils.calculate_partial_charges(mol, method="gasteiger")
            if not charges:
                logger.warning("Failed to calculate partial charges, using GAFF estimates")
                charges = forcefield_utils.get_gaff_charge_estimates(mol, atom_types)
            
            # Generate topology
            topology = forcefield_utils.generate_topology(mol, atom_types)
            
            # Create LAMMPS data file
            data_content = forcefield_utils.create_lammps_data_file(mol, atom_types, topology, charges)
            if not data_content:
                raise ValueError("Failed to create LAMMPS data file")
            
            # Save the file
            filename = f"{molecule_name}_from_smiles.data"
            file_path = self.input_dir / filename
            
            with open(file_path, 'w') as f:
                f.write(data_content)
            
            # Save complete parameter information
            metadata = {
                "smiles": smiles,
                "molecule_name": molecule_name,
                "atom_types": atom_types,
                "charges": charges,
                "topology": topology,
                "topology_stats": {
                    "bonds": len(topology["bonds"]),
                    "angles": len(topology["angles"]),
                    "dihedrals": len(topology["dihedrals"])
                },
                "properties": molecular_utils.calculate_molecular_properties(mol)
            }
            
            metadata_file = file_path.with_suffix(".json")
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Successfully imported SMILES structure: {molecule_name}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to import SMILES structure: {e}")
            raise
    
    def import_mol2_file(self, mol2_content: str, filename: str) -> Path:
        """
        Import MOL2 format file with force field parameters.
        
        Args:
            mol2_content: MOL2 file content as string
            filename: Name for the output file
            
        Returns:
            Path to the processed structure file
        """
        try:
            from .utils.molecular_utils import molecular_utils
            
            # Save original MOL2 file
            mol2_file = self.input_dir / f"{filename}.mol2"
            with open(mol2_file, 'w') as f:
                f.write(mol2_content)
            
            # Convert to LAMMPS format if needed
            # For now, we'll try to convert using OpenBabel
            if molecular_utils.openbabel_available:
                # Convert MOL2 to SDF first, then process with RDKit
                sdf_content = molecular_utils.convert_format(mol2_content, "mol2", "sdf")
                if sdf_content:
                    return self.import_sdf_file(sdf_content, filename)
            
            logger.info(f"Imported MOL2 file: {filename}")
            return mol2_file
            
        except Exception as e:
            logger.error(f"Failed to import MOL2 file: {e}")
            raise
    
    def import_sdf_file(self, sdf_content: str, filename: str) -> Path:
        """
        Import SDF format file.
        
        Args:
            sdf_content: SDF file content as string
            filename: Name for the output file
            
        Returns:
            Path to the processed structure file
        """
        try:
            from .utils.molecular_utils import molecular_utils
            from .utils.forcefield_utils import forcefield_utils
            
            # Save original SDF file
            sdf_file = self.input_dir / f"{filename}.sdf"
            with open(sdf_file, 'w') as f:
                f.write(sdf_content)
            
            # Try to process with RDKit if available
            if molecular_utils.rdkit_available:
                from rdkit import Chem
                
                # Read molecule from SDF
                mol_supplier = Chem.SDMolSupplier()
                mol_supplier.SetData(sdf_content)
                
                for mol in mol_supplier:
                    if mol is not None:
                        # Assign GAFF atom types
                        atom_types = forcefield_utils.assign_gaff_atom_types(mol)
                        if atom_types:
                            # Calculate partial charges
                            charges = forcefield_utils.calculate_partial_charges(mol, method="gasteiger")
                            if not charges:
                                charges = forcefield_utils.get_gaff_charge_estimates(mol, atom_types)
                            
                            # Generate topology
                            topology = forcefield_utils.generate_topology(mol, atom_types)
                            
                            # Create LAMMPS data file
                            data_content = forcefield_utils.create_lammps_data_file(mol, atom_types, topology, charges)
                            if data_content:
                                data_file = self.input_dir / f"{filename}.data"
                                with open(data_file, 'w') as f:
                                    f.write(data_content)
                                
                                logger.info(f"Successfully processed SDF file: {filename}")
                                return data_file
                        break
            
            logger.info(f"Imported SDF file: {filename}")
            return sdf_file
            
        except Exception as e:
            logger.error(f"Failed to import SDF file: {e}")
            raise
    
    def import_pdb_organic(self, pdb_content: str, filename: str) -> Path:
        """
        Import PDB format file for organic molecules.
        
        Args:
            pdb_content: PDB file content as string
            filename: Name for the output file
            
        Returns:
            Path to the processed structure file
        """
        try:
            from .utils.molecular_utils import molecular_utils
            from .utils.forcefield_utils import forcefield_utils
            
            # Save original PDB file
            pdb_file = self.input_dir / f"{filename}.pdb"
            with open(pdb_file, 'w') as f:
                f.write(pdb_content)
            
            # Try to process with RDKit if available
            if molecular_utils.rdkit_available:
                from rdkit import Chem
                
                # Read molecule from PDB
                mol = Chem.MolFromPDBBlock(pdb_content, removeHs=False)
                if mol is not None:
                    # Sanitize molecule
                    mol = molecular_utils.sanitize_molecule(mol)
                    if mol is not None:
                        # Assign GAFF atom types
                        atom_types = forcefield_utils.assign_gaff_atom_types(mol)
                        if atom_types:
                            # Calculate partial charges
                            charges = forcefield_utils.calculate_partial_charges(mol, method="gasteiger")
                            if not charges:
                                charges = forcefield_utils.get_gaff_charge_estimates(mol, atom_types)
                            
                            # Generate topology
                            topology = forcefield_utils.generate_topology(mol, atom_types)
                            
                            # Create LAMMPS data file
                            data_content = forcefield_utils.create_lammps_data_file(mol, atom_types, topology, charges)
                            if data_content:
                                data_file = self.input_dir / f"{filename}.data"
                                with open(data_file, 'w') as f:
                                    f.write(data_content)
                                
                                logger.info(f"Successfully processed PDB file: {filename}")
                                return data_file
            
            logger.info(f"Imported PDB file: {filename}")
            return pdb_file
            
        except Exception as e:
            logger.error(f"Failed to import PDB file: {e}")
            raise
    
    def create_liquid_box_file(
        self,
        molecules: List[Dict[str, Any]],
        target_density: float = 1.0,
        box_type: str = "cubic"
    ) -> Path:
        """
        Create a liquid box data file for multi-component systems.
        
        Args:
            molecules: List of molecule dictionaries with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            box_type: Type of simulation box ('cubic', 'orthorhombic')
            
        Returns:
            Path to the liquid box file
        """
        try:
            from .utils.molecular_utils import molecular_utils
            from .utils.forcefield_utils import forcefield_utils
            import random
            import math
            
            if not molecules:
                raise ValueError("No molecules provided for liquid box creation")
            
            # Process each molecule type
            processed_molecules = []
            total_mass = 0.0
            total_molecules = 0
            
            for mol_info in molecules:
                smiles = mol_info.get('smiles', '')
                count = mol_info.get('count', 1)
                name = mol_info.get('name', f'mol_{len(processed_molecules)}')
                
                if not smiles:
                    raise ValueError(f"No SMILES provided for molecule: {name}")
                
                # Convert SMILES to 3D structure
                mol = molecular_utils.smiles_to_3d(smiles)
                if mol is None:
                    raise ValueError(f"Failed to convert SMILES for {name}: {smiles}")
                
                # Calculate molecular properties
                properties = molecular_utils.calculate_molecular_properties(mol)
                mol_mass = properties.get('molecular_weight', 0.0)
                
                # Assign atom types and generate topology
                atom_types = forcefield_utils.assign_gaff_atom_types(mol)
                topology = forcefield_utils.generate_topology(mol, atom_types)
                
                # Calculate partial charges
                charges = forcefield_utils.calculate_partial_charges(mol, method="gasteiger")
                if not charges:
                    logger.warning(f"Failed to calculate partial charges for {name}, using GAFF estimates")
                    charges = forcefield_utils.get_gaff_charge_estimates(mol, atom_types)
                
                # Validate charges
                charges_valid, charge_issues = forcefield_utils.validate_charges(charges, mol)
                if not charges_valid:
                    logger.warning(f"Charge validation issues for {name}: {charge_issues}")
                
                processed_molecules.append({
                    'name': name,
                    'smiles': smiles,
                    'count': count,
                    'mol': mol,
                    'atom_types': atom_types,
                    'charges': charges,
                    'topology': topology,
                    'mass': mol_mass
                })
                
                total_mass += mol_mass * count
                total_molecules += count
            
            # Calculate box size based on target density
            # density = mass / volume, so volume = mass / density
            avogadro = 6.022e23
            total_mass_g = total_mass / avogadro  # Convert from amu to grams
            volume_cm3 = total_mass_g / target_density
            volume_angstrom3 = volume_cm3 * 1e24  # Convert cm³ to Å³
            
            if box_type == "cubic":
                box_size = volume_angstrom3 ** (1/3)
            else:
                # For simplicity, use cubic box even for orthorhombic
                box_size = volume_angstrom3 ** (1/3)
            
            # Create liquid box content
            total_atoms = 0
            total_bonds = 0
            total_angles = 0
            total_dihedrals = 0
            
            # Count totals
            for mol_info in processed_molecules:
                mol = mol_info['mol']
                count = mol_info['count']
                topology = mol_info['topology']
                
                total_atoms += mol.GetNumAtoms() * count
                total_bonds += len(topology['bonds']) * count
                total_angles += len(topology['angles']) * count
                total_dihedrals += len(topology['dihedrals']) * count
            
            # Get all unique atom types
            all_atom_types = set()
            for mol_info in processed_molecules:
                all_atom_types.update(mol_info['atom_types'].values())
            
            unique_atom_types = list(all_atom_types)
            type_mapping = {atype: i+1 for i, atype in enumerate(unique_atom_types)}
            
            # Get unique topology types across all molecules
            topology_types = self._get_unique_topology_types_multi(processed_molecules)
            
            # Create data file content
            lines = [
                "LAMMPS data file - Liquid box generated by MCP LAMMPS",
                "",
                f"{total_atoms} atoms",
                f"{total_bonds} bonds",
                f"{total_angles} angles",
                f"{total_dihedrals} dihedrals",
                "0 impropers",
                "",
                f"{len(unique_atom_types)} atom types",
                f"{len(topology_types['unique_bond_types'])} bond types",
                f"{len(topology_types['unique_angle_types'])} angle types",
                f"{len(topology_types['unique_dihedral_types'])} dihedral types",
                "0 improper types",
                "",
                f"{-box_size/2:.6f} {box_size/2:.6f} xlo xhi",
                f"{-box_size/2:.6f} {box_size/2:.6f} ylo yhi",
                f"{-box_size/2:.6f} {box_size/2:.6f} zlo zhi",
                "",
                "Masses",
                ""
            ]
            
            # Add masses
            for i, atype in enumerate(unique_atom_types):
                mass = forcefield_utils._get_atomic_mass(atype)
                lines.append(f"{i+1} {mass:.4f}")
            
            lines.extend(["", "Atoms", ""])
            
            # Place molecules randomly in the box and track atom offsets
            atom_id = 1
            molecule_id = 1
            atom_offset_map = {}  # Maps molecule instance to starting atom index
            
            for mol_info in processed_molecules:
                mol = mol_info['mol']
                count = mol_info['count']
                atom_types = mol_info['atom_types']
                name = mol_info['name']
                
                for mol_instance in range(count):
                    # Track atom offset for this molecule instance
                    instance_key = f"{name}_{mol_instance}"
                    atom_offset_map[instance_key] = atom_id - 1  # 0-based offset for topology
                    
                    # Random position for molecule center
                    center_x = random.uniform(-box_size/3, box_size/3)
                    center_y = random.uniform(-box_size/3, box_size/3)
                    center_z = random.uniform(-box_size/3, box_size/3)
                    
                    # Random rotation
                    rotation = random.uniform(0, 2*math.pi)
                    
                    conf = mol.GetConformer()
                    for atom_idx in range(mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(atom_idx)
                        pos = conf.GetAtomPosition(atom_idx)
                        
                        # Apply rotation and translation
                        x = pos.x * math.cos(rotation) - pos.y * math.sin(rotation) + center_x
                        y = pos.x * math.sin(rotation) + pos.y * math.cos(rotation) + center_y
                        z = pos.z + center_z
                        
                        atype = atom_types[atom_idx]
                        type_id = type_mapping[atype]
                        charge = mol_info['charges'][atom_idx]  # Use calculated charges
                        
                        lines.append(f"{atom_id} {molecule_id} {type_id} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}")
                        atom_id += 1
                    
                    molecule_id += 1
            
            # Add topology sections if there are bonds/angles/dihedrals
            if total_bonds > 0:
                bonds_lines = self._write_bonds_section(
                    processed_molecules, 
                    topology_types['bond_type_mapping'], 
                    atom_offset_map
                )
                lines.extend(bonds_lines)
            
            if total_angles > 0:
                angles_lines = self._write_angles_section(
                    processed_molecules, 
                    topology_types['angle_type_mapping'], 
                    atom_offset_map
                )
                lines.extend(angles_lines)
            
            if total_dihedrals > 0:
                dihedrals_lines = self._write_dihedrals_section(
                    processed_molecules, 
                    topology_types['dihedral_type_mapping'], 
                    atom_offset_map
                )
                lines.extend(dihedrals_lines)
            
            # Save the file
            filename = f"liquid_box_{len(molecules)}components.data"
            file_path = self.input_dir / filename
            
            with open(file_path, 'w') as f:
                f.write("\n".join(lines))
            
            # Calculate total charge for validation
            total_charge = 0.0
            molecule_charge_info = []
            
            for mol_info in processed_molecules:
                mol_charges = mol_info['charges']
                mol_total_charge = sum(mol_charges.values()) * mol_info['count']
                total_charge += mol_total_charge
                
                molecule_charge_info.append({
                    'name': mol_info['name'],
                    'smiles': mol_info['smiles'],
                    'count': mol_info['count'],
                    'charge_per_molecule': sum(mol_charges.values()),
                    'total_charge_contribution': mol_total_charge,
                    'num_atoms_per_molecule': len(mol_charges)
                })
            
            # Save metadata
            metadata = {
                "molecules": molecules,
                "target_density": target_density,
                "calculated_box_size": box_size,
                "total_atoms": total_atoms,
                "total_molecules": total_molecules,
                "box_type": box_type,
                "charge_information": {
                    "total_system_charge": total_charge,
                    "molecule_charges": molecule_charge_info,
                    "charge_method": "gasteiger"
                }
            }
            
            metadata_file = file_path.with_suffix(".json")
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Created liquid box file with {total_molecules} molecules: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to create liquid box file: {e}")
            raise
    
    def assign_gaff_parameters(self, structure_file: Path) -> Dict[str, Any]:
        """
        Assign GAFF parameters to an existing structure file.
        
        Args:
            structure_file: Path to structure file
            
        Returns:
            Dictionary containing GAFF parameters and assignments
        """
        try:
            from .utils.molecular_utils import molecular_utils
            from .utils.forcefield_utils import forcefield_utils
            
            if not structure_file.exists():
                raise ValueError(f"Structure file not found: {structure_file}")
            
            # Try to read the structure file
            mol = None
            
            if structure_file.suffix.lower() == '.pdb':
                with open(structure_file, 'r') as f:
                    pdb_content = f.read()
                if molecular_utils.rdkit_available:
                    from rdkit import Chem
                    mol = Chem.MolFromPDBBlock(pdb_content)
            
            elif structure_file.suffix.lower() == '.sdf':
                if molecular_utils.rdkit_available:
                    from rdkit import Chem
                    mol = Chem.MolFromMolFile(str(structure_file))
            
            elif structure_file.suffix.lower() == '.mol2':
                # Try to convert using OpenBabel
                if molecular_utils.openbabel_available:
                    with open(structure_file, 'r') as f:
                        mol2_content = f.read()
                    sdf_content = molecular_utils.convert_format(mol2_content, "mol2", "sdf")
                    if sdf_content and molecular_utils.rdkit_available:
                        from rdkit import Chem
                        mol = Chem.MolFromMolBlock(sdf_content)
            
            if mol is None:
                raise ValueError(f"Could not read molecule from file: {structure_file}")
            
            # Sanitize molecule
            mol = molecular_utils.sanitize_molecule(mol)
            if mol is None:
                raise ValueError("Failed to sanitize molecule")
            
            # Assign GAFF atom types
            atom_types = forcefield_utils.assign_gaff_atom_types(mol)
            if not atom_types:
                raise ValueError("Failed to assign GAFF atom types")
            
            # Calculate partial charges
            charges = forcefield_utils.calculate_partial_charges(mol, method="gasteiger")
            if not charges:
                logger.warning("Failed to calculate partial charges, using GAFF estimates")
                charges = forcefield_utils.get_gaff_charge_estimates(mol, atom_types)
            
            # Validate charges
            charges_valid, charge_issues = forcefield_utils.validate_charges(charges, mol)
            if not charges_valid:
                logger.warning(f"Charge validation issues: {charge_issues}")
            
            # Generate topology
            topology = forcefield_utils.generate_topology(mol, atom_types)
            
            # Validate parameters
            is_valid, issues = forcefield_utils.validate_parameters(atom_types, topology)
            
            # Calculate molecular properties
            properties = molecular_utils.calculate_molecular_properties(mol)
            
            result = {
                "atom_types": atom_types,
                "charges": charges,
                "topology": topology,
                "properties": properties,
                "validation": {
                    "is_valid": is_valid and charges_valid,
                    "issues": issues + charge_issues
                },
                "statistics": {
                    "num_atoms": mol.GetNumAtoms(),
                    "num_bonds": len(topology["bonds"]),
                    "num_angles": len(topology["angles"]),
                    "num_dihedrals": len(topology["dihedrals"]),
                    "total_charge": sum(charges.values()) if charges else 0.0
                }
            }
            
            logger.info(f"Successfully assigned GAFF parameters to: {structure_file.name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to assign GAFF parameters: {e}")
            raise
    
    def _get_unique_topology_types_multi(self, processed_molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get unique bond, angle, and dihedral types across all molecule types.
        
        Args:
            processed_molecules: List of processed molecule dictionaries
            
        Returns:
            Dictionary containing unique types and their mappings
        """
        try:
            # Collect all unique bond types
            all_bond_types = set()
            all_angle_types = set()
            all_dihedral_types = set()
            
            for mol_info in processed_molecules:
                topology = mol_info['topology']
                
                # Collect bond types
                for bond_info in topology["bonds"]:
                    bond_type = tuple(sorted(bond_info["types"]))
                    all_bond_types.add(bond_type)
                
                # Collect angle types
                for angle_info in topology["angles"]:
                    angle_type = tuple(angle_info["types"])
                    all_angle_types.add(angle_type)
                
                # Collect dihedral types
                for dihedral_info in topology["dihedrals"]:
                    dihedral_type = tuple(dihedral_info["types"])
                    all_dihedral_types.add(dihedral_type)
            
            # Create type mappings
            unique_bond_types = list(all_bond_types)
            unique_angle_types = list(all_angle_types)
            unique_dihedral_types = list(all_dihedral_types)
            
            bond_type_mapping = {bond_type: i+1 for i, bond_type in enumerate(unique_bond_types)}
            angle_type_mapping = {angle_type: i+1 for i, angle_type in enumerate(unique_angle_types)}
            dihedral_type_mapping = {dihedral_type: i+1 for i, dihedral_type in enumerate(unique_dihedral_types)}
            
            return {
                "unique_bond_types": unique_bond_types,
                "unique_angle_types": unique_angle_types,
                "unique_dihedral_types": unique_dihedral_types,
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
    
    def _write_bonds_section(
        self,
        processed_molecules: List[Dict[str, Any]],
        bond_type_mapping: Dict[tuple, int],
        atom_offset_map: Dict[str, int]
    ) -> List[str]:
        """
        Write bonds section with correct global atom indices.
        
        Args:
            processed_molecules: List of processed molecule dictionaries
            bond_type_mapping: Mapping from bond types to type IDs
            atom_offset_map: Mapping from molecule instance to atom offset
            
        Returns:
            List of bond section lines
        """
        try:
            lines = ["", "Bonds", ""]
            bond_id = 1
            
            for mol_info in processed_molecules:
                mol = mol_info['mol']
                count = mol_info['count']
                topology = mol_info['topology']
                name = mol_info['name']
                
                for mol_instance in range(count):
                    instance_key = f"{name}_{mol_instance}"
                    atom_offset = atom_offset_map[instance_key]
                    
                    for bond_info in topology["bonds"]:
                        atoms = bond_info["atoms"]
                        bond_type_key = tuple(sorted(bond_info["types"]))
                        bond_type_id = bond_type_mapping[bond_type_key]
                        
                        # Convert local atom indices to global indices
                        global_atom1 = atoms[0] + atom_offset + 1  # +1 for 1-based indexing
                        global_atom2 = atoms[1] + atom_offset + 1
                        
                        lines.append(f"{bond_id} {bond_type_id} {global_atom1} {global_atom2}")
                        bond_id += 1
            
            return lines
            
        except Exception as e:
            logger.error(f"Error writing bonds section: {e}")
            return ["", "Bonds", ""]
    
    def _write_angles_section(
        self,
        processed_molecules: List[Dict[str, Any]],
        angle_type_mapping: Dict[tuple, int],
        atom_offset_map: Dict[str, int]
    ) -> List[str]:
        """
        Write angles section with correct global atom indices.
        
        Args:
            processed_molecules: List of processed molecule dictionaries
            angle_type_mapping: Mapping from angle types to type IDs
            atom_offset_map: Mapping from molecule instance to atom offset
            
        Returns:
            List of angle section lines
        """
        try:
            lines = ["", "Angles", ""]
            angle_id = 1
            
            for mol_info in processed_molecules:
                mol = mol_info['mol']
                count = mol_info['count']
                topology = mol_info['topology']
                name = mol_info['name']
                
                for mol_instance in range(count):
                    instance_key = f"{name}_{mol_instance}"
                    atom_offset = atom_offset_map[instance_key]
                    
                    for angle_info in topology["angles"]:
                        atoms = angle_info["atoms"]
                        angle_type_key = tuple(angle_info["types"])
                        angle_type_id = angle_type_mapping[angle_type_key]
                        
                        # Convert local atom indices to global indices
                        global_atom1 = atoms[0] + atom_offset + 1
                        global_atom2 = atoms[1] + atom_offset + 1
                        global_atom3 = atoms[2] + atom_offset + 1
                        
                        lines.append(f"{angle_id} {angle_type_id} {global_atom1} {global_atom2} {global_atom3}")
                        angle_id += 1
            
            return lines
            
        except Exception as e:
            logger.error(f"Error writing angles section: {e}")
            return ["", "Angles", ""]
    
    def _write_dihedrals_section(
        self,
        processed_molecules: List[Dict[str, Any]],
        dihedral_type_mapping: Dict[tuple, int],
        atom_offset_map: Dict[str, int]
    ) -> List[str]:
        """
        Write dihedrals section with correct global atom indices.
        
        Args:
            processed_molecules: List of processed molecule dictionaries
            dihedral_type_mapping: Mapping from dihedral types to type IDs
            atom_offset_map: Mapping from molecule instance to atom offset
            
        Returns:
            List of dihedral section lines
        """
        try:
            lines = ["", "Dihedrals", ""]
            dihedral_id = 1
            
            for mol_info in processed_molecules:
                mol = mol_info['mol']
                count = mol_info['count']
                topology = mol_info['topology']
                name = mol_info['name']
                
                for mol_instance in range(count):
                    instance_key = f"{name}_{mol_instance}"
                    atom_offset = atom_offset_map[instance_key]
                    
                    for dihedral_info in topology["dihedrals"]:
                        atoms = dihedral_info["atoms"]
                        dihedral_type_key = tuple(dihedral_info["types"])
                        dihedral_type_id = dihedral_type_mapping[dihedral_type_key]
                        
                        # Convert local atom indices to global indices
                        global_atom1 = atoms[0] + atom_offset + 1
                        global_atom2 = atoms[1] + atom_offset + 1
                        global_atom3 = atoms[2] + atom_offset + 1
                        global_atom4 = atoms[3] + atom_offset + 1
                        
                        lines.append(f"{dihedral_id} {dihedral_type_id} {global_atom1} {global_atom2} {global_atom3} {global_atom4}")
                        dihedral_id += 1
            
            return lines
            
        except Exception as e:
            logger.error(f"Error writing dihedrals section: {e}")
            return ["", "Dihedrals", ""] 