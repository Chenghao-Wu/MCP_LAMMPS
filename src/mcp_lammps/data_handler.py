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
        optimize_geometry: bool = True,
        box_size: float = 20.0

    ) -> Path:
        """
        Import molecular structure from SMILES string using OpenFF.
        
        Args:
            smiles: SMILES string representation
            molecule_name: Name for the molecule
            optimize_geometry: Whether to optimize 3D geometry
            box_size: Simulation box size
        Returns:
            Path to the created structure file
        """
        try:
            from .utils.openff_utils import openff_forcefield
            
            # Create OpenFF molecule from SMILES with 3D coordinates
            molecule = openff_forcefield.from_smiles(
                smiles, molecule_name, optimize=optimize_geometry
            )
            
            # Assign charges using NAGL
            openff_forcefield.assign_charges(molecule)
            from openff.units import unit
            box_vectors = np.array([
                [box_size, 0, 0],
                [0, box_size, 0],
                [0, 0, box_size]
            ]) * unit.angstrom

            # Create single-molecule topology
            topology = openff_forcefield.create_topology([molecule])
            
            # Create Interchange system
            interchange = openff_forcefield.create_interchange(topology, [molecule])
            
            interchange.box = box_vectors
            # Export to LAMMPS files
            output_prefix = self.input_dir / f"{molecule_name}_from_smiles"
            data_file, script_file = openff_forcefield.to_lammps(interchange, str(output_prefix))
            
            # Save metadata
            metadata = {
                "smiles": smiles,
                "molecule_name": molecule_name,
                "charges": {i: float(c.m_as(unit.elementary_charge)) for i, c in enumerate(molecule.partial_charges)},
                "n_atoms": molecule.n_atoms,
                "n_bonds": molecule.n_bonds,
                "force_field": "openff-sage-2.2.0",
                "charge_method": "am1bcc",
                "molecular_weight": sum([atom.mass.m_as(unit.dalton) for atom in molecule.atoms])
            }
            
            metadata_file = data_file.with_suffix(".json")
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Successfully imported SMILES structure using OpenFF: {molecule_name}")
            return data_file
            
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
        Import SDF format file using OpenFF.
        
        Args:
            sdf_content: SDF file content as string
            filename: Name for the output file
            
        Returns:
            Path to the processed structure file
        """
        try:
            from .utils.openff_utils import openff_forcefield
            from openff.toolkit import Molecule
            
            # Save original SDF file
            sdf_file = self.input_dir / f"{filename}.sdf"
            with open(sdf_file, 'w') as f:
                f.write(sdf_content)
            
            try:
                # Load molecule from SDF using OpenFF
                molecule = Molecule.from_file(str(sdf_file), file_format='sdf')
                molecule.name = filename
                
                # Assign charges using NAGL
                openff_forcefield.assign_charges(molecule)
                
                # Create topology
                topology = openff_forcefield.create_topology([molecule])
                
                # Create Interchange system
                interchange = openff_forcefield.create_interchange(topology, [molecule])
                
                # Export to LAMMPS files
                output_prefix = self.input_dir / f"{filename}_from_sdf"
                data_file, script_file = openff_forcefield.to_lammps(interchange, str(output_prefix))
                
                logger.info(f"Successfully processed SDF file with OpenFF: {filename}")
                return data_file
                
            except Exception as e:
                logger.warning(f"Could not process SDF with OpenFF: {e}. Returning original file.")
                return sdf_file
            
        except Exception as e:
            logger.error(f"Failed to import SDF file: {e}")
            raise
    
    def import_pdb_organic(self, pdb_content: str, filename: str) -> Path:
        """
        Import PDB format file for organic molecules using OpenFF.
        
        Note: PDB files require molecule templates with bond orders.
        For simple cases, the file is saved but may need SMILES-based processing.
        
        Args:
            pdb_content: PDB file content as string
            filename: Name for the output file
            
        Returns:
            Path to the processed structure file
        """
        try:
            # Save original PDB file
            pdb_file = self.input_dir / f"{filename}.pdb"
            with open(pdb_file, 'w') as f:
                f.write(pdb_content)
            
            # Note: OpenFF requires molecule templates for PDB loading
            # For now, we save the PDB file and recommend using SMILES-based import
            # or providing molecule templates via from_pdb() method
            logger.info(f"Imported PDB file: {filename}")
            logger.info("Note: For force field assignment, use import_from_smiles with the molecule's SMILES string")
            
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
        Create a liquid box data file for multi-component systems using OpenFF.
        
        Args:
            molecules: List of molecule dictionaries with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            box_type: Type of simulation box ('cubic', 'orthorhombic')
            
        Returns:
            Path to the liquid box file
        """
        try:
            from .utils.openff_utils import openff_forcefield
            
            if not molecules:
                raise ValueError("No molecules provided for liquid box creation")
            
            # Use OpenFF to build the complete liquid box system
            output_prefix = self.input_dir / f"liquid_box_{len(molecules)}components"
            interchange, data_file, script_file = openff_forcefield.build_liquid_box(
                molecule_specs=molecules,
                target_density=target_density,
                box_type=box_type,
                output_prefix=str(output_prefix)
            )
            
            # Get system info for metadata
            system_info = openff_forcefield.get_system_info(interchange)
            
            box_size = system_info['box_vectors'][0][0] if system_info['box_vectors'] is not None else 0.0
            # rescale box size to be a multiple of 2 to make sure the box size is large enough for the molecules to be separated
            
            # Save simplified metadata
            metadata = {
                "molecules": molecules,
                "target_density": target_density,
                "calculated_box_size": box_size,
                "total_atoms": system_info['n_atoms'],
                "total_molecules": system_info['n_molecules'],
                "box_type": box_type,
                "force_field": "openff-sage-2.2.0",
                "charge_method": "am1bcc",
                "interchange_info": system_info
            }
            
            metadata_file = data_file.with_suffix(".json")
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Created liquid box file using OpenFF with {system_info['n_molecules']} molecules")
            
            return data_file
            
        except Exception as e:
            logger.error(f"Failed to create liquid box file: {e}")
            raise
    
    # REMOVED: assign_gaff_parameters() - OpenFF Interchange handles all force field assignment
    # REMOVED: _get_unique_topology_types_multi() - No longer needed with OpenFF
    
    # REMOVED: _write_bonds_section(), _write_angles_section(), _write_dihedrals_section()
    # These are no longer needed as OpenFF Interchange handles all topology file generation