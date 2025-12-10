"""
Organic Tools - MCP tools for organic molecule simulations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp.server import Context

logger = logging.getLogger(__name__)


def register_organic_tools(server: Any, lammps_server: Any) -> None:
    """
    Register organic simulation tools with the MCP server.
    
    Args:
        server: MCP server instance
        lammps_server: LAMMPS server instance
    """
    
    @server.tool(
        name="import_from_smiles",
        title="Import from SMILES",
        description="Convert SMILES to 3D structure with GAFF parameters"
    )
    async def import_from_smiles(
        ctx: Context,
        smiles: str,
        molecule_name: str,
        optimize_geometry: bool = True
    ) -> Dict[str, Any]:
        """
        Convert SMILES string to 3D structure with GAFF parameters.
        
        Args:
            smiles: SMILES string representation of the molecule
            molecule_name: Name for the molecule
            optimize_geometry: Whether to optimize 3D geometry
            
        Returns:
            Molecule information and file paths
        """
        try:
            # Import structure from SMILES
            structure_file = lammps_server.data_handler.import_smiles_structure(
                smiles=smiles,
                molecule_name=molecule_name,
                optimize_geometry=optimize_geometry
            )
            
            # Get file info
            file_info = lammps_server.data_handler.get_file_info(structure_file)
            
            # Load metadata if available
            metadata_file = structure_file.with_suffix(".json")
            metadata = {}
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            ctx.info(f"Successfully imported molecule from SMILES: {molecule_name}")
            
            return {
                "molecule_name": molecule_name,
                "smiles": smiles,
                "structure_file": str(structure_file),
                "file_info": file_info,
                "metadata": metadata,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to import from SMILES: {e}")
            raise
    
    @server.tool(
        name="import_molecular_file",
        title="Import Molecular File",
        description="Universal importer for molecular files (MOL2, SDF, PDB auto-detection)"
    )
    async def import_molecular_file(
        ctx: Context,
        file_content: str,
        filename: str,
        file_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import molecular file with automatic format detection.
        
        Args:
            file_content: Content of the molecular file
            filename: Name for the output file
            file_format: File format (mol2, sdf, pdb) - auto-detected if None
            
        Returns:
            Import result information
        """
        try:
            # Auto-detect format if not provided
            if file_format is None:
                if filename.lower().endswith('.mol2'):
                    file_format = 'mol2'
                elif filename.lower().endswith('.sdf'):
                    file_format = 'sdf'
                elif filename.lower().endswith('.pdb'):
                    file_format = 'pdb'
                else:
                    # Try to detect from content
                    if '@<TRIPOS>MOLECULE' in file_content:
                        file_format = 'mol2'
                    elif file_content.strip().startswith('M  END') or '$$$$' in file_content:
                        file_format = 'sdf'
                    elif 'ATOM  ' in file_content or 'HETATM' in file_content:
                        file_format = 'pdb'
                    else:
                        raise ValueError("Could not auto-detect file format")
            
            # Import based on format
            if file_format.lower() == 'mol2':
                structure_file = lammps_server.data_handler.import_mol2_file(file_content, filename)
            elif file_format.lower() == 'sdf':
                structure_file = lammps_server.data_handler.import_sdf_file(file_content, filename)
            elif file_format.lower() == 'pdb':
                structure_file = lammps_server.data_handler.import_pdb_organic(file_content, filename)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Get file info
            file_info = lammps_server.data_handler.get_file_info(structure_file)
            
            ctx.info(f"Successfully imported molecular file: {filename} ({file_format})")
            
            return {
                "filename": filename,
                "detected_format": file_format,
                "structure_file": str(structure_file),
                "file_info": file_info,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to import molecular file: {e}")
            raise
    
    @server.tool(
        name="build_liquid_box",
        title="Build Liquid Box",
        description="Create multi-component liquid systems with target density"
    )
    async def build_liquid_box(
        ctx: Context,
        molecules: List[Dict[str, Any]],
        target_density: float = 1.0,
        box_type: str = "cubic"
    ) -> Dict[str, Any]:
        """
        Build a liquid box with multiple molecule types.
        
        Args:
            molecules: List of molecule dicts with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            box_type: Type of simulation box ('cubic', 'orthorhombic')
            
        Returns:
            Liquid box information
        """
        try:
            # Validate input
            if not molecules:
                raise ValueError("No molecules provided")
            
            for i, mol in enumerate(molecules):
                if 'smiles' not in mol:
                    raise ValueError(f"Molecule {i} missing 'smiles' field")
                if 'count' not in mol:
                    mol['count'] = 1
                if 'name' not in mol:
                    mol['name'] = f'molecule_{i}'
            
            # Create liquid box
            box_file = lammps_server.data_handler.create_liquid_box_file(
                molecules=molecules,
                target_density=target_density,
                box_type=box_type
            )
            
            # Get file info
            file_info = lammps_server.data_handler.get_file_info(box_file)
            
            # Load metadata
            metadata_file = box_file.with_suffix(".json")
            metadata = {}
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            ctx.info(f"Successfully created liquid box with {len(molecules)} component types")
            
            return {
                "liquid_box_file": str(box_file),
                "molecules": molecules,
                "target_density": target_density,
                "box_type": box_type,
                "file_info": file_info,
                "metadata": metadata,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to build liquid box: {e}")
            raise
    
    @server.tool(
        name="create_organic_simulation",
        title="Create Organic Simulation",
        description="Complete organic liquid simulation setup"
    )
    async def create_organic_simulation(
        ctx: Context,
        name: str,
        molecules: List[Dict[str, Any]],
        target_density: float = 1.0,
        force_field: str = "gaff",
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 0.001,
        equilibration_steps: int = 5000,
        production_steps: int = 50000
    ) -> Dict[str, Any]:
        """
        Create a complete organic liquid simulation setup.
        
        Args:
            name: Name of the simulation
            molecules: List of molecule dicts with 'smiles', 'count', 'name'
            target_density: Target density in g/cm³
            force_field: Force field to use (gaff, amber)
            temperature: Simulation temperature (K)
            pressure: Simulation pressure (atm)
            timestep: Simulation timestep (ps)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            
        Returns:
            Simulation information
        """
        try:
            # Create liquid box
            box_file = lammps_server.data_handler.create_liquid_box_file(
                molecules=molecules,
                target_density=target_density
            )

            # Load metadata from liquid box file to extract topology information
            metadata_file = box_file.with_suffix(".json")
            atom_types = None
            topology = None

            if metadata_file.exists():
                import json
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Extract atom types and topology from the first molecule (assuming single component for now)
                    if "molecule_details" in metadata and metadata["molecule_details"]:
                        mol_detail = metadata["molecule_details"][0]  # Use first molecule for single component
                        atom_types = mol_detail.get("atom_types")
                        topology = mol_detail.get("topology")

                        ctx.info(f"Extracted topology from metadata: {len(topology.get('bonds', []))} bonds, "
                                f"{len(topology.get('angles', []))} angles, {len(topology.get('dihedrals', []))} dihedrals")
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_file}: {e}")

            # Create simulation configuration
            config = {
                "name": name,
                "type": "organic_liquid_simulation",
                "molecules": molecules,
                "target_density": target_density,
                "force_field": force_field,
                "temperature": temperature,
                "pressure": pressure,
                "timestep": timestep,
                "equilibration_steps": equilibration_steps,
                "production_steps": production_steps,
                "structure_file": str(box_file)
            }

            # Create simulation in manager
            simulation_id = lammps_server.simulation_manager.create_simulation(name, config)

            # Create LAMMPS script for liquid simulation with topology information
            script_content = lammps_server.lammps_interface.create_liquid_script(
                structure_file=str(box_file),
                force_field=force_field,
                temperature=temperature,
                pressure=pressure,
                timestep=timestep,
                equilibration_steps=equilibration_steps,
                production_steps=production_steps,
                density_target=target_density,
                atom_types=atom_types,
                topology=topology
            )
            
            # Save script file
            script_file = lammps_server.data_handler.save_script_file(
                f"{simulation_id}_organic_liquid.lmp",
                script_content
            )
            
            config["script_file"] = str(script_file)
            
            # Update simulation with script file info
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if sim:
                sim.config = config
                sim.save_state()
            
            ctx.info(f"Created organic liquid simulation '{name}' with ID: {simulation_id}")
            
            return {
                "simulation_id": simulation_id,
                "name": name,
                "type": "organic_liquid_simulation",
                "status": "created",
                "config": config,
                "liquid_box_file": str(box_file),
                "script_file": str(script_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to create organic simulation: {e}")
            raise
    
    @server.tool(
        name="setup_amber_forcefield",
        title="Setup AMBER Force Field",
        description="Configure AMBER/GAFF parameters for molecules"
    )
    async def setup_amber_forcefield(
        ctx: Context,
        structure_file: str,
        force_field_variant: str = "gaff"
    ) -> Dict[str, Any]:
        """
        Setup AMBER/GAFF force field parameters for a molecular structure.
        
        Args:
            structure_file: Path to structure file
            force_field_variant: Force field variant (gaff, gaff2, amber)
            
        Returns:
            Force field parameter information
        """
        try:
            structure_path = Path(structure_file)
            if not structure_path.exists():
                raise ValueError(f"Structure file not found: {structure_file}")
            
            # Assign comprehensive GAFF parameters using new system
            ff_result = lammps_server.data_handler.assign_gaff_parameters(structure_path)
            
            
            # Save force field parameters
            ff_file = structure_path.with_suffix(f".{force_field_variant}")
            with open(ff_file, 'w') as f:
                import json
                json.dump(ff_result, f, indent=2, default=str)
            
            ctx.info(f"Setup comprehensive {force_field_variant.upper()} parameters for: {structure_path.name}")
            
            return {
                "structure_file": structure_file,
                "force_field_variant": force_field_variant,
                "parameter_file": str(ff_file),
                "atom_types": ff_result.get("atom_types", {}),
                "statistics": ff_result.get("statistics", {}),
                "validation": ff_result.get("validation", {}),
                "parameter_database_stats": {},
                "comprehensive_gaff": True,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to setup AMBER force field: {e}")
            raise
    
    @server.tool(
        name="create_mixture_simulation",
        title="Create Mixture Simulation",
        description="Multi-component liquid mixture setup"
    )
    async def create_mixture_simulation(
        ctx: Context,
        name: str,
        components: List[Dict[str, Any]],
        mixture_type: str = "homogeneous",
        target_density: float = 1.0,
        force_field: str = "gaff",
        temperature: float = 300.0,
        pressure: float = 1.0,
        equilibration_steps: int = 10000,
        production_steps: int = 100000
    ) -> Dict[str, Any]:
        """
        Create a multi-component liquid mixture simulation.
        
        Args:
            name: Name of the simulation
            components: List of component dicts with 'smiles', 'mole_fraction', 'name'
            mixture_type: Type of mixture (homogeneous, layered)
            target_density: Target density in g/cm³
            force_field: Force field to use
            temperature: Simulation temperature (K)
            pressure: Simulation pressure (atm)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            
        Returns:
            Mixture simulation information
        """
        try:
            # Validate components
            if not components:
                raise ValueError("No components provided")
            
            total_mole_fraction = sum(comp.get('mole_fraction', 0) for comp in components)
            if abs(total_mole_fraction - 1.0) > 0.01:
                raise ValueError(f"Mole fractions must sum to 1.0, got {total_mole_fraction}")
            
            # Convert mole fractions to molecule counts (assuming 1000 total molecules)
            total_molecules = 1000
            molecules = []
            
            for comp in components:
                mole_frac = comp.get('mole_fraction', 0)
                count = int(mole_frac * total_molecules)
                if count > 0:
                    molecules.append({
                        'smiles': comp['smiles'],
                        'count': count,
                        'name': comp.get('name', f"comp_{len(molecules)}")
                    })
            
            # Create mixture box
            if mixture_type == "homogeneous":
                box_file = lammps_server.data_handler.create_liquid_box_file(
                    molecules=molecules,
                    target_density=target_density
                )
            else:
                # For layered mixtures, create separate layers (simplified implementation)
                box_file = lammps_server.data_handler.create_liquid_box_file(
                    molecules=molecules,
                    target_density=target_density
                )
            
            # Create simulation configuration
            config = {
                "name": name,
                "type": "mixture_simulation",
                "components": components,
                "molecules": molecules,
                "mixture_type": mixture_type,
                "target_density": target_density,
                "force_field": force_field,
                "temperature": temperature,
                "pressure": pressure,
                "equilibration_steps": equilibration_steps,
                "production_steps": production_steps,
                "structure_file": str(box_file)
            }
            
            # Create simulation
            simulation_id = lammps_server.simulation_manager.create_simulation(name, config)
            
            # Create enhanced liquid script for mixtures
            script_content = lammps_server.lammps_interface.create_liquid_script(
                structure_file=str(box_file),
                force_field=force_field,
                temperature=temperature,
                pressure=pressure,
                timestep=0.0005,  # Smaller timestep for stability
                equilibration_steps=equilibration_steps,
                production_steps=production_steps,
                density_target=target_density
            )
            
            # Save script file
            script_file = lammps_server.data_handler.save_script_file(
                f"{simulation_id}_mixture.lmp",
                script_content
            )
            
            config["script_file"] = str(script_file)
            
            # Update simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if sim:
                sim.config = config
                sim.save_state()
            
            ctx.info(f"Created mixture simulation '{name}' with {len(components)} components")
            
            return {
                "simulation_id": simulation_id,
                "name": name,
                "type": "mixture_simulation",
                "status": "created",
                "config": config,
                "components": components,
                "total_molecules": sum(mol['count'] for mol in molecules),
                "mixture_box_file": str(box_file),
                "script_file": str(script_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to create mixture simulation: {e}")
            raise
