"""
Setup Tools - Tools for setting up LAMMPS simulations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp.server import Context

logger = logging.getLogger(__name__)


def register_setup_tools(server: Any, lammps_server: Any) -> None:
    """
    Register setup tools with the MCP server.
    
    Args:
        server: MCP server instance
        lammps_server: LAMMPS server instance
    """
    
    @server.tool(
        name="create_simulation",
        title="Create Simulation",
        description="Create a new LAMMPS simulation with basic parameters"
    )
    async def create_simulation(
        ctx: Context,
        name: str,
        structure_file: Optional[str] = None,
        force_field: str = "lj/cut",
        temperature: float = 300.0,
        pressure: float = 1.0,
        timestep: float = 0.001,
        equilibration_steps: int = 1000,
        production_steps: int = 10000
    ) -> Dict[str, Any]:
        """
        Create a new LAMMPS simulation.
        
        Args:
            name: Name of the simulation
            structure_file: Path to structure file (optional)
            force_field: Force field to use
            temperature: Simulation temperature (K)
            pressure: Simulation pressure (atm)
            timestep: Simulation timestep (ps)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            
        Returns:
            Simulation information
        """
        try:
            # Create simulation configuration
            config = {
                "name": name,
                "structure_file": structure_file,
                "force_field": force_field,
                "temperature": temperature,
                "pressure": pressure,
                "timestep": timestep,
                "equilibration_steps": equilibration_steps,
                "production_steps": production_steps
            }
            
            # Create simulation in manager
            simulation_id = lammps_server.simulation_manager.create_simulation(name, config)
            
            # Create LAMMPS script if structure file is provided
            if structure_file:
                # Use liquid script generator with best practice protocol
                script_content = lammps_server.lammps_interface.create_liquid_script(
                    structure_file=structure_file,
                    force_field=force_field,
                    temperature=temperature,
                    pressure=pressure,
                    timestep=timestep,
                    equilibration_steps=equilibration_steps,
                    production_steps=production_steps
                )
                
                # Save script file
                script_file = lammps_server.data_handler.save_script_file(
                    f"{simulation_id}_script.lmp",
                    script_content
                )
                
                config["script_file"] = str(script_file)
            
            # Update simulation with script file info
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if sim:
                sim.config = config
                sim.save_state()
            
            ctx.info(f"Created simulation '{name}' with ID: {simulation_id}")
            
            return {
                "simulation_id": simulation_id,
                "name": name,
                "status": "created",
                "config": config
            }
            
        except Exception as e:
            logger.error(f"Failed to create simulation: {e}")
            raise
    
    @server.tool(
        name="load_structure",
        title="Load Structure",
        description="Load a molecular structure from a file"
    )
    async def load_structure(
        ctx: Context,
        filename: str,
        content: str,
        file_type: str = "data"
    ) -> Dict[str, Any]:
        """
        Load a molecular structure from a file.
        
        Args:
            filename: Name of the structure file
            content: File content
            file_type: Type of structure file (data, xyz, pdb, etc.)
            
        Returns:
            File information
        """
        try:
            # Save structure file
            file_path = lammps_server.data_handler.save_structure_file(
                filename, content, file_type
            )
            
            # Get file info
            file_info = lammps_server.data_handler.get_file_info(file_path)
            
            ctx.info(f"Loaded structure file: {filename}")
            
            return {
                "filename": filename,
                "file_path": str(file_path),
                "file_type": file_type,
                "file_info": file_info
            }
            
        except Exception as e:
            logger.error(f"Failed to load structure: {e}")
            raise
    
    @server.tool(
        name="list_simulations",
        title="List Simulations",
        description="List all available simulations"
    )
    async def list_simulations(ctx: Context) -> Dict[str, Any]:
        """
        List all available simulations.
        
        Returns:
            List of simulations
        """
        try:
            simulations = lammps_server.simulation_manager.list_simulations()
            
            ctx.info(f"Found {len(simulations)} simulations")
            
            return {
                "simulations": simulations,
                "count": len(simulations)
            }
            
        except Exception as e:
            logger.error(f"Failed to list simulations: {e}")
            raise
    
    @server.tool(
        name="get_simulation_info",
        title="Get Simulation Info",
        description="Get detailed information about a simulation"
    )
    async def get_simulation_info(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a simulation.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation information
        """
        try:
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            summary = lammps_server.simulation_manager.get_simulation_summary(simulation_id)
            
            ctx.info(f"Retrieved info for simulation: {sim.name}")
            
            return {
                "simulation": sim.to_dict(),
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get simulation info: {e}")
            raise
    
    @server.tool(
        name="delete_simulation",
        title="Delete Simulation",
        description="Delete a simulation"
    )
    async def delete_simulation(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Delete a simulation.
        
        Args:
            simulation_id: Simulation ID to delete
            
        Returns:
            Deletion result
        """
        try:
            success = lammps_server.simulation_manager.delete_simulation(simulation_id)
            
            if success:
                ctx.info(f"Deleted simulation: {simulation_id}")
                return {
                    "simulation_id": simulation_id,
                    "status": "deleted",
                    "success": True
                }
            else:
                return {
                    "simulation_id": simulation_id,
                    "status": "not_found",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Failed to delete simulation: {e}")
            raise 