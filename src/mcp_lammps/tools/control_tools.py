"""
Control Tools - Tools for controlling LAMMPS simulations.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp.server import Context

logger = logging.getLogger(__name__)


def register_control_tools(server: Any, lammps_server: Any) -> None:
    """
    Register control tools with the MCP server.
    
    Args:
        server: MCP server instance
        lammps_server: LAMMPS server instance
    """
    
    @server.tool(
        name="run_simulation",
        title="Run Simulation",
        description="Run a LAMMPS simulation"
    )
    async def run_simulation(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Run a LAMMPS simulation.
        
        Args:
            simulation_id: Simulation ID to run
            
        Returns:
            Simulation run result
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Check if LAMMPS is available
            if not lammps_server.lammps_interface.is_available():
                raise RuntimeError("LAMMPS is not available")
            
            # Update simulation state
            sim.update_state("setup")
            sim.add_log("Starting simulation setup")
            
            # Create LAMMPS instance
            lmp = lammps_server.lammps_interface.create_instance()
            if not lmp:
                raise RuntimeError("Failed to create LAMMPS instance")
            
            sim.lammps_instance = lmp
            
            # Run simulation in background thread
            def run_sim_thread():
                try:
                    sim.update_state("running")
                    sim.add_log("Simulation started")
                    
                    # Get script file from config
                    script_file = sim.config.get("script_file")
                    if script_file and Path(script_file).exists():
                        # Run script file
                        with open(script_file, 'r') as f:
                            script_content = f.read()
                        
                        success = lammps_server.lammps_interface.run_script(lmp, script_content)
                        if not success:
                            raise RuntimeError("Failed to run LAMMPS script")
                    else:
                        # Create and run liquid script with best practice protocol
                        structure_file = sim.config.get("structure_file")
                        if structure_file:
                            script_content = lammps_server.lammps_interface.create_liquid_script(
                                structure_file=structure_file,
                                force_field=sim.config.get("force_field", "openff"),
                                temperature=sim.config.get("temperature", 300.0),
                                pressure=sim.config.get("pressure", 1.0),
                                timestep=sim.config.get("timestep", 1.0),
                                equilibration_steps=sim.config.get("equilibration_steps", 100000),
                                production_steps=sim.config.get("production_steps", 1000000)
                            )
                            
                            success = lammps_server.lammps_interface.run_script(lmp, script_content)
                            if not success:
                                raise RuntimeError("Failed to run LAMMPS script")
                        else:
                            raise ValueError("No structure file or script file available")
                    
                    # Get final results
                    system_info = lammps_server.lammps_interface.get_system_info(lmp)
                    thermo_data = lammps_server.lammps_interface.get_thermo_data(lmp)
                    
                    # Save results
                    sim.add_result("system_info", system_info)
                    sim.add_result("final_thermo", thermo_data)
                    
                    sim.update_state("completed")
                    sim.add_log("Simulation completed successfully")
                    
                except Exception as e:
                    sim.set_error(str(e))
                    sim.add_log(f"Simulation failed: {e}")
                    logger.error(f"Simulation {simulation_id} failed: {e}")
            
            # Start simulation thread
            sim_thread = threading.Thread(target=run_sim_thread)
            sim_thread.daemon = True
            sim_thread.start()
            
            ctx.info(f"Started simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "status": "started",
                "message": "Simulation is running in background"
            }
            
        except Exception as e:
            logger.error(f"Failed to run simulation: {e}")
            raise
    
    @server.tool(
        name="stop_simulation",
        title="Stop Simulation",
        description="Stop a running simulation"
    )
    async def stop_simulation(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Stop a running simulation.
        
        Args:
            simulation_id: Simulation ID to stop
            
        Returns:
            Stop result
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Stop simulation
            sim.stop()
            sim.add_log("Simulation stopped by user")
            
            ctx.info(f"Stopped simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "status": "stopped",
                "message": "Simulation stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to stop simulation: {e}")
            raise
    
    @server.tool(
        name="get_simulation_status",
        title="Get Simulation Status",
        description="Get the current status of a simulation"
    )
    async def get_simulation_status(
        ctx: Context,
        simulation_id: str,
        include_system_properties: bool = False,
        include_energy_data: bool = False
    ) -> Dict[str, Any]:
        """
        Get the current status of a simulation.
        
        Args:
            simulation_id: Simulation ID
            include_system_properties: Whether to include detailed system properties
            include_energy_data: Whether to include energy data
            
        Returns:
            Simulation status
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Get summary
            summary = lammps_server.simulation_manager.get_simulation_summary(simulation_id)
            
            result = {
                "simulation_id": simulation_id,
                "name": sim.name,
                "status": summary
            }
            
            # Add system properties if requested and simulation is running
            if include_system_properties and sim.state == "running" and sim.lammps_instance:
                try:
                    system_info = lammps_server.lammps_interface.get_system_info(sim.lammps_instance)
                    thermo_data = lammps_server.lammps_interface.get_thermo_data(sim.lammps_instance)
                    result["system_properties"] = {
                        "system_info": system_info,
                        "thermo_data": thermo_data,
                        "timestamp": time.time()
                    }
                except Exception as e:
                    logger.warning(f"Failed to get system properties: {e}")
                    result["system_properties"] = {"error": str(e)}
            
            # Add energy data if requested and simulation is running
            if include_energy_data and sim.state == "running" and sim.lammps_instance:
                try:
                    thermo_data = lammps_server.lammps_interface.get_thermo_data(sim.lammps_instance)
                    result["energy_data"] = {
                        "potential_energy": thermo_data.get("pe", 0.0),
                        "kinetic_energy": thermo_data.get("ke", 0.0),
                        "total_energy": thermo_data.get("etotal", 0.0),
                        "step": thermo_data.get("step", 0),
                        "timestamp": time.time()
                    }
                except Exception as e:
                    logger.warning(f"Failed to get energy data: {e}")
                    result["energy_data"] = {"error": str(e)}
            
            ctx.info(f"Retrieved status for simulation: {sim.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get simulation status: {e}")
            raise