"""
Control Tools - Tools for controlling LAMMPS simulations.
"""

import asyncio
import logging
import threading
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
        name="run_equilibration",
        title="Run Equilibration",
        description="Run equilibration phase of a simulation"
    )
    async def run_equilibration(
        ctx: Context,
        simulation_id: str,
        temperature: float = 300.0,
        duration: int = 1000
    ) -> Dict[str, Any]:
        """
        Run equilibration phase of a simulation.
        
        Args:
            simulation_id: Simulation ID
            temperature: Equilibration temperature (K)
            duration: Equilibration duration (steps)
            
        Returns:
            Equilibration result
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
            sim.add_log(f"Starting equilibration at {temperature}K for {duration} steps")
            
            # Create LAMMPS instance
            lmp = lammps_server.lammps_interface.create_instance()
            if not lmp:
                raise RuntimeError("Failed to create LAMMPS instance")
            
            sim.lammps_instance = lmp
            
            # Run equilibration in background thread
            def run_equilibration_thread():
                try:
                    sim.update_state("running")
                    sim.add_log("Equilibration started")
                    
                    # Setup basic simulation
                    structure_file = sim.config.get("structure_file")
                    if structure_file:
                        # Read structure
                        lammps_server.lammps_interface.run_command(lmp, f"read_data {structure_file}")
                    
                    # Setup force field
                    force_field = sim.config.get("force_field", "lj/cut")
                    lammps_server.lammps_interface.run_command(lmp, f"pair_style {force_field}")
                    lammps_server.lammps_interface.run_command(lmp, "pair_coeff * *")
                    
                    # Setup equilibration
                    timestep = sim.config.get("timestep", 0.001)
                    lammps_server.lammps_interface.run_command(lmp, f"timestep {timestep}")
                    lammps_server.lammps_interface.run_command(lmp, f"fix 1 all nvt temp {temperature} {temperature} 100.0")
                    
                    # Run equilibration
                    lammps_server.lammps_interface.run_command(lmp, f"run {duration}")
                    lammps_server.lammps_interface.run_command(lmp, "unfix 1")
                    
                    # Get results
                    system_info = lammps_server.lammps_interface.get_system_info(lmp)
                    thermo_data = lammps_server.lammps_interface.get_thermo_data(lmp)
                    
                    # Save results
                    sim.add_result("equilibration_system_info", system_info)
                    sim.add_result("equilibration_thermo", thermo_data)
                    
                    sim.update_state("completed")
                    sim.add_log("Equilibration completed successfully")
                    
                except Exception as e:
                    sim.set_error(str(e))
                    sim.add_log(f"Equilibration failed: {e}")
                    logger.error(f"Equilibration {simulation_id} failed: {e}")
            
            # Start equilibration thread
            sim_thread = threading.Thread(target=run_equilibration_thread)
            sim_thread.daemon = True
            sim_thread.start()
            
            ctx.info(f"Started equilibration for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "status": "equilibration_started",
                "temperature": temperature,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Failed to run equilibration: {e}")
            raise
    
    @server.tool(
        name="run_production",
        title="Run Production",
        description="Run production phase of a simulation"
    )
    async def run_production(
        ctx: Context,
        simulation_id: str,
        temperature: float = 300.0,
        pressure: float = 1.0,
        duration: int = 10000
    ) -> Dict[str, Any]:
        """
        Run production phase of a simulation.
        
        Args:
            simulation_id: Simulation ID
            temperature: Production temperature (K)
            pressure: Production pressure (atm)
            duration: Production duration (steps)
            
        Returns:
            Production result
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
            sim.add_log(f"Starting production at {temperature}K, {pressure}atm for {duration} steps")
            
            # Create LAMMPS instance
            lmp = lammps_server.lammps_interface.create_instance()
            if not lmp:
                raise RuntimeError("Failed to create LAMMPS instance")
            
            sim.lammps_instance = lmp
            
            # Run production in background thread
            def run_production_thread():
                try:
                    sim.update_state("running")
                    sim.add_log("Production started")
                    
                    # Setup basic simulation
                    structure_file = sim.config.get("structure_file")
                    if structure_file:
                        # Read structure
                        lammps_server.lammps_interface.run_command(lmp, f"read_data {structure_file}")
                    
                    # Setup force field
                    force_field = sim.config.get("force_field", "lj/cut")
                    lammps_server.lammps_interface.run_command(lmp, f"pair_style {force_field}")
                    lammps_server.lammps_interface.run_command(lmp, "pair_coeff * *")
                    
                    # Setup production
                    timestep = sim.config.get("timestep", 0.001)
                    lammps_server.lammps_interface.run_command(lmp, f"timestep {timestep}")
                    lammps_server.lammps_interface.run_command(lmp, f"fix 1 all npt temp {temperature} {temperature} 100.0 iso {pressure} {pressure} 1000.0")
                    
                    # Run production
                    lammps_server.lammps_interface.run_command(lmp, f"run {duration}")
                    lammps_server.lammps_interface.run_command(lmp, "unfix 1")
                    
                    # Get results
                    system_info = lammps_server.lammps_interface.get_system_info(lmp)
                    thermo_data = lammps_server.lammps_interface.get_thermo_data(lmp)
                    
                    # Save results
                    sim.add_result("production_system_info", system_info)
                    sim.add_result("production_thermo", thermo_data)
                    
                    sim.update_state("completed")
                    sim.add_log("Production completed successfully")
                    
                except Exception as e:
                    sim.set_error(str(e))
                    sim.add_log(f"Production failed: {e}")
                    logger.error(f"Production {simulation_id} failed: {e}")
            
            # Start production thread
            sim_thread = threading.Thread(target=run_production_thread)
            sim_thread.daemon = True
            sim_thread.start()
            
            ctx.info(f"Started production for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "status": "production_started",
                "temperature": temperature,
                "pressure": pressure,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Failed to run production: {e}")
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
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get the current status of a simulation.
        
        Args:
            simulation_id: Simulation ID
            
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
            
            ctx.info(f"Retrieved status for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "status": summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get simulation status: {e}")
            raise
    
    @server.tool(
        name="pause_simulation",
        title="Pause Simulation",
        description="Pause a running simulation"
    )
    async def pause_simulation(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Pause a running simulation.
        
        Args:
            simulation_id: Simulation ID to pause
            
        Returns:
            Pause result
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Pause simulation
            if sim.state == "running":
                sim.update_state("paused")
                sim.add_log("Simulation paused by user")
                
                ctx.info(f"Paused simulation: {sim.name}")
                
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "status": "paused",
                    "message": "Simulation paused successfully"
                }
            else:
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "status": sim.state,
                    "message": "Simulation is not running"
                }
                
        except Exception as e:
            logger.error(f"Failed to pause simulation: {e}")
            raise
    
    @server.tool(
        name="resume_simulation",
        title="Resume Simulation",
        description="Resume a paused simulation"
    )
    async def resume_simulation(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Resume a paused simulation.
        
        Args:
            simulation_id: Simulation ID to resume
            
        Returns:
            Resume result
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Resume simulation
            if sim.state == "paused":
                sim.update_state("running")
                sim.add_log("Simulation resumed by user")
                
                ctx.info(f"Resumed simulation: {sim.name}")
                
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "status": "resumed",
                    "message": "Simulation resumed successfully"
                }
            else:
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "status": sim.state,
                    "message": "Simulation is not paused"
                }
                
        except Exception as e:
            logger.error(f"Failed to resume simulation: {e}")
            raise 