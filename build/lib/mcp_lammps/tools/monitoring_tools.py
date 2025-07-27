"""
Monitoring Tools - Tools for monitoring LAMMPS simulations in real-time.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp.server import Context

logger = logging.getLogger(__name__)


def register_monitoring_tools(server: Any, lammps_server: Any) -> None:
    """
    Register monitoring tools with the MCP server.
    
    Args:
        server: MCP server instance
        lammps_server: LAMMPS server instance
    """
    
    @server.tool(
        name="get_simulation_progress",
        title="Get Simulation Progress",
        description="Get the current progress of a simulation"
    )
    async def get_simulation_progress(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get the current progress of a simulation.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation progress information
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Get progress information
            progress_info = {
                "simulation_id": simulation_id,
                "name": sim.name,
                "state": sim.state,
                "current_step": sim.current_step,
                "total_steps": sim.total_steps,
                "progress_percentage": sim.progress_percentage,
                "started_at": sim.started_at.isoformat() if sim.started_at else None,
                "estimated_completion": None
            }
            
            # Calculate estimated completion time
            if sim.started_at and sim.state == "running" and sim.progress_percentage > 0:
                elapsed_time = time.time() - sim.started_at.timestamp()
                if sim.progress_percentage > 0:
                    total_estimated_time = elapsed_time / (sim.progress_percentage / 100)
                    remaining_time = total_estimated_time - elapsed_time
                    progress_info["estimated_completion"] = {
                        "elapsed_seconds": elapsed_time,
                        "remaining_seconds": remaining_time,
                        "total_estimated_seconds": total_estimated_time
                    }
            
            ctx.info(f"Retrieved progress for simulation: {sim.name}")
            
            return progress_info
            
        except Exception as e:
            logger.error(f"Failed to get simulation progress: {e}")
            raise
    
    @server.tool(
        name="get_system_properties",
        title="Get System Properties",
        description="Get real-time system properties from a running simulation"
    )
    async def get_system_properties(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get real-time system properties from a running simulation.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            System properties
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Check if simulation is running and has LAMMPS instance
            if sim.state != "running" or not sim.lammps_instance:
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "state": sim.state,
                    "message": "Simulation is not running or LAMMPS instance not available"
                }
            
            # Get system information
            system_info = lammps_server.lammps_interface.get_system_info(sim.lammps_instance)
            thermo_data = lammps_server.lammps_interface.get_thermo_data(sim.lammps_instance)
            
            # Combine information
            properties = {
                "simulation_id": simulation_id,
                "name": sim.name,
                "state": sim.state,
                "system_info": system_info,
                "thermo_data": thermo_data,
                "timestamp": time.time()
            }
            
            ctx.info(f"Retrieved system properties for simulation: {sim.name}")
            
            return properties
            
        except Exception as e:
            logger.error(f"Failed to get system properties: {e}")
            raise
    
    @server.tool(
        name="get_energy_data",
        title="Get Energy Data",
        description="Get energy components from a running simulation"
    )
    async def get_energy_data(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get energy components from a running simulation.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Energy data
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Check if simulation is running and has LAMMPS instance
            if sim.state != "running" or not sim.lammps_instance:
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "state": sim.state,
                    "message": "Simulation is not running or LAMMPS instance not available"
                }
            
            # Get thermodynamic data
            thermo_data = lammps_server.lammps_interface.get_thermo_data(sim.lammps_instance)
            
            # Extract energy components
            energy_data = {
                "simulation_id": simulation_id,
                "name": sim.name,
                "state": sim.state,
                "potential_energy": thermo_data.get("pe", 0.0),
                "kinetic_energy": thermo_data.get("ke", 0.0),
                "total_energy": thermo_data.get("etotal", 0.0),
                "step": thermo_data.get("step", 0),
                "timestamp": time.time()
            }
            
            ctx.info(f"Retrieved energy data for simulation: {sim.name}")
            
            return energy_data
            
        except Exception as e:
            logger.error(f"Failed to get energy data: {e}")
            raise
    
    @server.tool(
        name="get_structure_data",
        title="Get Structure Data",
        description="Get structural information from a running simulation"
    )
    async def get_structure_data(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get structural information from a running simulation.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Structure data
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Check if simulation is running and has LAMMPS instance
            if sim.state != "running" or not sim.lammps_instance:
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "state": sim.state,
                    "message": "Simulation is not running or LAMMPS instance not available"
                }
            
            # Get system information
            system_info = lammps_server.lammps_interface.get_system_info(sim.lammps_instance)
            atom_data = lammps_server.lammps_interface.get_atom_data(sim.lammps_instance)
            
            # Extract structural information
            structure_data = {
                "simulation_id": simulation_id,
                "name": sim.name,
                "state": sim.state,
                "num_atoms": system_info.get("num_atoms", 0),
                "num_types": system_info.get("num_types", 0),
                "box": system_info.get("box", {}),
                "has_positions": "positions" in atom_data,
                "has_velocities": "velocities" in atom_data,
                "has_forces": "forces" in atom_data,
                "timestamp": time.time()
            }
            
            # Add atom data if available (limit size for monitoring)
            if atom_data.get("positions") is not None:
                positions = atom_data["positions"]
                structure_data["position_stats"] = {
                    "mean_x": float(positions[:, 0].mean()),
                    "mean_y": float(positions[:, 1].mean()),
                    "mean_z": float(positions[:, 2].mean()),
                    "std_x": float(positions[:, 0].std()),
                    "std_y": float(positions[:, 1].std()),
                    "std_z": float(positions[:, 2].std())
                }
            
            ctx.info(f"Retrieved structure data for simulation: {sim.name}")
            
            return structure_data
            
        except Exception as e:
            logger.error(f"Failed to get structure data: {e}")
            raise
    
    @server.tool(
        name="monitor_simulation",
        title="Monitor Simulation",
        description="Monitor a simulation for a specified duration"
    )
    async def monitor_simulation(
        ctx: Context,
        simulation_id: str,
        duration_seconds: int = 60,
        interval_seconds: int = 5
    ) -> Dict[str, Any]:
        """
        Monitor a simulation for a specified duration.
        
        Args:
            simulation_id: Simulation ID
            duration_seconds: Total monitoring duration in seconds
            interval_seconds: Monitoring interval in seconds
            
        Returns:
            Monitoring results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Check if simulation is running
            if sim.state != "running":
                return {
                    "simulation_id": simulation_id,
                    "name": sim.name,
                    "state": sim.state,
                    "message": "Simulation is not running"
                }
            
            # Monitor simulation
            monitoring_data = []
            start_time = time.time()
            
            ctx.info(f"Starting monitoring for simulation: {sim.name}")
            
            while time.time() - start_time < duration_seconds:
                # Check if simulation is still running
                if sim.state != "running":
                    break
                
                # Get current data
                try:
                    thermo_data = lammps_server.lammps_interface.get_thermo_data(sim.lammps_instance)
                    system_info = lammps_server.lammps_interface.get_system_info(sim.lammps_instance)
                    
                    monitoring_data.append({
                        "timestamp": time.time(),
                        "step": thermo_data.get("step", 0),
                        "temperature": thermo_data.get("temperature", 0.0),
                        "pressure": thermo_data.get("pressure", 0.0),
                        "potential_energy": thermo_data.get("pe", 0.0),
                        "kinetic_energy": thermo_data.get("ke", 0.0),
                        "num_atoms": system_info.get("num_atoms", 0),
                        "progress_percentage": sim.progress_percentage
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to get monitoring data: {e}")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
            
            # Calculate monitoring statistics
            if monitoring_data:
                temps = [d["temperature"] for d in monitoring_data]
                pressures = [d["pressure"] for d in monitoring_data]
                energies = [d["potential_energy"] for d in monitoring_data]
                
                stats = {
                    "temperature": {
                        "mean": float(sum(temps) / len(temps)),
                        "std": float((sum((t - sum(temps)/len(temps))**2 for t in temps) / len(temps))**0.5),
                        "min": float(min(temps)),
                        "max": float(max(temps))
                    },
                    "pressure": {
                        "mean": float(sum(pressures) / len(pressures)),
                        "std": float((sum((p - sum(pressures)/len(pressures))**2 for p in pressures) / len(pressures))**0.5),
                        "min": float(min(pressures)),
                        "max": float(max(pressures))
                    },
                    "energy": {
                        "mean": float(sum(energies) / len(energies)),
                        "std": float((sum((e - sum(energies)/len(energies))**2 for e in energies) / len(energies))**0.5),
                        "min": float(min(energies)),
                        "max": float(max(energies))
                    }
                }
            else:
                stats = {}
            
            ctx.info(f"Completed monitoring for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "monitoring_duration": duration_seconds,
                "monitoring_interval": interval_seconds,
                "data_points": len(monitoring_data),
                "final_state": sim.state,
                "statistics": stats,
                "monitoring_data": monitoring_data[-10:]  # Last 10 data points
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor simulation: {e}")
            raise
    
    @server.tool(
        name="get_active_simulations",
        title="Get Active Simulations",
        description="Get list of all active simulations"
    )
    async def get_active_simulations(ctx: Context) -> Dict[str, Any]:
        """
        Get list of all active simulations.
        
        Returns:
            List of active simulations
        """
        try:
            # Get active simulation IDs
            active_ids = lammps_server.simulation_manager.get_active_simulations()
            
            # Get detailed information for each active simulation
            active_simulations = []
            for sim_id in active_ids:
                sim = lammps_server.simulation_manager.get_simulation(sim_id)
                if sim:
                    summary = lammps_server.simulation_manager.get_simulation_summary(sim_id)
                    active_simulations.append({
                        "simulation_id": sim_id,
                        "name": sim.name,
                        "state": sim.state,
                        "summary": summary
                    })
            
            ctx.info(f"Found {len(active_simulations)} active simulations")
            
            return {
                "active_simulations": active_simulations,
                "count": len(active_simulations)
            }
            
        except Exception as e:
            logger.error(f"Failed to get active simulations: {e}")
            raise
    
    @server.tool(
        name="get_simulation_logs",
        title="Get Simulation Logs",
        description="Get recent logs from a simulation"
    )
    async def get_simulation_logs(
        ctx: Context,
        simulation_id: str,
        num_entries: int = 50
    ) -> Dict[str, Any]:
        """
        Get recent logs from a simulation.
        
        Args:
            simulation_id: Simulation ID
            num_entries: Number of log entries to retrieve
            
        Returns:
            Simulation logs
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Get recent logs
            recent_logs = sim.logs[-num_entries:] if sim.logs else []
            
            ctx.info(f"Retrieved {len(recent_logs)} log entries for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "total_logs": len(sim.logs),
                "recent_logs": recent_logs,
                "num_entries": len(recent_logs)
            }
            
        except Exception as e:
            logger.error(f"Failed to get simulation logs: {e}")
            raise 