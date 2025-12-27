"""
SLURM Tools - MCP tools for SLURM job operations.

This module provides MCP tools for:
- Submitting simulations to SLURM
- Monitoring job status
- Cancelling jobs
- Retrieving results
- Listing jobs
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp.server import Context

logger = logging.getLogger(__name__)


def register_slurm_tools(server: Any, lammps_server: Any) -> None:
    """
    Register SLURM tools with the MCP server.
    
    Args:
        server: MCP server instance
        lammps_server: LAMMPS server instance
    """
    
    @server.tool(
        name="submit_simulation_to_slurm",
        title="Submit Simulation to SLURM",
        description="Submit a LAMMPS simulation to SLURM cluster for execution"
    )
    async def submit_simulation_to_slurm(
        ctx: Context,
        simulation_id: str,
        partition: Optional[str] = None,
        nodes: Optional[int] = None,
        ntasks_per_node: Optional[int] = None,
        time_limit: Optional[str] = None,
        memory_per_cpu: Optional[str] = None,
        mail_type: Optional[str] = None,
        mail_user: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a simulation to SLURM cluster.
        
        Args:
            simulation_id: Simulation ID to submit
            partition: SLURM partition (optional, uses config default)
            nodes: Number of nodes (optional)
            ntasks_per_node: Tasks per node (optional)
            time_limit: Time limit (HH:MM:SS format, optional)
            memory_per_cpu: Memory per CPU (e.g., "4G", optional)
            mail_type: Email notification type (NONE/BEGIN/END/FAIL/ALL, optional)
            mail_user: Email address for notifications (optional)
            
        Returns:
            Submission result with job ID and status
        """
        try:
            # Check if SLURM manager is available
            if not hasattr(lammps_server, 'slurm_manager'):
                raise RuntimeError("SLURM manager not initialized")
            
            if not lammps_server.slurm_manager.is_available():
                raise RuntimeError("SLURM not available on this system")
            
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Check if already submitted to SLURM
            if sim.execution_mode == "slurm" and sim.slurm_job_id:
                raise ValueError(f"Simulation already submitted to SLURM (job ID: {sim.slurm_job_id})")
            
            # Build config overrides
            config_overrides = {}
            if partition:
                config_overrides["partition"] = partition
            if nodes:
                config_overrides["nodes"] = nodes
            if ntasks_per_node:
                config_overrides["ntasks_per_node"] = ntasks_per_node
            if time_limit:
                config_overrides["time_limit"] = time_limit
            if memory_per_cpu:
                config_overrides["memory_per_cpu"] = memory_per_cpu
            if mail_type:
                config_overrides["mail_type"] = mail_type
            if mail_user:
                config_overrides["mail_user"] = mail_user
            
            # Get or create LAMMPS script
            script_file = sim.config.get("script_file")
            if not script_file:
                # Generate a LAMMPS script if not exists
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
                    script_file = sim.work_dir / f"{sim.name}.in"
                    with open(script_file, 'w') as f:
                        f.write(script_content)
                    sim.config["script_file"] = str(script_file)
                    sim.save_state()
                else:
                    raise ValueError("No LAMMPS script or structure file available")
            
            # Submit to SLURM
            job_id = lammps_server.slurm_manager.submit_job(
                simulation_id=simulation_id,
                simulation_name=sim.name,
                lammps_script=script_file,
                work_dir=sim.work_dir,
                config_overrides=config_overrides
            )
            
            if not job_id:
                raise RuntimeError("Failed to submit job to SLURM")
            
            # Update simulation with SLURM info
            sim.execution_mode = "slurm"
            sim.slurm_job_id = job_id
            sim.slurm_status = "PENDING"
            sim.update_state("setup")
            sim.add_log(f"Submitted to SLURM with job ID: {job_id}")
            sim.save_state()
            
            ctx.info(f"Submitted simulation {sim.name} to SLURM (job ID: {job_id})")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "job_id": job_id,
                "status": "submitted",
                "execution_mode": "slurm",
                "message": f"Simulation submitted to SLURM cluster with job ID {job_id}"
            }
            
        except Exception as e:
            logger.error(f"Failed to submit simulation to SLURM: {e}")
            raise
    
    @server.tool(
        name="get_slurm_job_status",
        title="Get SLURM Job Status",
        description="Get the current status of a SLURM job"
    )
    async def get_slurm_job_status(
        ctx: Context,
        simulation_id: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get status of a SLURM job.
        
        Args:
            simulation_id: Simulation ID (provide either this or job_id)
            job_id: SLURM job ID (provide either this or simulation_id)
            
        Returns:
            Job status information
        """
        try:
            if not hasattr(lammps_server, 'slurm_manager'):
                raise RuntimeError("SLURM manager not initialized")
            
            if not simulation_id and not job_id:
                raise ValueError("Must provide either simulation_id or job_id")
            
            # Get job status
            if simulation_id:
                status = lammps_server.slurm_manager.get_job_status(simulation_id)
                if not status:
                    raise ValueError(f"No SLURM job found for simulation {simulation_id}")
            else:
                status = lammps_server.slurm_manager.get_job_by_job_id(job_id)
                if not status:
                    raise ValueError(f"SLURM job {job_id} not found")
            
            # Update simulation state if needed
            if simulation_id:
                sim = lammps_server.simulation_manager.get_simulation(simulation_id)
                if sim:
                    sim.slurm_status = status.get("slurm_status", {}).get("state")
                    
                    # Map SLURM state to simulation state
                    slurm_state = status.get("slurm_status", {}).get("state")
                    if slurm_state == "RUNNING":
                        sim.update_state("running")
                    elif slurm_state == "COMPLETED":
                        sim.update_state("completed")
                    elif slurm_state in ["FAILED", "TIMEOUT", "NODE_FAIL"]:
                        sim.update_state("failed")
                    elif slurm_state == "CANCELLED":
                        sim.update_state("stopped")
                    
                    sim.save_state()
            
            ctx.info(f"Retrieved SLURM job status: {status.get('state', 'UNKNOWN')}")
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get SLURM job status: {e}")
            raise
    
    @server.tool(
        name="cancel_slurm_job",
        title="Cancel SLURM Job",
        description="Cancel a running or pending SLURM job"
    )
    async def cancel_slurm_job(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Cancel a SLURM job.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Cancellation result
        """
        try:
            if not hasattr(lammps_server, 'slurm_manager'):
                raise RuntimeError("SLURM manager not initialized")
            
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            if sim.execution_mode != "slurm":
                raise ValueError(f"Simulation is not running on SLURM (mode: {sim.execution_mode})")
            
            if not sim.slurm_job_id:
                raise ValueError("No SLURM job ID found for this simulation")
            
            # Cancel job
            success = lammps_server.slurm_manager.cancel_job(simulation_id)
            
            if success:
                sim.update_state("stopped")
                sim.slurm_status = "CANCELLED"
                sim.add_log(f"SLURM job {sim.slurm_job_id} cancelled")
                sim.save_state()
                
                ctx.info(f"Cancelled SLURM job {sim.slurm_job_id}")
                
                return {
                    "simulation_id": simulation_id,
                    "job_id": sim.slurm_job_id,
                    "status": "cancelled",
                    "message": f"SLURM job {sim.slurm_job_id} cancelled successfully"
                }
            else:
                raise RuntimeError(f"Failed to cancel SLURM job {sim.slurm_job_id}")
            
        except Exception as e:
            logger.error(f"Failed to cancel SLURM job: {e}")
            raise
    
    @server.tool(
        name="retrieve_slurm_results",
        title="Retrieve SLURM Job Results",
        description="Retrieve output files and results from a SLURM job"
    )
    async def retrieve_slurm_results(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve results from a SLURM job.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Job output and results
        """
        try:
            if not hasattr(lammps_server, 'slurm_manager'):
                raise RuntimeError("SLURM manager not initialized")
            
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            if sim.execution_mode != "slurm":
                raise ValueError(f"Simulation is not running on SLURM (mode: {sim.execution_mode})")
            
            # Retrieve output
            results = lammps_server.slurm_manager.retrieve_job_output(simulation_id)
            
            # Add simulation results if available
            results["simulation_results"] = sim.results
            results["simulation_state"] = sim.state
            
            ctx.info(f"Retrieved results for simulation {sim.name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve SLURM results: {e}")
            raise
    
    @server.tool(
        name="list_slurm_jobs",
        title="List SLURM Jobs",
        description="List all SLURM jobs managed by this server"
    )
    async def list_slurm_jobs(
        ctx: Context,
        active_only: bool = False
    ) -> Dict[str, Any]:
        """
        List all SLURM jobs.
        
        Args:
            active_only: If True, only list active jobs (pending/running)
            
        Returns:
            List of job information
        """
        try:
            if not hasattr(lammps_server, 'slurm_manager'):
                raise RuntimeError("SLURM manager not initialized")
            
            # Get jobs
            jobs = lammps_server.slurm_manager.list_jobs(active_only=active_only)
            
            ctx.info(f"Listed {len(jobs)} SLURM jobs (active_only={active_only})")
            
            return {
                "total_jobs": len(jobs),
                "active_only": active_only,
                "jobs": jobs
            }
            
        except Exception as e:
            logger.error(f"Failed to list SLURM jobs: {e}")
            raise
    
    @server.tool(
        name="update_slurm_job_statuses",
        title="Update All SLURM Job Statuses",
        description="Update status for all active SLURM jobs"
    )
    async def update_slurm_job_statuses(
        ctx: Context
    ) -> Dict[str, Any]:
        """
        Update status for all active SLURM jobs.
        
        Returns:
            Update result
        """
        try:
            if not hasattr(lammps_server, 'slurm_manager'):
                raise RuntimeError("SLURM manager not initialized")
            
            # Update all job statuses
            lammps_server.slurm_manager.update_all_job_statuses()
            
            # Get updated job list
            jobs = lammps_server.slurm_manager.list_jobs(active_only=True)
            
            ctx.info(f"Updated status for all active SLURM jobs")
            
            return {
                "status": "updated",
                "active_jobs": len(jobs),
                "message": "All active job statuses updated"
            }
            
        except Exception as e:
            logger.error(f"Failed to update SLURM job statuses: {e}")
            raise

