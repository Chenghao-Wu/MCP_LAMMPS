"""
SLURM Manager - High-level SLURM job management and lifecycle tracking.

This module provides job management functionality including:
- Batch script generation
- Job submission and tracking
- Status monitoring and polling
- Job metadata persistence
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .slurm_interface import SlurmInterface, SlurmJobState
from .utils.slurm_config import SlurmConfig

logger = logging.getLogger(__name__)


class SlurmJob:
    """
    Represents a SLURM job with associated metadata.
    
    Tracks job information, status, and provides methods for
    monitoring and managing the job lifecycle.
    """
    
    def __init__(
        self,
        simulation_id: str,
        job_name: str,
        work_dir: Path,
        slurm_config: Dict[str, Any]
    ):
        """
        Initialize a SLURM job.
        
        Args:
            simulation_id: Associated simulation ID
            job_name: Name of the SLURM job
            work_dir: Working directory for the job
            slurm_config: SLURM configuration parameters
        """
        self.simulation_id = simulation_id
        self.job_name = job_name
        self.work_dir = Path(work_dir)
        self.slurm_config = slurm_config
        
        # Job tracking
        self.job_id: Optional[str] = None
        self.state = SlurmJobState.UNKNOWN
        self.submitted_at: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # File paths
        self.script_path = self.work_dir / f"slurm_{job_name}.sh"
        self.output_path = self.work_dir / f"slurm-{job_name}.out"
        self.error_path = self.work_dir / f"slurm-{job_name}.err"
        
        # Additional metadata
        self.metadata: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.Lock()
    
    def update_state(self, new_state: str) -> None:
        """
        Update job state with timestamp tracking.
        
        Args:
            new_state: New job state
        """
        with self._lock:
            old_state = self.state
            self.state = new_state
            
            if new_state == SlurmJobState.RUNNING and not self.started_at:
                self.started_at = datetime.now()
            elif new_state in [SlurmJobState.COMPLETED, SlurmJobState.FAILED, 
                              SlurmJobState.CANCELLED, SlurmJobState.TIMEOUT]:
                if not self.completed_at:
                    self.completed_at = datetime.now()
            
            logger.info(f"Job {self.job_id} state: {old_state} -> {new_state}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert job to dictionary for serialization.
        
        Returns:
            Dictionary representation of the job
        """
        with self._lock:
            return {
                "simulation_id": self.simulation_id,
                "job_name": self.job_name,
                "job_id": self.job_id,
                "state": self.state,
                "work_dir": str(self.work_dir),
                "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "slurm_config": self.slurm_config,
                "script_path": str(self.script_path),
                "output_path": str(self.output_path),
                "error_path": str(self.error_path),
                "metadata": self.metadata
            }
    
    def save_state(self) -> None:
        """Save job state to file."""
        state_file = self.work_dir / f"slurm_job_{self.job_name}.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving job state: {e}")
    
    @classmethod
    def load_state(cls, state_file: Path) -> Optional['SlurmJob']:
        """
        Load job state from file.
        
        Args:
            state_file: Path to state file
            
        Returns:
            SlurmJob instance or None if load fails
        """
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            job = cls(
                simulation_id=data["simulation_id"],
                job_name=data["job_name"],
                work_dir=Path(data["work_dir"]),
                slurm_config=data.get("slurm_config", {})
            )
            
            job.job_id = data.get("job_id")
            job.state = data.get("state", SlurmJobState.UNKNOWN)
            job.metadata = data.get("metadata", {})
            
            return job
            
        except Exception as e:
            logger.error(f"Error loading job state from {state_file}: {e}")
            return None


class SlurmManager:
    """
    Manager for SLURM jobs associated with simulations.
    
    Provides high-level job management including:
    - Script generation
    - Job submission
    - Status monitoring
    - Job tracking and persistence
    """
    
    def __init__(
        self,
        work_dir: Path,
        config_file: Optional[Path] = None
    ):
        """
        Initialize SLURM manager.
        
        Args:
            work_dir: Base working directory for jobs
            config_file: Optional SLURM configuration file
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.slurm_interface = SlurmInterface()
        self.slurm_config = SlurmConfig(config_file)
        
        # Job tracking
        self._jobs: Dict[str, SlurmJob] = {}  # simulation_id -> SlurmJob
        self._job_id_map: Dict[str, str] = {}  # job_id -> simulation_id
        self._lock = threading.Lock()
        
        # Load existing jobs
        self._load_existing_jobs()
        
        logger.info(f"SLURM manager initialized (SLURM available: {self.slurm_interface.is_available()})")
    
    def _load_existing_jobs(self) -> None:
        """Load existing job states from disk."""
        for state_file in self.work_dir.glob("**/slurm_job_*.json"):
            try:
                job = SlurmJob.load_state(state_file)
                if job:
                    self._jobs[job.simulation_id] = job
                    if job.job_id:
                        self._job_id_map[job.job_id] = job.simulation_id
                    logger.info(f"Loaded existing job: {job.job_name} ({job.job_id})")
            except Exception as e:
                logger.error(f"Failed to load job from {state_file}: {e}")
    
    def is_available(self) -> bool:
        """
        Check if SLURM is available.
        
        Returns:
            True if SLURM is available
        """
        return self.slurm_interface.is_available()
    
    def generate_batch_script(
        self,
        simulation_id: str,
        simulation_name: str,
        lammps_script: str,
        work_dir: Path,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate a SLURM batch script for a simulation.
        
        Args:
            simulation_id: Simulation ID
            simulation_name: Human-readable simulation name
            lammps_script: Path to LAMMPS input script
            work_dir: Working directory for the job
            config_overrides: Optional configuration overrides
            
        Returns:
            Path to generated batch script
        """
        # Get configuration
        config = self.slurm_config.get_config(simulation_id)
        if config_overrides:
            config.update(config_overrides)
        
        # Generate script path
        script_path = work_dir / f"slurm_{simulation_name}.sh"
        
        # Build module load commands
        module_commands = []
        if config.get("lammps_module"):
            module_commands.append(config["lammps_module"])
        for module in config.get("additional_modules", []):
            module_commands.append(f"module load {module}")
        module_str = "\n".join(module_commands) if module_commands else "# No modules to load"
        
        # Build LAMMPS command
        lammps_executable = config.get("lammps_executable", "lmp")
        mpi_command = self.slurm_config.format_mpi_command(simulation_id)
        lammps_command = f"{mpi_command} {lammps_executable} -in {lammps_script}"
        
        # Generate batch script
        script_content = f"""#!/bin/bash
#SBATCH --job-name={simulation_name}
#SBATCH --partition={config['partition']}
#SBATCH --nodes={config['nodes']}
#SBATCH --ntasks-per-node={config['ntasks_per_node']}
#SBATCH --time={config['time_limit']}
#SBATCH --mem-per-cpu={config['memory_per_cpu']}
#SBATCH --output={work_dir}/slurm-%j.out
#SBATCH --error={work_dir}/slurm-%j.err
"""
        
        # Add optional directives
        if config.get("mail_type"):
            script_content += f"#SBATCH --mail-type={config['mail_type']}\n"
        if config.get("mail_user"):
            script_content += f"#SBATCH --mail-user={config['mail_user']}\n"
        
        script_content += f"""
# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: {work_dir}"
echo ""

# Load modules
{module_str}

# Change to working directory
cd {work_dir} || exit 1

# Run LAMMPS
echo "Running LAMMPS..."
echo "Command: {lammps_command}"
echo ""

{lammps_command}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "LAMMPS completed successfully"
    echo "End time: $(date)"
    exit 0
else
    echo ""
    echo "LAMMPS failed with error code $?"
    echo "End time: $(date)"
    exit 1
fi
"""
        
        # Write script
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            script_path.chmod(0o755)
            
            logger.info(f"Generated batch script: {script_path}")
            return script_path
            
        except Exception as e:
            logger.error(f"Error generating batch script: {e}")
            raise
    
    def submit_job(
        self,
        simulation_id: str,
        simulation_name: str,
        lammps_script: str,
        work_dir: Path,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Submit a SLURM job for a simulation.
        
        Args:
            simulation_id: Simulation ID
            simulation_name: Human-readable simulation name
            lammps_script: Path to LAMMPS input script
            work_dir: Working directory for the job
            config_overrides: Optional configuration overrides
            
        Returns:
            SLURM job ID if successful, None otherwise
        """
        if not self.is_available():
            logger.error("SLURM not available")
            return None
        
        try:
            # Generate batch script
            script_path = self.generate_batch_script(
                simulation_id,
                simulation_name,
                lammps_script,
                work_dir,
                config_overrides
            )
            
            # Validate script
            is_valid, error = self.slurm_interface.validate_script(script_path)
            if not is_valid:
                logger.error(f"Invalid batch script: {error}")
                return None
            
            # Submit job
            job_id = self.slurm_interface.submit_job(script_path)
            if not job_id:
                logger.error("Failed to submit job")
                return None
            
            # Create job object
            config = self.slurm_config.get_config(simulation_id)
            if config_overrides:
                config.update(config_overrides)
            
            job = SlurmJob(simulation_id, simulation_name, work_dir, config)
            job.job_id = job_id
            job.state = SlurmJobState.PENDING
            job.submitted_at = datetime.now()
            job.script_path = script_path
            
            # Store job
            with self._lock:
                self._jobs[simulation_id] = job
                self._job_id_map[job_id] = simulation_id
            
            # Save state
            job.save_state()
            
            logger.info(f"Submitted job {job_id} for simulation {simulation_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def get_job_status(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a job by simulation ID.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Job status dictionary or None if not found
        """
        job = self._jobs.get(simulation_id)
        if not job or not job.job_id:
            return None
        
        # Query SLURM for current status
        status = self.slurm_interface.get_job_status(job.job_id)
        
        # Update job state
        if status["state"] != SlurmJobState.UNKNOWN:
            job.update_state(status["state"])
            job.save_state()
        
        # Combine job info with SLURM status
        result = job.to_dict()
        result["slurm_status"] = status
        
        return result
    
    def cancel_job(self, simulation_id: str) -> bool:
        """
        Cancel a SLURM job.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            True if successful, False otherwise
        """
        job = self._jobs.get(simulation_id)
        if not job or not job.job_id:
            logger.error(f"Job not found for simulation {simulation_id}")
            return False
        
        success = self.slurm_interface.cancel_job(job.job_id)
        if success:
            job.update_state(SlurmJobState.CANCELLED)
            job.save_state()
        
        return success
    
    def list_jobs(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        List all managed jobs.
        
        Args:
            active_only: If True, only return active jobs
            
        Returns:
            List of job information dictionaries
        """
        with self._lock:
            jobs = list(self._jobs.values())
        
        result = []
        for job in jobs:
            if active_only:
                active_states = {SlurmJobState.PENDING, SlurmJobState.RUNNING, SlurmJobState.SUSPENDED}
                if job.state not in active_states:
                    continue
            
            result.append(job.to_dict())
        
        return result
    
    def get_job_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job information by SLURM job ID.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Job information dictionary or None if not found
        """
        simulation_id = self._job_id_map.get(job_id)
        if not simulation_id:
            return None
        
        return self.get_job_status(simulation_id)
    
    def update_all_job_statuses(self) -> None:
        """Update status for all active jobs."""
        with self._lock:
            simulation_ids = list(self._jobs.keys())
        
        for simulation_id in simulation_ids:
            job = self._jobs.get(simulation_id)
            if job and job.state in [SlurmJobState.PENDING, SlurmJobState.RUNNING]:
                self.get_job_status(simulation_id)
    
    def retrieve_job_output(self, simulation_id: str) -> Dict[str, Any]:
        """
        Retrieve output files from a completed job.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Dictionary with output file contents
        """
        job = self._jobs.get(simulation_id)
        if not job:
            return {"error": "Job not found"}
        
        result = {
            "simulation_id": simulation_id,
            "job_id": job.job_id,
            "state": job.state,
            "files": {}
        }
        
        # Read standard output
        if job.output_path.exists():
            try:
                with open(job.output_path, 'r') as f:
                    result["files"]["stdout"] = f.read()
            except Exception as e:
                result["files"]["stdout_error"] = str(e)
        
        # Read standard error
        if job.error_path.exists():
            try:
                with open(job.error_path, 'r') as f:
                    result["files"]["stderr"] = f.read()
            except Exception as e:
                result["files"]["stderr_error"] = str(e)
        
        # List all files in work directory
        try:
            all_files = list(job.work_dir.glob("*"))
            result["files"]["available_files"] = [str(f.name) for f in all_files if f.is_file()]
        except Exception as e:
            result["files"]["listing_error"] = str(e)
        
        return result

