"""
SLURM Interface - Low-level wrapper for SLURM commands.

This module provides a Python interface to SLURM workload manager commands,
enabling job submission, monitoring, and management on HPC clusters.
"""

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SlurmJobState:
    """Enumeration of SLURM job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


class SlurmInterface:
    """
    Interface for interacting with SLURM workload manager.
    
    This class provides methods for:
    - Checking SLURM availability
    - Submitting jobs via sbatch
    - Querying job status via squeue and sacct
    - Cancelling jobs via scancel
    - Parsing SLURM output formats
    """
    
    def __init__(self):
        """Initialize the SLURM interface."""
        self._slurm_available = False
        self._slurm_version = None
        self._check_availability()
    
    def _check_availability(self) -> None:
        """Check if SLURM is available on the system."""
        try:
            result = subprocess.run(
                ["squeue", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self._slurm_available = True
                self._slurm_version = result.stdout.strip()
                logger.info(f"SLURM available: {self._slurm_version}")
            else:
                logger.warning("SLURM squeue command not working")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"SLURM not available: {e}")
    
    def is_available(self) -> bool:
        """
        Check if SLURM is available.
        
        Returns:
            True if SLURM is available, False otherwise
        """
        return self._slurm_available
    
    def get_version(self) -> Optional[str]:
        """
        Get SLURM version.
        
        Returns:
            SLURM version string or None if not available
        """
        return self._slurm_version
    
    def submit_job(self, script_path: Path) -> Optional[str]:
        """
        Submit a job to SLURM using sbatch.
        
        Args:
            script_path: Path to the SLURM batch script
            
        Returns:
            Job ID string if successful, None otherwise
        """
        if not self._slurm_available:
            logger.error("SLURM not available")
            return None
        
        if not script_path.exists():
            logger.error(f"Script file not found: {script_path}")
            return None
        
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse job ID from sbatch output
                # Expected format: "Submitted batch job 12345"
                match = re.search(r"Submitted batch job (\d+)", result.stdout)
                if match:
                    job_id = match.group(1)
                    logger.info(f"Job submitted successfully: {job_id}")
                    return job_id
                else:
                    logger.error(f"Could not parse job ID from sbatch output: {result.stdout}")
                    return None
            else:
                logger.error(f"sbatch failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a SLURM job.
        
        First tries squeue for active jobs, then falls back to sacct for completed jobs.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Dictionary with job status information
        """
        # Try squeue first (for active jobs)
        status = self._get_status_from_squeue(job_id)
        
        # If not found in squeue, try sacct (for completed jobs)
        if status["state"] == SlurmJobState.UNKNOWN:
            status = self._get_status_from_sacct(job_id)
        
        return status
    
    def _get_status_from_squeue(self, job_id: str) -> Dict[str, Any]:
        """
        Query job status from squeue (active jobs).
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Dictionary with job status
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "--format=%i,%T,%M,%l,%N,%r"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    # Parse format: JOBID,STATE,TIME,TIME_LIMIT,NODELIST,REASON
                    parts = lines[1].split(',')
                    if len(parts) >= 4:
                        return {
                            "job_id": parts[0],
                            "state": self._normalize_state(parts[1]),
                            "elapsed_time": parts[2],
                            "time_limit": parts[3],
                            "nodes": parts[4] if len(parts) > 4 else "",
                            "reason": parts[5] if len(parts) > 5 else "",
                            "source": "squeue"
                        }
            
            return {"job_id": job_id, "state": SlurmJobState.UNKNOWN, "source": "squeue"}
            
        except Exception as e:
            logger.error(f"Error querying squeue: {e}")
            return {"job_id": job_id, "state": SlurmJobState.UNKNOWN, "source": "squeue", "error": str(e)}
    
    def _get_status_from_sacct(self, job_id: str) -> Dict[str, Any]:
        """
        Query job status from sacct (completed jobs).
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Dictionary with job status
        """
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=JobID,State,Elapsed,Start,End,ExitCode", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # Get the first line (main job, not job steps)
                for line in lines:
                    if not '.batch' in line and not '.extern' in line:
                        # Parse format: JobID|State|Elapsed|Start|End|ExitCode
                        parts = line.split('|')
                        if len(parts) >= 6:
                            exit_code = parts[5].split(':')[0] if ':' in parts[5] else parts[5]
                            return {
                                "job_id": parts[0],
                                "state": self._normalize_state(parts[1]),
                                "elapsed_time": parts[2],
                                "start_time": parts[3],
                                "end_time": parts[4],
                                "exit_code": exit_code,
                                "source": "sacct"
                            }
            
            return {"job_id": job_id, "state": SlurmJobState.UNKNOWN, "source": "sacct"}
            
        except Exception as e:
            logger.error(f"Error querying sacct: {e}")
            return {"job_id": job_id, "state": SlurmJobState.UNKNOWN, "source": "sacct", "error": str(e)}
    
    def _normalize_state(self, slurm_state: str) -> str:
        """
        Normalize SLURM state strings to standard states.
        
        Args:
            slurm_state: Raw SLURM state string
            
        Returns:
            Normalized state string
        """
        state = slurm_state.upper()
        
        # Map SLURM states to our standard states
        if state in ["PENDING", "PD", "CONFIGURING", "CF"]:
            return SlurmJobState.PENDING
        elif state in ["RUNNING", "R"]:
            return SlurmJobState.RUNNING
        elif state in ["COMPLETED", "CD"]:
            return SlurmJobState.COMPLETED
        elif state in ["FAILED", "F", "BOOT_FAIL", "BF", "DEADLINE", "DL", "OUT_OF_MEMORY", "OOM"]:
            return SlurmJobState.FAILED
        elif state in ["CANCELLED", "CA", "CANCELLED+"]:
            return SlurmJobState.CANCELLED
        elif state in ["TIMEOUT", "TO"]:
            return SlurmJobState.TIMEOUT
        elif state in ["NODE_FAIL", "NF"]:
            return SlurmJobState.NODE_FAIL
        elif state in ["PREEMPTED", "PR"]:
            return SlurmJobState.PREEMPTED
        elif state in ["SUSPENDED", "S"]:
            return SlurmJobState.SUSPENDED
        else:
            return SlurmJobState.UNKNOWN
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.
        
        Args:
            job_id: SLURM job ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if not self._slurm_available:
            logger.error("SLURM not available")
            return False
        
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Job {job_id} cancelled successfully")
                return True
            else:
                logger.error(f"Failed to cancel job {job_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def get_queue_info(self, user: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get information about jobs in the queue.
        
        Args:
            user: Optional username to filter jobs
            
        Returns:
            List of job information dictionaries
        """
        if not self._slurm_available:
            logger.error("SLURM not available")
            return []
        
        try:
            cmd = ["squeue", "--format=%i,%T,%u,%j,%M,%l,%N"]
            if user:
                cmd.extend(["-u", user])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                jobs = []
                lines = result.stdout.strip().split('\n')
                
                # Skip header
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 5:
                        jobs.append({
                            "job_id": parts[0],
                            "state": self._normalize_state(parts[1]),
                            "user": parts[2],
                            "name": parts[3],
                            "elapsed_time": parts[4],
                            "time_limit": parts[5] if len(parts) > 5 else "",
                            "nodes": parts[6] if len(parts) > 6 else ""
                        })
                
                return jobs
            else:
                logger.error(f"Failed to get queue info: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting queue info: {e}")
            return []
    
    def validate_script(self, script_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate a SLURM batch script without submitting it.
        
        Args:
            script_path: Path to the SLURM batch script
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not script_path.exists():
            return False, f"Script file not found: {script_path}"
        
        try:
            # Basic validation: check for shebang and SBATCH directives
            with open(script_path, 'r') as f:
                content = f.read()
            
            if not content.startswith('#!/bin/bash') and not content.startswith('#!/bin/sh'):
                return False, "Script must start with shebang (#!/bin/bash or #!/bin/sh)"
            
            # Check for at least one SBATCH directive
            if '#SBATCH' not in content:
                return False, "Script must contain at least one #SBATCH directive"
            
            # Could add more sophisticated validation here
            return True, None
            
        except Exception as e:
            return False, f"Error validating script: {e}"
    
    def get_node_info(self) -> List[Dict[str, Any]]:
        """
        Get information about cluster nodes.
        
        Returns:
            List of node information dictionaries
        """
        if not self._slurm_available:
            logger.error("SLURM not available")
            return []
        
        try:
            result = subprocess.run(
                ["sinfo", "--format=%N,%P,%T,%C,%m"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                nodes = []
                lines = result.stdout.strip().split('\n')
                
                # Skip header
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        nodes.append({
                            "name": parts[0],
                            "partition": parts[1],
                            "state": parts[2],
                            "cpus": parts[3],
                            "memory": parts[4] if len(parts) > 4 else ""
                        })
                
                return nodes
            else:
                logger.error(f"Failed to get node info: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting node info: {e}")
            return []

