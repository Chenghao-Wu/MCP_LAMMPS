"""
Unit tests for SLURM interface module.

Tests SLURM command wrappers and job management functionality.
"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_lammps.slurm_interface import SlurmInterface, SlurmJobState


class TestSlurmInterface(unittest.TestCase):
    """Test suite for SlurmInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.slurm_interface = SlurmInterface()
    
    @patch('subprocess.run')
    def test_check_availability_success(self, mock_run):
        """Test SLURM availability check when SLURM is available."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="slurm 23.02.0"
        )
        
        interface = SlurmInterface()
        self.assertTrue(interface.is_available())
        self.assertIsNotNone(interface.get_version())
    
    @patch('subprocess.run')
    def test_check_availability_failure(self, mock_run):
        """Test SLURM availability check when SLURM is not available."""
        mock_run.side_effect = FileNotFoundError()
        
        interface = SlurmInterface()
        self.assertFalse(interface.is_available())
        self.assertIsNone(interface.get_version())
    
    @patch('subprocess.run')
    def test_submit_job_success(self, mock_run):
        """Test successful job submission."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Submitted batch job 12345"
        )
        
        # Create temporary script file
        test_script = Path("/tmp/test_script.sh")
        test_script.write_text("#!/bin/bash\necho test")
        
        try:
            job_id = self.slurm_interface.submit_job(test_script)
            self.assertEqual(job_id, "12345")
        finally:
            test_script.unlink()
    
    @patch('subprocess.run')
    def test_submit_job_failure(self, mock_run):
        """Test job submission failure."""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Error: Invalid partition"
        )
        
        test_script = Path("/tmp/test_script.sh")
        test_script.write_text("#!/bin/bash\necho test")
        
        try:
            job_id = self.slurm_interface.submit_job(test_script)
            self.assertIsNone(job_id)
        finally:
            test_script.unlink()
    
    @patch('subprocess.run')
    def test_get_job_status_running(self, mock_run):
        """Test getting status of a running job."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="JOBID,STATE,TIME,TIME_LIMIT,NODELIST,REASON\n12345,RUNNING,10:30,24:00:00,node01,"
        )
        
        status = self.slurm_interface.get_job_status("12345")
        self.assertEqual(status["state"], SlurmJobState.RUNNING)
        self.assertEqual(status["job_id"], "12345")
    
    @patch('subprocess.run')
    def test_get_job_status_completed(self, mock_run):
        """Test getting status of a completed job via sacct."""
        # squeue returns empty (job not in queue)
        def side_effect(*args, **kwargs):
            if 'squeue' in args[0]:
                return Mock(returncode=0, stdout="")
            else:  # sacct
                return Mock(
                    returncode=0,
                    stdout="12345|COMPLETED|1:30:00|2024-01-01T10:00:00|2024-01-01T11:30:00|0:0"
                )
        
        mock_run.side_effect = side_effect
        
        status = self.slurm_interface.get_job_status("12345")
        self.assertEqual(status["state"], SlurmJobState.COMPLETED)
    
    def test_normalize_state(self):
        """Test state normalization."""
        self.assertEqual(
            self.slurm_interface._normalize_state("RUNNING"),
            SlurmJobState.RUNNING
        )
        self.assertEqual(
            self.slurm_interface._normalize_state("R"),
            SlurmJobState.RUNNING
        )
        self.assertEqual(
            self.slurm_interface._normalize_state("COMPLETED"),
            SlurmJobState.COMPLETED
        )
        self.assertEqual(
            self.slurm_interface._normalize_state("FAILED"),
            SlurmJobState.FAILED
        )
        self.assertEqual(
            self.slurm_interface._normalize_state("CANCELLED"),
            SlurmJobState.CANCELLED
        )
    
    @patch('subprocess.run')
    def test_cancel_job_success(self, mock_run):
        """Test successful job cancellation."""
        mock_run.return_value = Mock(returncode=0)
        
        result = self.slurm_interface.cancel_job("12345")
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_cancel_job_failure(self, mock_run):
        """Test job cancellation failure."""
        mock_run.return_value = Mock(returncode=1, stderr="Job not found")
        
        result = self.slurm_interface.cancel_job("12345")
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_get_queue_info(self, mock_run):
        """Test getting queue information."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="JOBID,STATE,USER,NAME,TIME,TIME_LIMIT,NODELIST\n12345,RUNNING,user1,job1,10:30,24:00:00,node01\n12346,PENDING,user2,job2,0:00,12:00:00,"
        )
        
        jobs = self.slurm_interface.get_queue_info()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["job_id"], "12345")
        self.assertEqual(jobs[0]["state"], SlurmJobState.RUNNING)
        self.assertEqual(jobs[1]["state"], SlurmJobState.PENDING)
    
    def test_validate_script_valid(self):
        """Test validation of a valid script."""
        test_script = Path("/tmp/test_valid.sh")
        test_script.write_text("#!/bin/bash\n#SBATCH --job-name=test\necho test")
        
        try:
            is_valid, error = self.slurm_interface.validate_script(test_script)
            self.assertTrue(is_valid)
            self.assertIsNone(error)
        finally:
            test_script.unlink()
    
    def test_validate_script_no_shebang(self):
        """Test validation of script without shebang."""
        test_script = Path("/tmp/test_no_shebang.sh")
        test_script.write_text("#SBATCH --job-name=test\necho test")
        
        try:
            is_valid, error = self.slurm_interface.validate_script(test_script)
            self.assertFalse(is_valid)
            self.assertIn("shebang", error)
        finally:
            test_script.unlink()
    
    def test_validate_script_no_sbatch(self):
        """Test validation of script without SBATCH directives."""
        test_script = Path("/tmp/test_no_sbatch.sh")
        test_script.write_text("#!/bin/bash\necho test")
        
        try:
            is_valid, error = self.slurm_interface.validate_script(test_script)
            self.assertFalse(is_valid)
            self.assertIn("SBATCH", error)
        finally:
            test_script.unlink()
    
    def test_validate_script_not_found(self):
        """Test validation of non-existent script."""
        test_script = Path("/tmp/nonexistent.sh")
        
        is_valid, error = self.slurm_interface.validate_script(test_script)
        self.assertFalse(is_valid)
        self.assertIn("not found", error)


class TestSlurmJobState(unittest.TestCase):
    """Test suite for SlurmJobState enumeration."""
    
    def test_states_defined(self):
        """Test that all expected states are defined."""
        states = [
            SlurmJobState.PENDING,
            SlurmJobState.RUNNING,
            SlurmJobState.COMPLETED,
            SlurmJobState.FAILED,
            SlurmJobState.CANCELLED,
            SlurmJobState.TIMEOUT,
            SlurmJobState.NODE_FAIL,
            SlurmJobState.PREEMPTED,
            SlurmJobState.SUSPENDED,
            SlurmJobState.UNKNOWN
        ]
        
        for state in states:
            self.assertIsNotNone(state)
            self.assertIsInstance(state, str)


if __name__ == '__main__':
    unittest.main()

