"""
Unit tests for SLURM manager module.

Tests high-level SLURM job management functionality.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_lammps.slurm_manager import SlurmManager, SlurmJob
from mcp_lammps.slurm_interface import SlurmJobState


class TestSlurmJob(unittest.TestCase):
    """Test suite for SlurmJob class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.job = SlurmJob(
            simulation_id="test_sim_123",
            job_name="test_job",
            work_dir=self.temp_dir,
            slurm_config={"partition": "compute", "nodes": 1}
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test SlurmJob initialization."""
        self.assertEqual(self.job.simulation_id, "test_sim_123")
        self.assertEqual(self.job.job_name, "test_job")
        self.assertEqual(self.job.state, SlurmJobState.UNKNOWN)
        self.assertIsNone(self.job.job_id)
    
    def test_update_state(self):
        """Test state updates."""
        self.job.update_state(SlurmJobState.PENDING)
        self.assertEqual(self.job.state, SlurmJobState.PENDING)
        
        self.job.update_state(SlurmJobState.RUNNING)
        self.assertEqual(self.job.state, SlurmJobState.RUNNING)
        self.assertIsNotNone(self.job.started_at)
        
        self.job.update_state(SlurmJobState.COMPLETED)
        self.assertEqual(self.job.state, SlurmJobState.COMPLETED)
        self.assertIsNotNone(self.job.completed_at)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        self.job.job_id = "12345"
        self.job.state = SlurmJobState.RUNNING
        
        job_dict = self.job.to_dict()
        
        self.assertEqual(job_dict["simulation_id"], "test_sim_123")
        self.assertEqual(job_dict["job_id"], "12345")
        self.assertEqual(job_dict["state"], SlurmJobState.RUNNING)
        self.assertIn("slurm_config", job_dict)
    
    def test_save_and_load_state(self):
        """Test saving and loading job state."""
        self.job.job_id = "12345"
        self.job.state = SlurmJobState.RUNNING
        self.job.metadata = {"test": "value"}
        
        # Save state
        self.job.save_state()
        
        # Load state
        state_file = self.temp_dir / f"slurm_job_{self.job.job_name}.json"
        loaded_job = SlurmJob.load_state(state_file)
        
        self.assertIsNotNone(loaded_job)
        self.assertEqual(loaded_job.simulation_id, self.job.simulation_id)
        self.assertEqual(loaded_job.job_id, "12345")
        self.assertEqual(loaded_job.state, SlurmJobState.RUNNING)


class TestSlurmManager(unittest.TestCase):
    """Test suite for SlurmManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Mock SLURM interface to avoid actual SLURM commands
        with patch('mcp_lammps.slurm_manager.SlurmInterface') as mock_interface:
            mock_interface.return_value.is_available.return_value = True
            self.manager = SlurmManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test SlurmManager initialization."""
        self.assertEqual(self.manager.work_dir, self.temp_dir)
        self.assertIsNotNone(self.manager.slurm_interface)
        self.assertIsNotNone(self.manager.slurm_config)
    
    def test_is_available(self):
        """Test SLURM availability check."""
        with patch.object(self.manager.slurm_interface, 'is_available', return_value=True):
            self.assertTrue(self.manager.is_available())
    
    def test_generate_batch_script(self):
        """Test batch script generation."""
        lammps_script = "test.in"
        work_dir = self.temp_dir / "test_sim"
        work_dir.mkdir()
        
        script_path = self.manager.generate_batch_script(
            simulation_id="test_sim_123",
            simulation_name="test_simulation",
            lammps_script=lammps_script,
            work_dir=work_dir
        )
        
        self.assertTrue(script_path.exists())
        
        # Check script content
        content = script_path.read_text()
        self.assertIn("#!/bin/bash", content)
        self.assertIn("#SBATCH", content)
        self.assertIn("test_simulation", content)
        self.assertIn(lammps_script, content)
    
    def test_generate_batch_script_with_overrides(self):
        """Test batch script generation with config overrides."""
        lammps_script = "test.in"
        work_dir = self.temp_dir / "test_sim"
        work_dir.mkdir()
        
        config_overrides = {
            "nodes": 4,
            "ntasks_per_node": 16,
            "partition": "gpu"
        }
        
        script_path = self.manager.generate_batch_script(
            simulation_id="test_sim_123",
            simulation_name="test_simulation",
            lammps_script=lammps_script,
            work_dir=work_dir,
            config_overrides=config_overrides
        )
        
        content = script_path.read_text()
        self.assertIn("--nodes=4", content)
        self.assertIn("--ntasks-per-node=16", content)
        self.assertIn("--partition=gpu", content)
    
    @patch.object(SlurmManager, 'generate_batch_script')
    def test_submit_job(self, mock_generate_script):
        """Test job submission."""
        mock_script = self.temp_dir / "test_script.sh"
        mock_script.write_text("#!/bin/bash\n#SBATCH --job-name=test\necho test")
        mock_generate_script.return_value = mock_script
        
        with patch.object(self.manager.slurm_interface, 'submit_job', return_value="12345"):
            job_id = self.manager.submit_job(
                simulation_id="test_sim_123",
                simulation_name="test_job",
                lammps_script="test.in",
                work_dir=self.temp_dir
            )
        
        self.assertEqual(job_id, "12345")
        self.assertIn("test_sim_123", self.manager._jobs)
        self.assertIn("12345", self.manager._job_id_map)
    
    def test_get_job_status(self):
        """Test getting job status."""
        # Create a job
        job = SlurmJob("test_sim", "test_job", self.temp_dir, {})
        job.job_id = "12345"
        job.state = SlurmJobState.RUNNING
        self.manager._jobs["test_sim"] = job
        
        # Mock SLURM interface response
        with patch.object(self.manager.slurm_interface, 'get_job_status') as mock_status:
            mock_status.return_value = {
                "job_id": "12345",
                "state": SlurmJobState.RUNNING,
                "source": "squeue"
            }
            
            status = self.manager.get_job_status("test_sim")
        
        self.assertIsNotNone(status)
        self.assertEqual(status["job_id"], "12345")
    
    def test_cancel_job(self):
        """Test job cancellation."""
        # Create a job
        job = SlurmJob("test_sim", "test_job", self.temp_dir, {})
        job.job_id = "12345"
        self.manager._jobs["test_sim"] = job
        
        with patch.object(self.manager.slurm_interface, 'cancel_job', return_value=True):
            result = self.manager.cancel_job("test_sim")
        
        self.assertTrue(result)
        self.assertEqual(job.state, SlurmJobState.CANCELLED)
    
    def test_list_jobs(self):
        """Test listing jobs."""
        # Create multiple jobs
        job1 = SlurmJob("sim1", "job1", self.temp_dir, {})
        job1.state = SlurmJobState.RUNNING
        
        job2 = SlurmJob("sim2", "job2", self.temp_dir, {})
        job2.state = SlurmJobState.COMPLETED
        
        self.manager._jobs["sim1"] = job1
        self.manager._jobs["sim2"] = job2
        
        # List all jobs
        all_jobs = self.manager.list_jobs(active_only=False)
        self.assertEqual(len(all_jobs), 2)
        
        # List only active jobs
        active_jobs = self.manager.list_jobs(active_only=True)
        self.assertEqual(len(active_jobs), 1)
    
    def test_get_job_by_job_id(self):
        """Test getting job by SLURM job ID."""
        job = SlurmJob("test_sim", "test_job", self.temp_dir, {})
        job.job_id = "12345"
        self.manager._jobs["test_sim"] = job
        self.manager._job_id_map["12345"] = "test_sim"
        
        with patch.object(self.manager.slurm_interface, 'get_job_status') as mock_status:
            mock_status.return_value = {
                "job_id": "12345",
                "state": SlurmJobState.RUNNING,
                "source": "squeue"
            }
            
            status = self.manager.get_job_by_job_id("12345")
        
        self.assertIsNotNone(status)
        self.assertEqual(status["job_id"], "12345")
    
    def test_retrieve_job_output(self):
        """Test retrieving job output."""
        # Create a job with output files
        work_dir = self.temp_dir / "test_sim"
        work_dir.mkdir()
        
        job = SlurmJob("test_sim", "test_job", work_dir, {})
        job.job_id = "12345"
        job.state = SlurmJobState.COMPLETED
        job.output_path = work_dir / "slurm-12345.out"
        job.error_path = work_dir / "slurm-12345.err"
        
        # Create output files
        job.output_path.write_text("Standard output content")
        job.error_path.write_text("Standard error content")
        
        self.manager._jobs["test_sim"] = job
        
        # Retrieve output
        output = self.manager.retrieve_job_output("test_sim")
        
        self.assertEqual(output["simulation_id"], "test_sim")
        self.assertEqual(output["job_id"], "12345")
        self.assertIn("stdout", output["files"])
        self.assertIn("stderr", output["files"])
        self.assertEqual(output["files"]["stdout"], "Standard output content")


if __name__ == '__main__':
    unittest.main()

