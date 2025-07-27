"""
Simulation Manager - Handles simulation workflows and state management.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class SimulationState:
    """Enumeration of simulation states."""
    CREATED = "created"
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Simulation:
    """
    Represents a single simulation instance.
    
    This class tracks the state and progress of a simulation,
    including its configuration, current status, and results.
    """
    
    def __init__(
        self,
        simulation_id: str,
        name: str,
        config: Dict[str, Any],
        work_dir: Path
    ):
        """
        Initialize a simulation.
        
        Args:
            simulation_id: Unique identifier for the simulation
            name: Human-readable name for the simulation
            config: Simulation configuration
            work_dir: Working directory for the simulation
        """
        self.simulation_id = simulation_id
        self.name = name
        self.config = config
        self.work_dir = work_dir / simulation_id
        self.work_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.state = SimulationState.CREATED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Progress tracking
        self.current_step = 0
        self.total_steps = 0
        self.progress_percentage = 0.0
        
        # Results and data
        self.results: Dict[str, Any] = {}
        self.logs: List[str] = []
        self.error_message: Optional[str] = None
        
        # LAMMPS instance
        self.lammps_instance: Optional[Any] = None
        
        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def update_state(self, new_state: str) -> None:
        """
        Update the simulation state.
        
        Args:
            new_state: New state to set
        """
        with self._lock:
            self.state = new_state
            if new_state == SimulationState.RUNNING and not self.started_at:
                self.started_at = datetime.now()
            elif new_state in [SimulationState.COMPLETED, SimulationState.FAILED, SimulationState.STOPPED]:
                self.completed_at = datetime.now()
    
    def update_progress(self, current_step: int, total_steps: int) -> None:
        """
        Update simulation progress.
        
        Args:
            current_step: Current simulation step
            total_steps: Total number of steps
        """
        with self._lock:
            self.current_step = current_step
            self.total_steps = total_steps
            if total_steps > 0:
                self.progress_percentage = (current_step / total_steps) * 100.0
    
    def add_log(self, message: str) -> None:
        """
        Add a log message.
        
        Args:
            message: Log message to add
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            self.logs.append(f"[{timestamp}] {message}")
    
    def set_error(self, error_message: str) -> None:
        """
        Set an error message.
        
        Args:
            error_message: Error message to set
        """
        with self._lock:
            self.error_message = error_message
            self.state = SimulationState.FAILED
    
    def add_result(self, key: str, value: Any) -> None:
        """
        Add a result value.
        
        Args:
            key: Result key
            value: Result value
        """
        with self._lock:
            self.results[key] = value
    
    def should_stop(self) -> bool:
        """
        Check if simulation should stop.
        
        Returns:
            True if stop event is set
        """
        return self._stop_event.is_set()
    
    def stop(self) -> None:
        """Signal the simulation to stop."""
        self._stop_event.set()
        self.update_state(SimulationState.STOPPED)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert simulation to dictionary for serialization.
        
        Returns:
            Dictionary representation of the simulation
        """
        with self._lock:
            return {
                "simulation_id": self.simulation_id,
                "name": self.name,
                "state": self.state,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "progress_percentage": self.progress_percentage,
                "error_message": self.error_message,
                "config": self.config,
                "results": self.results,
                "logs": self.logs[-100:]  # Keep only last 100 log entries
            }
    
    def save_state(self) -> None:
        """Save simulation state to file."""
        state_file = self.work_dir / "simulation_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def load_state(self) -> None:
        """Load simulation state from file."""
        state_file = self.work_dir / "simulation_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
                # Restore state from saved data
                self.state = data.get("state", SimulationState.CREATED)
                self.current_step = data.get("current_step", 0)
                self.total_steps = data.get("total_steps", 0)
                self.progress_percentage = data.get("progress_percentage", 0.0)
                self.error_message = data.get("error_message")
                self.results = data.get("results", {})
                self.logs = data.get("logs", [])


class SimulationManager:
    """
    Manages multiple simulation instances.
    
    This class provides functionality for:
    - Creating and managing simulation instances
    - Tracking simulation states and progress
    - Providing simulation status and results
    - Managing simulation workflows
    """
    
    def __init__(self, work_dir: Path):
        """
        Initialize the simulation manager.
        
        Args:
            work_dir: Base working directory for all simulations
        """
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        
        # Simulation storage
        self._simulations: Dict[str, Simulation] = {}
        self._lock = threading.Lock()
        
        # Load existing simulations
        self._load_existing_simulations()
        
        logger.info(f"Simulation manager initialized with work directory: {work_dir}")
    
    def _load_existing_simulations(self) -> None:
        """Load existing simulations from disk."""
        for sim_dir in self.work_dir.iterdir():
            if sim_dir.is_dir():
                state_file = sim_dir / "simulation_state.json"
                if state_file.exists():
                    try:
                        with open(state_file, 'r') as f:
                            data = json.load(f)
                        
                        simulation_id = data.get("simulation_id")
                        name = data.get("name", "Unknown")
                        config = data.get("config", {})
                        
                        if simulation_id:
                            sim = Simulation(simulation_id, name, config, self.work_dir)
                            sim.load_state()
                            self._simulations[simulation_id] = sim
                            logger.info(f"Loaded existing simulation: {name} ({simulation_id})")
                    except Exception as e:
                        logger.error(f"Failed to load simulation from {sim_dir}: {e}")
    
    def create_simulation(
        self,
        name: str,
        config: Dict[str, Any],
        simulation_id: Optional[str] = None
    ) -> str:
        """
        Create a new simulation.
        
        Args:
            name: Human-readable name for the simulation
            config: Simulation configuration
            simulation_id: Optional custom simulation ID
            
        Returns:
            Simulation ID
        """
        if simulation_id is None:
            simulation_id = str(uuid4())
        
        with self._lock:
            if simulation_id in self._simulations:
                raise ValueError(f"Simulation with ID {simulation_id} already exists")
            
            sim = Simulation(simulation_id, name, config, self.work_dir)
            self._simulations[simulation_id] = sim
            sim.save_state()
            
            logger.info(f"Created simulation: {name} ({simulation_id})")
            return simulation_id
    
    def get_simulation(self, simulation_id: str) -> Optional[Simulation]:
        """
        Get a simulation by ID.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation instance or None if not found
        """
        with self._lock:
            return self._simulations.get(simulation_id)
    
    def list_simulations(self) -> List[Dict[str, Any]]:
        """
        List all simulations.
        
        Returns:
            List of simulation information dictionaries
        """
        with self._lock:
            # Get simulation IDs to avoid holding lock during to_dict() calls
            simulation_ids = list(self._simulations.keys())
        
        # Call to_dict() outside of manager lock to avoid deadlock
        results = []
        for sim_id in simulation_ids:
            sim = self.get_simulation(sim_id)
            if sim:
                results.append(sim.to_dict())
        
        return results
    
    def get_active_simulations(self) -> List[str]:
        """
        Get IDs of active simulations.
        
        Returns:
            List of active simulation IDs
        """
        with self._lock:
            active_states = {SimulationState.CREATED, SimulationState.SETUP, SimulationState.RUNNING, SimulationState.PAUSED}
            return [
                sim_id for sim_id, sim in self._simulations.items()
                if sim.state in active_states
            ]
    
    def delete_simulation(self, simulation_id: str) -> bool:
        """
        Delete a simulation.
        
        Args:
            simulation_id: Simulation ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if simulation_id not in self._simulations:
                return False
            
            sim = self._simulations[simulation_id]
            
            # Stop if running
            if sim.state == SimulationState.RUNNING:
                sim.stop()
            
            # Remove from memory
            del self._simulations[simulation_id]
            
            # Remove from disk (optional - could keep for archival)
            # import shutil
            # shutil.rmtree(sim.work_dir, ignore_errors=True)
            
            logger.info(f"Deleted simulation: {sim.name} ({simulation_id})")
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall status of the simulation manager.
        
        Returns:
            Status information dictionary
        """
        with self._lock:
            total_simulations = len(self._simulations)
            
            # Calculate active simulations directly to avoid deadlock
            active_states = {SimulationState.CREATED, SimulationState.SETUP, SimulationState.RUNNING, SimulationState.PAUSED}
            active_simulations = len([
                sim_id for sim_id, sim in self._simulations.items()
                if sim.state in active_states
            ])
            
            state_counts = {}
            for sim in self._simulations.values():
                state_counts[sim.state] = state_counts.get(sim.state, 0) + 1
            
            return {
                "total_simulations": total_simulations,
                "active_simulations": active_simulations,
                "state_counts": state_counts,
                "work_directory": str(self.work_dir)
            }
    
    def cleanup_completed_simulations(self, max_age_days: int = 30) -> int:
        """
        Clean up old completed simulations.
        
        Args:
            max_age_days: Maximum age in days for completed simulations
            
        Returns:
            Number of simulations cleaned up
        """
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        cleaned_count = 0
        
        with self._lock:
            to_delete = []
            for sim_id, sim in self._simulations.items():
                if (sim.state in [SimulationState.COMPLETED, SimulationState.FAILED] and
                    sim.completed_at and sim.completed_at.timestamp() < cutoff_date):
                    to_delete.append(sim_id)
            
            # Delete simulations directly to avoid deadlock
            for sim_id in to_delete:
                if sim_id in self._simulations:
                    sim = self._simulations[sim_id]
                    
                    # Stop if running
                    if sim.state == SimulationState.RUNNING:
                        sim.stop()
                    
                    # Remove from memory
                    del self._simulations[sim_id]
                    cleaned_count += 1
                    
                    logger.info(f"Cleaned up simulation: {sim.name} ({sim_id})")
        
        logger.info(f"Cleaned up {cleaned_count} old completed simulations")
        return cleaned_count
    
    def save_all_states(self) -> None:
        """Save state of all simulations to disk."""
        with self._lock:
            # Get simulation IDs to avoid holding lock during save_state() calls
            simulation_ids = list(self._simulations.keys())
        
        # Call save_state() outside of manager lock to avoid deadlock
        for sim_id in simulation_ids:
            sim = self.get_simulation(sim_id)
            if sim:
                sim.save_state()
    
    def get_simulation_summary(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a simulation.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation summary or None if not found
        """
        sim = self.get_simulation(simulation_id)
        if not sim:
            return None
        
        # Use to_dict() which already handles the lock properly
        summary = sim.to_dict()
        
        # Extract only the summary fields we want
        return {
            "simulation_id": summary["simulation_id"],
            "name": summary["name"],
            "state": summary["state"],
            "progress": {
                "current_step": summary["current_step"],
                "total_steps": summary["total_steps"],
                "percentage": summary["progress_percentage"]
            },
            "timing": {
                "created": summary["created_at"],
                "started": summary["started_at"],
                "completed": summary["completed_at"]
            },
            "error": summary["error_message"],
            "results_count": len(summary["results"]),
            "logs_count": len(summary["logs"])
        } 