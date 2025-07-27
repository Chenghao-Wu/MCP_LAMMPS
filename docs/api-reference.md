# API Reference

This document provides a complete reference for all functions and classes available in the MCP LAMMPS package.

## Core Functions

### Simulation Creation

#### `create_simulation`

Create a new LAMMPS simulation with basic parameters.

```python
create_simulation(
    name: str,
    structure_file: Optional[str] = None,
    force_field: str = "lj/cut",
    temperature: float = 300,
    pressure: float = 1,
    timestep: float = 0.001,
    equilibration_steps: int = 1000,
    production_steps: int = 10000
) -> Simulation
```

**Parameters:**
- `name` (str): Name of the simulation
- `structure_file` (str, optional): Path to structure file
- `force_field` (str): Force field to use (default: "lj/cut")
- `temperature` (float): Temperature in Kelvin (default: 300)
- `pressure` (float): Pressure in atm (default: 1)
- `timestep` (float): Simulation timestep in ps (default: 0.001)
- `equilibration_steps` (int): Number of equilibration steps (default: 1000)
- `production_steps` (int): Number of production steps (default: 10000)

**Returns:**
- `Simulation`: Simulation object with unique ID

#### `create_water_simulation`

Create a water simulation using TIP3P force field.

```python
create_water_simulation(
    name: str,
    num_molecules: int = 100,
    box_size: float = 30,
    temperature: float = 300,
    pressure: float = 1,
    timestep: float = 0.001,
    equilibration_steps: int = 1000,
    production_steps: int = 10000
) -> Simulation
```

**Parameters:**
- `name` (str): Name of the simulation
- `num_molecules` (int): Number of water molecules (default: 100)
- `box_size` (float): Simulation box size in Å (default: 30)
- `temperature` (float): Temperature in Kelvin (default: 300)
- `pressure` (float): Pressure in atm (default: 1)
- `timestep` (float): Simulation timestep in ps (default: 0.001)
- `equilibration_steps` (int): Number of equilibration steps (default: 1000)
- `production_steps` (int): Number of production steps (default: 10000)

**Returns:**
- `Simulation`: Simulation object with unique ID

### Structure Management

#### `load_structure`

Load a molecular structure from a file.

```python
load_structure(
    filename: str,
    content: str,
    file_type: str = "data"
) -> Structure
```

**Parameters:**
- `filename` (str): Name of the structure file
- `content` (str): File content as string
- `file_type` (str): Type of file ("data", "pdb", "xyz") (default: "data")

**Returns:**
- `Structure`: Structure object

#### `create_simple_structure`

Create a simple structure for testing.

```python
create_simple_structure(
    num_atoms: int = 100,
    box_size: float = 20,
    atom_type: int = 1
) -> Structure
```

**Parameters:**
- `num_atoms` (int): Number of atoms (default: 100)
- `box_size` (float): Simulation box size in Å (default: 20)
- `atom_type` (int): Atom type ID (default: 1)

**Returns:**
- `Structure`: Structure object

### Simulation Control

#### `run_simulation`

Run a LAMMPS simulation.

```python
run_simulation(simulation_id: str) -> bool
```

**Parameters:**
- `simulation_id` (str): ID of the simulation to run

**Returns:**
- `bool`: True if simulation started successfully

#### `run_equilibration`

Run equilibration phase of a simulation.

```python
run_equilibration(
    simulation_id: str,
    temperature: float = 300,
    duration: int = 1000
) -> bool
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `temperature` (float): Temperature in Kelvin (default: 300)
- `duration` (int): Number of steps (default: 1000)

**Returns:**
- `bool`: True if equilibration started successfully

#### `run_production`

Run production phase of a simulation.

```python
run_production(
    simulation_id: str,
    temperature: float = 300,
    pressure: float = 1,
    duration: int = 10000
) -> bool
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `temperature` (float): Temperature in Kelvin (default: 300)
- `pressure` (float): Pressure in atm (default: 1)
- `duration` (int): Number of steps (default: 10000)

**Returns:**
- `bool`: True if production started successfully

#### `stop_simulation`

Stop a running simulation.

```python
stop_simulation(simulation_id: str) -> bool
```

**Parameters:**
- `simulation_id` (str): ID of the simulation to stop

**Returns:**
- `bool`: True if simulation stopped successfully

#### `pause_simulation`

Pause a running simulation.

```python
pause_simulation(simulation_id: str) -> bool
```

**Parameters:**
- `simulation_id` (str): ID of the simulation to pause

**Returns:**
- `bool`: True if simulation paused successfully

#### `resume_simulation`

Resume a paused simulation.

```python
resume_simulation(simulation_id: str) -> bool
```

**Parameters:**
- `simulation_id` (str): ID of the simulation to resume

**Returns:**
- `bool`: True if simulation resumed successfully

### Monitoring and Status

#### `get_simulation_status`

Get the current status of a simulation.

```python
get_simulation_status(simulation_id: str) -> str
```

**Parameters:**
- `simulation_id` (str): ID of the simulation

**Returns:**
- `str`: Status string ("running", "paused", "completed", "error")

#### `get_simulation_progress`

Get the current progress of a simulation.

```python
get_simulation_progress(simulation_id: str) -> float
```

**Parameters:**
- `simulation_id` (str): ID of the simulation

**Returns:**
- `float`: Progress percentage (0.0 to 100.0)

#### `get_system_properties`

Get real-time system properties from a running simulation.

```python
get_system_properties(simulation_id: str) -> SystemProperties
```

**Parameters:**
- `simulation_id` (str): ID of the simulation

**Returns:**
- `SystemProperties`: Object containing temperature, pressure, volume, etc.

#### `get_energy_data`

Get energy components from a running simulation.

```python
get_energy_data(simulation_id: str) -> EnergyData
```

**Parameters:**
- `simulation_id` (str): ID of the simulation

**Returns:**
- `EnergyData`: Object containing kinetic, potential, and total energy

#### `get_structure_data`

Get structural information from a running simulation.

```python
get_structure_data(simulation_id: str) -> StructureData
```

**Parameters:**
- `simulation_id` (str): ID of the simulation

**Returns:**
- `StructureData`: Object containing atomic positions, velocities, etc.

#### `monitor_simulation`

Monitor a simulation for a specified duration.

```python
monitor_simulation(
    simulation_id: str,
    duration_seconds: int = 60,
    interval_seconds: int = 5
) -> List[MonitoringData]
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `duration_seconds` (int): Duration to monitor in seconds (default: 60)
- `interval_seconds` (int): Monitoring interval in seconds (default: 5)

**Returns:**
- `List[MonitoringData]`: List of monitoring data points

### Analysis Functions

#### `analyze_trajectory`

Analyze trajectory data from a simulation.

```python
analyze_trajectory(
    simulation_id: str,
    trajectory_file: Optional[str] = None
) -> TrajectoryAnalysis
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `trajectory_file` (str, optional): Path to trajectory file

**Returns:**
- `TrajectoryAnalysis`: Object containing RMSD, radius of gyration, etc.

#### `calculate_properties`

Calculate thermodynamic properties from simulation data.

```python
calculate_properties(
    simulation_id: str,
    log_file: Optional[str] = None
) -> ThermodynamicProperties
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `log_file` (str, optional): Path to log file

**Returns:**
- `ThermodynamicProperties`: Object containing average temperature, pressure, energy, etc.

#### `plot_results`

Generate plots from simulation results.

```python
plot_results(
    simulation_id: str,
    plot_type: str = "thermo",
    log_file: Optional[str] = None
) -> str
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `plot_type` (str): Type of plot ("thermo", "structure", "energy") (default: "thermo")
- `log_file` (str, optional): Path to log file

**Returns:**
- `str`: Path to generated plot file

#### `export_data`

Export simulation data in various formats.

```python
export_data(
    simulation_id: str,
    data_type: str = "thermo",
    format: str = "csv",
    log_file: Optional[str] = None
) -> str
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `data_type` (str): Type of data ("thermo", "trajectory", "structure") (default: "thermo")
- `format` (str): Export format ("csv", "json", "hdf5") (default: "csv")
- `log_file` (str, optional): Path to log file

**Returns:**
- `str`: Path to exported data file

### Management Functions

#### `list_simulations`

List all available simulations.

```python
list_simulations() -> List[SimulationInfo]
```

**Returns:**
- `List[SimulationInfo]`: List of simulation information objects

#### `get_simulation_info`

Get detailed information about a simulation.

```python
get_simulation_info(simulation_id: str) -> SimulationInfo
```

**Parameters:**
- `simulation_id` (str): ID of the simulation

**Returns:**
- `SimulationInfo`: Detailed simulation information

#### `get_simulation_results`

Get all results from a simulation.

```python
get_simulation_results(simulation_id: str) -> SimulationResults
```

**Parameters:**
- `simulation_id` (str): ID of the simulation

**Returns:**
- `SimulationResults`: Complete simulation results

#### `delete_simulation`

Delete a simulation.

```python
delete_simulation(simulation_id: str) -> bool
```

**Parameters:**
- `simulation_id` (str): ID of the simulation to delete

**Returns:**
- `bool`: True if simulation deleted successfully

#### `get_active_simulations`

Get list of all active simulations.

```python
get_active_simulations() -> List[str]
```

**Returns:**
- `List[str]`: List of active simulation IDs

#### `get_simulation_logs`

Get recent logs from a simulation.

```python
get_simulation_logs(
    simulation_id: str,
    num_entries: int = 50
) -> List[LogEntry]
```

**Parameters:**
- `simulation_id` (str): ID of the simulation
- `num_entries` (int): Number of log entries to retrieve (default: 50)

**Returns:**
- `List[LogEntry]`: List of log entries

### Utility Functions

#### `health_check`

Check the health status of the LAMMPS MCP server.

```python
health_check() -> HealthStatus
```

**Returns:**
- `HealthStatus`: Server health information

## Data Classes

### Simulation

```python
class Simulation:
    id: str
    name: str
    status: str
    created_at: datetime
    parameters: Dict[str, Any]
```

### Structure

```python
class Structure:
    id: str
    filename: str
    file_type: str
    num_atoms: int
    box_size: Tuple[float, float, float]
```

### SystemProperties

```python
class SystemProperties:
    temperature: float
    pressure: float
    volume: float
    density: float
    step: int
    time: float
```

### EnergyData

```python
class EnergyData:
    kinetic: float
    potential: float
    total: float
    step: int
    time: float
```

### StructureData

```python
class StructureData:
    positions: List[Tuple[float, float, float]]
    velocities: List[Tuple[float, float, float]]
    types: List[int]
    box_size: Tuple[float, float, float]
```

### TrajectoryAnalysis

```python
class TrajectoryAnalysis:
    rmsd: List[float]
    radius_of_gyration: List[float]
    end_to_end_distance: List[float]
    time_points: List[float]
```

### ThermodynamicProperties

```python
class ThermodynamicProperties:
    avg_temperature: float
    avg_pressure: float
    avg_energy: float
    std_temperature: float
    std_pressure: float
    std_energy: float
```

### SimulationInfo

```python
class SimulationInfo:
    id: str
    name: str
    status: str
    created_at: datetime
    parameters: Dict[str, Any]
    files: List[str]
```

### SimulationResults

```python
class SimulationResults:
    simulation_id: str
    trajectory_file: str
    log_file: str
    analysis: TrajectoryAnalysis
    properties: ThermodynamicProperties
    plots: List[str]
```

### LogEntry

```python
class LogEntry:
    timestamp: datetime
    level: str
    message: str
    simulation_id: str
```

### HealthStatus

```python
class HealthStatus:
    status: str
    version: str
    uptime: float
    active_simulations: int
    memory_usage: float
```

## Error Handling

All functions may raise the following exceptions:

### `SimulationError`

Raised when a simulation operation fails.

```python
class SimulationError(Exception):
    def __init__(self, message: str, simulation_id: str = None):
        self.message = message
        self.simulation_id = simulation_id
```

### `ConfigurationError`

Raised when there's a configuration issue.

```python
class ConfigurationError(Exception):
    def __init__(self, message: str):
        self.message = message
```

### `FileError`

Raised when there's a file I/O error.

```python
class FileError(Exception):
    def __init__(self, message: str, filename: str = None):
        self.message = message
        self.filename = filename
```

## Examples

### Basic Usage

```python
from mcp_lammps import create_simulation, run_simulation, get_simulation_status

# Create a simulation
sim = create_simulation(
    name="test_sim",
    temperature=300,
    pressure=1.0
)

# Run the simulation
run_simulation(sim.id)

# Check status
status = get_simulation_status(sim.id)
print(f"Simulation status: {status}")
```

### Advanced Usage

```python
from mcp_lammps import (
    create_water_simulation,
    run_equilibration,
    run_production,
    analyze_trajectory,
    plot_results
)

# Create water simulation
sim = create_water_simulation(
    name="water_test",
    num_molecules=100,
    temperature=300
)

# Run equilibration
run_equilibration(sim.id, temperature=300, duration=1000)

# Run production
run_production(sim.id, temperature=300, pressure=1.0, duration=10000)

# Analyze results
analysis = analyze_trajectory(sim.id)
plot_results(sim.id, plot_type="thermo")
```

For more examples, see the [Examples](examples.md) section. 