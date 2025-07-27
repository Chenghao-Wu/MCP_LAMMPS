# User Guide

This guide will help you get started with MCP LAMMPS and learn how to use it effectively for molecular dynamics simulations.

## Quick Start

### Starting the Server

1. **Basic server start:**
   ```bash
   python -m mcp_lammps.server
   ```

2. **With custom configuration:**
   ```bash
   export MCP_LAMMPS_WORK_DIR=/path/to/workspace
   export MCP_LAMMPS_LOG_LEVEL=INFO
   python -m mcp_lammps.server
   ```

3. **Check server status:**
   ```bash
   curl http://localhost:8000/health
   ```

### Basic Workflow

The typical workflow for using MCP LAMMPS involves:

1. **Create a simulation** - Set up the molecular system
2. **Configure parameters** - Define simulation conditions
3. **Run equilibration** - Prepare the system
4. **Run production** - Collect data
5. **Analyze results** - Process and visualize data

## Simulation Management

### Creating Simulations

#### Simple Structure Creation

```python
# Create a simple test structure
from mcp_lammps import create_simple_structure

structure = create_simple_structure(
    num_atoms=100,
    box_size=20.0,
    atom_type=1
)
```

#### Water Simulation

```python
# Create a water simulation
from mcp_lammps import create_water_simulation

simulation = create_water_simulation(
    name="water_test",
    num_molecules=100,
    box_size=30.0,
    temperature=300,
    pressure=1.0
)
```

#### Custom Structure

```python
# Load a custom structure
from mcp_lammps import load_structure

structure = load_structure(
    filename="molecule.data",
    content="""# LAMMPS data file
    100 atoms
    1 atom types
    
    0.0 20.0 xlo xhi
    0.0 20.0 ylo yhi
    0.0 20.0 zlo zhi
    
    Atoms
    
    1 1 10.0 10.0 10.0
    # ... more atoms
    """,
    file_type="data"
)
```

### Running Simulations

#### Basic Simulation

```python
# Create and run a simulation
simulation = create_simulation(
    name="my_simulation",
    temperature=300,
    pressure=1.0,
    timestep=0.001
)

# Run the simulation
run_simulation(simulation_id=simulation.id)
```

#### Equilibration and Production

```python
# Run equilibration phase
run_equilibration(
    simulation_id=simulation.id,
    temperature=300,
    duration=1000
)

# Run production phase
run_production(
    simulation_id=simulation.id,
    temperature=300,
    pressure=1.0,
    duration=10000
)
```

### Monitoring Simulations

#### Check Status

```python
# Get simulation status
status = get_simulation_status(simulation_id=simulation.id)
print(f"Status: {status}")

# Get progress
progress = get_simulation_progress(simulation_id=simulation.id)
print(f"Progress: {progress}%")
```

#### Real-time Monitoring

```python
# Monitor simulation for 60 seconds
monitor_simulation(
    simulation_id=simulation.id,
    duration_seconds=60,
    interval_seconds=5
)

# Get system properties
properties = get_system_properties(simulation_id=simulation.id)
print(f"Temperature: {properties.temperature} K")
print(f"Pressure: {properties.pressure} atm")

# Get energy data
energy = get_energy_data(simulation_id=simulation.id)
print(f"Total Energy: {energy.total} kcal/mol")
print(f"Potential Energy: {energy.potential} kcal/mol")
```

## Analysis and Results

### Trajectory Analysis

```python
# Analyze trajectory
analysis = analyze_trajectory(simulation_id=simulation.id)
print(f"RMSD: {analysis.rmsd}")
print(f"Radius of Gyration: {analysis.radius_of_gyration}")
```

### Thermodynamic Properties

```python
# Calculate properties
properties = calculate_properties(simulation_id=simulation.id)
print(f"Average Temperature: {properties.avg_temperature} K")
print(f"Average Pressure: {properties.avg_pressure} atm")
print(f"Average Energy: {properties.avg_energy} kcal/mol")
```

### Plotting Results

```python
# Generate plots
plot_results(
    simulation_id=simulation.id,
    plot_type="thermo"  # Options: thermo, structure, energy
)
```

### Exporting Data

```python
# Export data in various formats
export_data(
    simulation_id=simulation.id,
    data_type="thermo",  # Options: thermo, trajectory, structure
    format="csv"         # Options: csv, json, hdf5
)
```

## Advanced Features

### Multiple Simulations

```python
# List all simulations
simulations = list_simulations()

# Get simulation info
info = get_simulation_info(simulation_id=simulation.id)

# Delete simulation
delete_simulation(simulation_id=simulation.id)
```

### Simulation Control

```python
# Pause simulation
pause_simulation(simulation_id=simulation.id)

# Resume simulation
resume_simulation(simulation_id=simulation.id)

# Stop simulation
stop_simulation(simulation_id=simulation.id)
```

### Active Simulations

```python
# Get active simulations
active = get_active_simulations()

# Get simulation logs
logs = get_simulation_logs(
    simulation_id=simulation.id,
    num_entries=50
)
```

## Best Practices

### Performance Optimization

1. **Use appropriate system sizes:**
   - Start with small systems for testing
   - Scale up gradually for production runs

2. **Optimize simulation parameters:**
   - Use appropriate timesteps (0.001-0.002 fs for atomistic)
   - Choose suitable ensembles (NVT for equilibration, NPT for production)

3. **Monitor resource usage:**
   - Check memory usage during simulations
   - Monitor disk space for trajectory files

### Data Management

1. **Organize your workspace:**
   ```
   workspace/
   ├── simulations/
   │   ├── water_sim/
   │   ├── protein_sim/
   │   └── ...
   ├── structures/
   ├── results/
   └── logs/
   ```

2. **Backup important data:**
   - Use the built-in backup feature
   - Keep copies of input files and configurations

3. **Clean up old simulations:**
   - Delete completed simulations to save space
   - Archive important results

### Error Handling

1. **Check simulation status regularly:**
   - Monitor for convergence issues
   - Watch for energy drift

2. **Handle common errors:**
   - System crashes: Check memory and restart
   - Convergence issues: Adjust parameters
   - File I/O errors: Check disk space and permissions

## Common Workflows

### Protein Simulation

```python
# 1. Load protein structure
protein = load_structure(filename="protein.pdb")

# 2. Create simulation
sim = create_simulation(
    name="protein_sim",
    structure_file=protein,
    force_field="charmm",
    temperature=300
)

# 3. Equilibration
run_equilibration(sim.id, temperature=300, duration=5000)

# 4. Production
run_production(sim.id, temperature=300, pressure=1.0, duration=50000)

# 5. Analysis
analyze_trajectory(sim.id)
plot_results(sim.id, plot_type="structure")
```

### Material Properties

```python
# 1. Create material structure
material = create_simple_structure(num_atoms=1000, box_size=50.0)

# 2. Run at different temperatures
temperatures = [100, 200, 300, 400, 500]
for temp in temperatures:
    sim = create_simulation(
        name=f"material_{temp}K",
        temperature=temp
    )
    run_production(sim.id, temperature=temp, duration=10000)

# 3. Analyze temperature dependence
# ... analysis code
```

## Troubleshooting

### Common Issues

1. **Simulation not starting:**
   - Check LAMMPS installation
   - Verify input files
   - Check system resources

2. **Simulation crashes:**
   - Reduce system size
   - Check for bad contacts
   - Adjust simulation parameters

3. **Poor convergence:**
   - Increase equilibration time
   - Check force field parameters
   - Verify system setup

### Getting Help

- Check the [API Reference](api-reference.md) for detailed function documentation
- Look at [Examples](examples.md) for working code samples
- Report issues on [GitHub](https://github.com/mcp-lammps/mcp-lammps/issues)

## Next Steps

- Explore the [API Reference](api-reference.md) for advanced features
- Try the [Examples](examples.md) to see more complex workflows
- Check the [Configuration](configuration.md) guide for advanced setup options 