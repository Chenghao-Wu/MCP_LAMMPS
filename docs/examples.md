# Examples

This page contains working examples and tutorials for using MCP LAMMPS. Each example demonstrates different aspects of the package and can be used as a starting point for your own simulations.

## Basic Examples

### Example 1: Simple Water Simulation

Create and run a basic water simulation with 10 TIP3P water molecules.

```python
from mcp_lammps import (
    create_water_simulation,
    run_simulation,
    get_simulation_status,
    get_simulation_results
)

# Create a water simulation
sim = create_water_simulation(
    name="water_basic",
    num_molecules=10,
    box_size=15.0,
    temperature=300,
    pressure=1.0,
    timestep=0.001,
    equilibration_steps=1000,
    production_steps=5000
)

print(f"Created simulation: {sim.id}")

# Run the simulation
run_simulation(sim.id)

# Monitor the simulation
while True:
    status = get_simulation_status(sim.id)
    print(f"Status: {status}")
    
    if status == "completed":
        break
    elif status == "error":
        print("Simulation failed!")
        break
    
    import time
    time.sleep(5)

# Get results
results = get_simulation_results(sim.id)
print(f"Simulation completed. Files: {results.files}")
```

### Example 2: Custom Structure Simulation

Load a custom molecular structure and run a simulation.

```python
from mcp_lammps import (
    load_structure,
    create_simulation,
    run_equilibration,
    run_production,
    analyze_trajectory
)

# Define a simple molecular structure (LAMMPS data format)
structure_content = """# LAMMPS data file for simple molecule
50 atoms
1 atom types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Atoms

1 1 10.0 10.0 10.0
2 1 10.5 10.0 10.0
3 1 11.0 10.0 10.0
# ... add more atoms as needed
"""

# Load the structure
structure = load_structure(
    filename="custom_molecule.data",
    content=structure_content,
    file_type="data"
)

# Create simulation with the custom structure
sim = create_simulation(
    name="custom_structure_sim",
    structure_file=structure.id,
    force_field="lj/cut",
    temperature=300,
    pressure=1.0,
    timestep=0.001
)

# Run equilibration
run_equilibration(sim.id, temperature=300, duration=2000)

# Run production
run_production(sim.id, temperature=300, pressure=1.0, duration=10000)

# Analyze trajectory
analysis = analyze_trajectory(sim.id)
print(f"RMSD range: {min(analysis.rmsd):.3f} - {max(analysis.rmsd):.3f}")
```

### Example 3: Real-time Monitoring

Monitor a simulation in real-time and collect data.

```python
from mcp_lammps import (
    create_water_simulation,
    run_simulation,
    monitor_simulation,
    get_system_properties,
    get_energy_data
)

# Create and start simulation
sim = create_water_simulation(
    name="monitoring_test",
    num_molecules=50,
    temperature=300
)

run_simulation(sim.id)

# Monitor for 60 seconds with 5-second intervals
monitoring_data = monitor_simulation(
    simulation_id=sim.id,
    duration_seconds=60,
    interval_seconds=5
)

# Print monitoring results
for i, data in enumerate(monitoring_data):
    print(f"Step {i+1}:")
    print(f"  Temperature: {data.temperature:.2f} K")
    print(f"  Pressure: {data.pressure:.2f} atm")
    print(f"  Energy: {data.energy:.2f} kcal/mol")
    print()

# Get current system properties
properties = get_system_properties(sim.id)
print(f"Current temperature: {properties.temperature:.2f} K")
print(f"Current pressure: {properties.pressure:.2f} atm")
print(f"Current volume: {properties.volume:.2f} Å³")

# Get energy components
energy = get_energy_data(sim.id)
print(f"Kinetic energy: {energy.kinetic:.2f} kcal/mol")
print(f"Potential energy: {energy.potential:.2f} kcal/mol")
print(f"Total energy: {energy.total:.2f} kcal/mol")
```

## Advanced Examples

### Example 4: Protein Simulation Workflow

Complete workflow for protein simulation including equilibration and analysis.

```python
from mcp_lammps import (
    load_structure,
    create_simulation,
    run_equilibration,
    run_production,
    analyze_trajectory,
    calculate_properties,
    plot_results,
    export_data
)

# Load protein structure (PDB format)
protein_content = """ATOM      1  N   ALA A   1      27.462  14.410   5.000  1.00  0.00
ATOM      2  CA  ALA A   1      26.213  13.823   5.000  1.00  0.00
ATOM      3  C   ALA A   1      25.000  14.705   5.000  1.00  0.00
ATOM      4  O   ALA A   1      24.000  14.307   5.000  1.00  0.00
# ... more atoms
"""

structure = load_structure(
    filename="protein.pdb",
    content=protein_content,
    file_type="pdb"
)

# Create protein simulation
sim = create_simulation(
    name="protein_simulation",
    structure_file=structure.id,
    force_field="charmm",
    temperature=300,
    pressure=1.0,
    timestep=0.002,  # 2 fs timestep for protein
    equilibration_steps=5000,
    production_steps=50000
)

print(f"Starting protein simulation: {sim.id}")

# Step 1: Energy minimization and equilibration
print("Running equilibration...")
run_equilibration(sim.id, temperature=300, duration=5000)

# Step 2: Production run
print("Running production...")
run_production(sim.id, temperature=300, pressure=1.0, duration=50000)

# Step 3: Analysis
print("Analyzing trajectory...")
analysis = analyze_trajectory(sim.id)

print("Trajectory Analysis Results:")
print(f"RMSD: {analysis.rmsd[-1]:.3f} Å")
print(f"Radius of gyration: {analysis.radius_of_gyration[-1]:.3f} Å")

# Step 4: Calculate thermodynamic properties
properties = calculate_properties(sim.id)
print("\nThermodynamic Properties:")
print(f"Average temperature: {properties.avg_temperature:.2f} K")
print(f"Average pressure: {properties.avg_pressure:.2f} atm")
print(f"Average energy: {properties.avg_energy:.2f} kcal/mol")

# Step 5: Generate plots
plot_file = plot_results(sim.id, plot_type="thermo")
print(f"Thermodynamic plot saved: {plot_file}")

# Step 6: Export data
export_file = export_data(sim.id, data_type="thermo", format="csv")
print(f"Data exported: {export_file}")
```

### Example 5: Temperature Dependence Study

Run multiple simulations at different temperatures to study temperature dependence.

```python
from mcp_lammps import (
    create_water_simulation,
    run_simulation,
    get_simulation_results,
    calculate_properties,
    list_simulations
)

# Define temperature range
temperatures = [250, 275, 300, 325, 350, 375, 400]
simulations = []

# Create simulations at different temperatures
for temp in temperatures:
    sim = create_water_simulation(
        name=f"water_{temp}K",
        num_molecules=100,
        temperature=temp,
        pressure=1.0,
        equilibration_steps=2000,
        production_steps=10000
    )
    simulations.append((temp, sim))
    print(f"Created simulation at {temp}K: {sim.id}")

# Run all simulations
for temp, sim in simulations:
    print(f"Running simulation at {temp}K...")
    run_simulation(sim.id)

# Wait for all simulations to complete
import time
while True:
    all_completed = True
    for temp, sim in simulations:
        status = get_simulation_status(sim.id)
        if status != "completed":
            all_completed = False
            break
    
    if all_completed:
        break
    
    time.sleep(10)

# Analyze results
results = {}
for temp, sim in simulations:
    properties = calculate_properties(sim.id)
    results[temp] = {
        'avg_temperature': properties.avg_temperature,
        'avg_pressure': properties.avg_pressure,
        'avg_energy': properties.avg_energy,
        'std_energy': properties.std_energy
    }

# Print temperature dependence
print("\nTemperature Dependence Results:")
print("Temp (K) | Avg Energy (kcal/mol) | Std Energy")
print("-" * 45)
for temp in sorted(results.keys()):
    data = results[temp]
    print(f"{temp:7.0f} | {data['avg_energy']:16.2f} | {data['std_energy']:9.2f}")

# Export combined results
import pandas as pd
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('temperature_dependence.csv')
print("\nResults saved to temperature_dependence.csv")
```

### Example 6: Material Properties Calculation

Calculate material properties using molecular dynamics.

```python
from mcp_lammps import (
    create_simple_structure,
    create_simulation,
    run_equilibration,
    run_production,
    calculate_properties,
    get_system_properties
)

# Create a simple material structure (e.g., Lennard-Jones fluid)
material = create_simple_structure(
    num_atoms=1000,
    box_size=50.0,
    atom_type=1
)

# Create simulation for material properties
sim = create_simulation(
    name="material_properties",
    structure_file=material.id,
    force_field="lj/cut",
    temperature=300,
    pressure=1.0,
    timestep=0.001,
    equilibration_steps=5000,
    production_steps=20000
)

print(f"Starting material properties calculation: {sim.id}")

# Equilibration
run_equilibration(sim.id, temperature=300, duration=5000)

# Production run for property calculation
run_production(sim.id, temperature=300, pressure=1.0, duration=20000)

# Calculate properties
properties = calculate_properties(sim.id)

print("\nMaterial Properties:")
print(f"Average temperature: {properties.avg_temperature:.2f} K")
print(f"Average pressure: {properties.avg_pressure:.2f} atm")
print(f"Average energy: {properties.avg_energy:.2f} kcal/mol")

# Calculate density
system_props = get_system_properties(sim.id)
volume = system_props.volume  # in Å³
num_atoms = 1000
density = num_atoms / volume  # atoms/Å³
print(f"Average density: {density:.4f} atoms/Å³")

# Calculate specific heat (approximate)
# This would require multiple temperature simulations for accurate calculation
print(f"Energy fluctuations: {properties.std_energy:.4f} kcal/mol")
```

## Interactive Examples

### Example 7: Interactive Simulation Control

Interactive script for controlling simulations with user input.

```python
from mcp_lammps import (
    create_water_simulation,
    run_simulation,
    pause_simulation,
    resume_simulation,
    stop_simulation,
    get_simulation_status,
    get_system_properties
)

def interactive_simulation():
    # Create simulation
    sim = create_water_simulation(
        name="interactive_test",
        num_molecules=50,
        temperature=300
    )
    
    print(f"Created simulation: {sim.id}")
    
    # Start simulation
    run_simulation(sim.id)
    print("Simulation started. Commands: pause, resume, stop, status, properties, quit")
    
    while True:
        command = input("Enter command: ").lower().strip()
        
        if command == "quit":
            stop_simulation(sim.id)
            print("Simulation stopped. Exiting.")
            break
        elif command == "pause":
            pause_simulation(sim.id)
            print("Simulation paused.")
        elif command == "resume":
            resume_simulation(sim.id)
            print("Simulation resumed.")
        elif command == "stop":
            stop_simulation(sim.id)
            print("Simulation stopped.")
            break
        elif command == "status":
            status = get_simulation_status(sim.id)
            print(f"Status: {status}")
        elif command == "properties":
            try:
                props = get_system_properties(sim.id)
                print(f"Temperature: {props.temperature:.2f} K")
                print(f"Pressure: {props.pressure:.2f} atm")
                print(f"Volume: {props.volume:.2f} Å³")
            except Exception as e:
                print(f"Could not get properties: {e}")
        else:
            print("Unknown command. Try: pause, resume, stop, status, properties, quit")

if __name__ == "__main__":
    interactive_simulation()
```

### Example 8: Batch Processing

Process multiple simulations in batch mode.

```python
from mcp_lammps import (
    create_water_simulation,
    run_simulation,
    get_simulation_status,
    get_simulation_results,
    list_simulations
)
import time
import threading

def run_batch_simulations():
    # Define simulation parameters
    simulation_params = [
        {"name": "batch_1", "num_molecules": 50, "temperature": 280},
        {"name": "batch_2", "num_molecules": 50, "temperature": 300},
        {"name": "batch_3", "num_molecules": 50, "temperature": 320},
        {"name": "batch_4", "num_molecules": 100, "temperature": 300},
        {"name": "batch_5", "num_molecules": 150, "temperature": 300},
    ]
    
    simulations = []
    
    # Create all simulations
    for params in simulation_params:
        sim = create_water_simulation(**params)
        simulations.append(sim)
        print(f"Created: {sim.name} (ID: {sim.id})")
    
    # Start all simulations
    for sim in simulations:
        run_simulation(sim.id)
        print(f"Started: {sim.name}")
    
    # Monitor all simulations
    completed = []
    while len(completed) < len(simulations):
        for sim in simulations:
            if sim.id not in completed:
                status = get_simulation_status(sim.id)
                if status == "completed":
                    completed.append(sim.id)
                    print(f"Completed: {sim.name}")
                elif status == "error":
                    completed.append(sim.id)
                    print(f"Error: {sim.name}")
        
        if len(completed) < len(simulations):
            time.sleep(10)
    
    # Get results for all simulations
    print("\nBatch processing completed!")
    for sim in simulations:
        try:
            results = get_simulation_results(sim.id)
            print(f"{sim.name}: {len(results.files)} files generated")
        except Exception as e:
            print(f"{sim.name}: Error getting results - {e}")

if __name__ == "__main__":
    run_batch_simulations()
```

## Error Handling Examples

### Example 9: Robust Error Handling

Example showing how to handle errors gracefully.

```python
from mcp_lammps import (
    create_water_simulation,
    run_simulation,
    get_simulation_status,
    get_simulation_results
)

def robust_simulation():
    try:
        # Create simulation
        sim = create_water_simulation(
            name="robust_test",
            num_molecules=100,
            temperature=300
        )
        print(f"Created simulation: {sim.id}")
        
        # Run simulation with error handling
        try:
            run_simulation(sim.id)
            print("Simulation started successfully")
        except Exception as e:
            print(f"Failed to start simulation: {e}")
            return
        
        # Monitor with error handling
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                status = get_simulation_status(sim.id)
                print(f"Status: {status}")
                
                if status == "completed":
                    print("Simulation completed successfully!")
                    break
                elif status == "error":
                    print("Simulation encountered an error")
                    break
                
                time.sleep(10)
                
            except Exception as e:
                print(f"Error checking status: {e}")
                time.sleep(30)  # Wait longer on error
        
        # Get results with error handling
        try:
            results = get_simulation_results(sim.id)
            print(f"Results obtained: {len(results.files)} files")
        except Exception as e:
            print(f"Error getting results: {e}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    robust_simulation()
```

## Tips and Best Practices

1. **Start Small**: Begin with small systems (10-50 molecules) for testing
2. **Monitor Resources**: Keep an eye on memory and disk usage
3. **Use Appropriate Timesteps**: 0.001-0.002 ps for atomistic simulations
4. **Equilibrate Properly**: Always run equilibration before production
5. **Save Regularly**: Use the built-in backup features
6. **Handle Errors**: Always include error handling in production code
7. **Clean Up**: Delete old simulations to save space

For more complex examples and advanced usage, see the [User Guide](user-guide.md) and [API Reference](api-reference.md). 