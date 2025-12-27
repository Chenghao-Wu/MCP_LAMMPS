# Migration Guide: Removed Tools

This guide helps users migrate from deprecated tools to their modern replacements after the tool consolidation update.

## Overview

We've removed 14 redundant tools to streamline the API and improve maintainability. This guide shows how to use the replacement tools.

## Removed Tools and Replacements

### Analysis Tools

#### `calculate_viscosity` (REMOVED)
**Replacement:** Use `calculate_transport_properties`

```python
# Old way (REMOVED)
result = await calculate_viscosity(
    ctx, 
    simulation_id="sim_123",
    method="green_kubo",
    correlation_length=1000
)

# New way
result = await calculate_transport_properties(
    ctx,
    simulation_id="sim_123",
    property_types=["viscosity"],
    correlation_length=1000
)
# Access viscosity: result["properties"]["viscosity"]
```

#### `calculate_diffusion_coefficient` (REMOVED)
**Replacement:** Use `calculate_transport_properties`

```python
# Old way (REMOVED)
result = await calculate_diffusion_coefficient(
    ctx,
    simulation_id="sim_123",
    diffusion_type="self"
)

# New way
result = await calculate_transport_properties(
    ctx,
    simulation_id="sim_123",
    property_types=["diffusion"]
)
# Access diffusion: result["properties"]["diffusion"]
```

#### `calculate_hydrogen_bonds` (REMOVED)
**Replacement:** This was a placeholder that returned dummy data. For hydrogen bond analysis, use external tools like MDAnalysis directly or implement custom analysis.

---

### Control Tools

#### `pause_simulation` / `resume_simulation` (REMOVED)
**Reason:** These were non-functional in LAMMPS context (LAMMPS doesn't support true pause/resume).

**Replacement:** Use `stop_simulation` to stop, then restart with `run_simulation` if needed.

```python
# To stop a simulation
result = await stop_simulation(ctx, simulation_id="sim_123")

# To restart, create a new simulation from a checkpoint
# (implement checkpointing in your workflow)
```

#### `run_equilibration` / `run_production` (REMOVED)
**Replacement:** Use `run_simulation` with appropriate configuration

```python
# Old way (REMOVED)
await run_equilibration(ctx, simulation_id="sim_123", temperature=300.0, duration=1000)
await run_production(ctx, simulation_id="sim_123", temperature=300.0, pressure=1.0, duration=10000)

# New way
# Configure equilibration and production steps when creating simulation
result = await create_simulation(
    ctx,
    name="my_simulation",
    structure_file="structure.data",
    temperature=300.0,
    pressure=1.0,
    equilibration_steps=1000,
    production_steps=10000
)
# Then run everything at once
await run_simulation(ctx, simulation_id=result["simulation_id"])
```

---

### Monitoring Tools

#### `get_system_properties` (REMOVED)
**Replacement:** Use `get_simulation_status` with `include_system_properties=True`

```python
# Old way (REMOVED)
result = await get_system_properties(ctx, simulation_id="sim_123")

# New way
result = await get_simulation_status(
    ctx,
    simulation_id="sim_123",
    include_system_properties=True
)
# Access system properties: result["system_properties"]
```

#### `get_energy_data` (REMOVED)
**Replacement:** Use `get_simulation_status` with `include_energy_data=True`

```python
# Old way (REMOVED)
result = await get_energy_data(ctx, simulation_id="sim_123")

# New way
result = await get_simulation_status(
    ctx,
    simulation_id="sim_123",
    include_energy_data=True
)
# Access energy data: result["energy_data"]
```

#### `get_structure_data` (REMOVED)
**Replacement:** Use `get_simulation_status` with `include_system_properties=True` (includes structural info)

```python
# Old way (REMOVED)
result = await get_structure_data(ctx, simulation_id="sim_123")

# New way
result = await get_simulation_status(
    ctx,
    simulation_id="sim_123",
    include_system_properties=True
)
# System properties include structural information
```

---

### Property Tools

#### `compute_solvation_properties` (REMOVED)
**Reason:** This was a placeholder that returned dummy data.

**Replacement:** For solvation free energy calculations, use specialized tools like:
- GROMACS with free energy perturbation
- MDAnalysis for solvation shell analysis
- Custom scripts with thermodynamic integration

#### `analyze_phase_behavior` (REMOVED)
**Reason:** Niche functionality, rarely used.

**Replacement:** For phase behavior analysis:
- Use `calculate_thermodynamic_properties` for basic property analysis
- Use external tools like LAMMPS post-processing scripts
- Implement custom analysis using the exported trajectory and log data

---

### Setup Tools

#### `create_water_simulation` (REMOVED)
**Replacement:** Use `create_organic_simulation` with water SMILES

```python
# Old way (REMOVED)
result = await create_water_simulation(
    ctx,
    name="water_sim",
    num_molecules=100,
    temperature=300.0,
    pressure=1.0
)

# New way
result = await create_organic_simulation(
    ctx,
    name="water_sim",
    molecules=[
        {
            "smiles": "O",  # Water molecule
            "count": 100,
            "name": "water"
        }
    ],
    temperature=300.0,
    pressure=1.0
)
```

#### `create_simple_structure` (REMOVED)
**Reason:** Testing utility only.

**Replacement:** For testing purposes, create minimal structures using:
- `load_structure` with a simple LAMMPS data file
- `import_from_smiles` with simple molecules (e.g., "C" for methane)

---

## Summary of Changes

### Tools Removed: 14
- **Analysis:** 3 tools
- **Control:** 4 tools
- **Monitoring:** 3 tools
- **Property:** 2 tools
- **Setup:** 2 tools

### Tools Remaining: 34
All essential functionality is preserved through consolidated, more powerful tools.

## Benefits

1. **Simpler API:** Fewer tools to learn and maintain
2. **More Powerful:** Consolidated tools offer more options
3. **Better Performance:** Less code duplication
4. **Easier Maintenance:** Single implementation of each feature

## Need Help?

If you're having trouble migrating from a removed tool, please:
1. Check this guide for the replacement tool
2. Review the API reference for detailed parameters
3. Open an issue on GitHub if you need specific functionality that's missing

