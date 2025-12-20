# Ethanol Liquid Simulation

This directory contains a complete setup for simulating liquid ethanol using the MCP LAMMPS framework with OpenFF force fields.

## Overview

The ethanol simulation uses:
- **Molecule**: Ethanol (C₂H₆O)
- **SMILES**: `CCO`
- **Force Field**: OpenFF Sage 2.2.0
- **Charge Method**: AM1-BCC
- **Density**: ~0.789 g/cm³ at 298 K (experimental)

## Files

- `run_ethanol_simulation.py` - Main script to create the simulation
- `ethanol_simulation_config.json` - Simulation configuration (generated)
- `liquid_box.data` - LAMMPS structure file (generated)
- `liquid_box.in` - LAMMPS input script (generated)
- `liquid_box.json` - System metadata (generated)

## Usage

### Basic Usage

Run the script with default parameters (100 molecules, 0.789 g/cm³ density):

```bash
python run_ethanol_simulation.py
```

### Custom Parameters

```bash
python run_ethanol_simulation.py \
    --num-molecules 200 \
    --density 0.789 \
    --temperature 298.0 \
    --pressure 1.0 \
    --equilibration-steps 10000 \
    --production-steps 100000
```

### Command Line Options

- `-n, --num-molecules`: Number of ethanol molecules (default: 100)
- `-d, --density`: Target density in g/cm³ (default: 0.789)
- `-t, --temperature`: Temperature in K (default: 298.0)
- `-p, --pressure`: Pressure in atm (default: 1.0)
- `--equilibration-steps`: Number of equilibration steps (default: 5000)
- `--production-steps`: Number of production steps (default: 50000)
- `-o, --output-dir`: Output directory (default: examples directory)

## Running the Simulation

After creating the simulation files, you can run the LAMMPS simulation:

```bash
# If LAMMPS script was generated
lmp -in liquid_box.in

# Or using the MCP server tools
# (Use the run_simulation tool with the simulation_id)
```

## Simulation Details

### System Setup

1. **Molecule Creation**: Ethanol is created from SMILES using OpenFF Toolkit
2. **Geometry Optimization**: 3D coordinates are generated and optimized
3. **Charge Assignment**: AM1-BCC partial charges are assigned
4. **Box Creation**: Liquid box is created with target density
5. **Force Field**: OpenFF Sage 2.2.0 parameters are assigned
6. **LAMMPS Export**: System is exported to LAMMPS format

### Force Field

The simulation uses the OpenFF Sage 2.2.0 force field, which provides:
- SMIRNOFF-based atom typing
- AM1-BCC charge assignment
- Complete bond, angle, dihedral, and non-bonded parameters

### Simulation Protocol

1. **Minimization**: Energy minimization to remove bad contacts
2. **Equilibration**: NVT equilibration at target temperature
3. **Production**: NPT production run at target temperature and pressure

## Expected Output

After running the script, you should see:

```
======================================================================
Creating Ethanol Liquid Simulation
======================================================================
Number of molecules: 100
Target density: 0.789 g/cm³
Temperature: 298.0 K
Pressure: 1.0 atm
Equilibration steps: 5000
Production steps: 50000

Initializing MCP LAMMPS components...
Creating liquid box with 100 ethanol molecules...
SMILES: CCO
✓ Liquid box created: liquid_box.data
✓ System metadata:
  - Total atoms: 900
  - Total molecules: 100
  - Box dimensions: [30.5, 30.5, 30.5]
✓ LAMMPS script generated: liquid_box.in
✓ Configuration saved: ethanol_simulation_config.json
✓ Simulation created with ID: <simulation_id>

======================================================================
SIMULATION SETUP COMPLETE
======================================================================
```

## Troubleshooting

### Missing Dependencies

If you get import errors, make sure you have installed:

```bash
pip install openff-toolkit openff-interchange
```

### OpenFF Issues

If OpenFF fails to load the force field, check that the force field file is available:

```python
from openff.toolkit import ForceField
ff = ForceField("openff-2.2.0.offxml")
```

### Density Issues

If the calculated box size seems incorrect, verify:
- The target density matches experimental values
- The molecular weight is correctly calculated
- The number of molecules is reasonable

## References

- OpenFF Toolkit: https://github.com/openforcefield/openff-toolkit
- OpenFF Interchange: https://github.com/openforcefield/openff-interchange
- LAMMPS Documentation: https://docs.lammps.org/


