# Water Simulation with 10 TIP3P Water Molecules

This directory contains a complete LAMMPS simulation setup for 10 TIP3P water molecules at 300K under NVT ensemble.

## Files Description

- **`water_simulation.in`**: LAMMPS input script for the water simulation
- **`water_10.data`**: Molecular structure file with 10 TIP3P water molecules
- **`run_water_simulation.py`**: Python script to set up and run the simulation
- **`run_simulation.sh`**: Executable shell script to run the simulation
- **`README_water_simulation.md`**: This documentation file

## Simulation Details

- **System**: 10 TIP3P water molecules
- **Temperature**: 300K
- **Ensemble**: NVT (constant Number, Volume, Temperature)
- **Box size**: 15×15×15 Å
- **Force field**: TIP3P water model
- **Total atoms**: 30 (10 O + 20 H)
- **Total bonds**: 20 (O-H bonds)
- **Total angles**: 10 (H-O-H angles)

## TIP3P Water Model Parameters

- **Oxygen charge**: -0.830 e
- **Hydrogen charge**: +0.415 e
- **O-H bond length**: 0.9572 Å
- **H-O-H angle**: 104.52°
- **O-O LJ parameters**: ε = 0.1521 kcal/mol, σ = 3.188 Å
- **H-H LJ parameters**: 0.0 (no van der Waals interactions)

## Running the Simulation

### Option 1: Using the shell script (recommended)
```bash
./run_simulation.sh
```

### Option 2: Direct LAMMPS execution
```bash
lmp -in water_simulation.in
```

### Option 3: Using Python script
```bash
python3 run_water_simulation.py
```

## Simulation Protocol

1. **Minimization**: Energy minimization to remove bad contacts
2. **Equilibration**: 1000 steps at 300K under NVT ensemble
3. **Production**: 5000 steps at 300K under NVT ensemble

## Output Files

After running the simulation, you will get:
- **`water_10.lammpstrj`**: Trajectory file with atomic positions and velocities
- **`log.lammps`**: Thermodynamic data (temperature, energy, pressure, etc.)

## Analysis

The simulation provides:
- Temperature evolution
- Potential and kinetic energy
- Total energy
- Pressure
- Volume
- Density

## Requirements

- LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator)
- Python 3 (for the setup script)
- Bash shell (for the run script)

## Notes

- The simulation uses periodic boundary conditions
- Long-range electrostatic interactions are handled with PPPM method
- The system is well-equilibrated for studying water properties
- The box size provides adequate spacing between water molecules

## Troubleshooting

If you encounter issues:
1. Make sure LAMMPS is installed and accessible in your PATH
2. Check that all input files are present
3. Verify that the data file format is correct
4. Ensure sufficient disk space for output files 