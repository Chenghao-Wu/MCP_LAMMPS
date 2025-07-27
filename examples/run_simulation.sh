#!/bin/bash
# Script to run the water simulation

echo "Running water simulation with 10 TIP3P water molecules..."
echo "Temperature: 300K, Ensemble: NVT"
echo "=================================="

# Check if LAMMPS is available
if command -v lmp &> /dev/null; then
    echo "LAMMPS found, starting simulation..."
    lmp -in water_simulation.in
elif command -v lammps &> /dev/null; then
    echo "LAMMPS found (lammps command), starting simulation..."
    lammps -in water_simulation.in
else
    echo "ERROR: LAMMPS not found in PATH"
    echo "Please install LAMMPS or add it to your PATH"
    exit 1
fi

echo "Simulation completed!"
echo "Check the following output files:"
echo "- water_10.lammpstrj (trajectory)"
echo "- log.lammps (thermodynamic data)"
