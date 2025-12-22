# Quick Start Guide - Ethanol System Test

## TL;DR

Test the mcp_lammps capability to create LAMMPS files for an ethanol liquid system:

```bash
cd /home/zhenghaowu/mcp_lammps/examples
python test_ethanol_setup.py
```

Expected result: ✅ ALL TESTS PASSED

## Prerequisites

Install required packages:

```bash
pip install openff-toolkit openff-interchange openff-units rdkit
```

## What This Does

Creates LAMMPS input files for a 100-molecule ethanol liquid system using:
- **OpenFF Sage 2.2.1** force field (modern, not legacy GAFF)
- **NAGL AM1-BCC** charges (neural network-based)
- **Target density**: 0.789 g/cm³ (ethanol at 298 K)

## Output Files

Generated in `examples/output/`:
- `ethanol_liquid_test.data` - LAMMPS data file (~125 KB)
- `ethanol_liquid_test.in` - LAMMPS input script (~3 KB)
- `ethanol_liquid_test.json` - Metadata (~2 KB)

## Run the Simulation

After the test passes, run the simulation:

```bash
lmp -in output/ethanol_liquid_test.in
```

## Customize

Edit `test_ethanol_setup.py` to change:
- `smiles`: Different molecule (e.g., "O" for water, "c1ccccc1" for benzene)
- `molecule_count`: Number of molecules
- `target_density`: Density in g/cm³
- `temperature`: Temperature in K

## Troubleshooting

**Test fails?**
1. Check dependencies: `pip list | grep openff`
2. Verify you're in the examples directory
3. Check the error message in the output

**Need help?**
- See `README.md` for detailed documentation
- See `EXAMPLE_OUTPUT.md` for expected output
- See `TEST_SUMMARY.md` for technical details

## What's Being Tested

✅ SMILES → 3D structure conversion  
✅ Automatic atom typing (OpenFF)  
✅ NAGL charge assignment  
✅ Liquid box creation  
✅ LAMMPS file generation  
✅ Charge neutrality validation  

## Success Looks Like

```
================================================================================
  ✓ ALL TESTS PASSED
  Ethanol system setup is working correctly with OpenFF!
================================================================================
```

## Next Steps

1. ✅ Test passes → Files are ready for simulation
2. 📊 Visualize with OVITO or VMD
3. 🚀 Run simulation with LAMMPS
4. 🔬 Analyze results
5. 🎯 Create your own systems by modifying the script

---

**Note**: This tests the **modern OpenFF-based system** (December 2025), not the legacy GAFF implementation.

