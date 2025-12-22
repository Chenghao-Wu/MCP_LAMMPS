# MCP LAMMPS Examples

This directory contains example scripts demonstrating the capabilities of the MCP LAMMPS package.

## Test Scripts

### test_ethanol_setup.py

A comprehensive test script that demonstrates the capability of `mcp_lammps` to set up an organic liquid system (ethanol) and generate LAMMPS data files ready for simulation.

**Features Tested:**
- SMILES to 3D structure conversion using OpenFF Toolkit
- Automatic atom typing via OpenFF Sage 2.2.1 force field
- NAGL AM1-BCC charge assignment (neural network-based)
- Multi-component liquid box creation using OpenFF Interchange
- LAMMPS data and input file generation
- System validation and information display

**Note:** This script tests the modern OpenFF-based system (migrated December 2025), not the legacy GAFF implementation.

#### Requirements

Before running the test script, ensure you have the required dependencies installed:

```bash
pip install openff-toolkit openff-interchange openff-units rdkit
```

#### Usage

Run the test script from the examples directory:

```bash
cd examples
python test_ethanol_setup.py
```

Or run it directly if it's executable:

```bash
./test_ethanol_setup.py
```

#### What It Does

The script performs the following tests:

1. **Dependency Check**: Verifies that OpenFF Toolkit and Interchange are available
2. **Single Molecule Creation**: Creates a single ethanol molecule from SMILES ("CCO")
   - Generates 3D coordinates
   - Assigns NAGL AM1-BCC charges
   - Validates charge neutrality
3. **Liquid Box Creation**: Creates a liquid box with 100 ethanol molecules
   - Target density: 0.789 g/cm³ (ethanol at 298 K)
   - Generates LAMMPS data file
   - Generates LAMMPS input script
   - Creates metadata file
4. **Data File Validation**: Validates the generated LAMMPS data file
   - Checks for required sections (Atoms, Bonds, Angles, Dihedrals)
   - Extracts system statistics
   - Validates charge neutrality
5. **Information Display**: Shows comprehensive system information
   - Force field details (OpenFF Sage 2.2.1)
   - Charge method (NAGL AM1-BCC)
   - System composition
   - Box dimensions
   - File locations and sizes

#### Output

The script creates an `output/` directory containing:

- `ethanol_liquid_test.data` - LAMMPS data file with force field parameters
- `ethanol_liquid_test.in` - LAMMPS input script ready for simulation
- `ethanol_liquid_test.json` - Metadata file with system information

#### Expected Results

When successful, you should see output like:

```
================================================================================
  ✓ ALL TESTS PASSED
  Ethanol system setup is working correctly with OpenFF!
================================================================================
```

The test validates:
- ✓ OpenFF dependencies are available
- ✓ Single molecule creation works
- ✓ Liquid box creation succeeds
- ✓ Generated files contain all required sections
- ✓ System is electrically neutral
- ✓ Force field parameters are properly assigned

#### Test Parameters

The default test parameters are:

```python
{
    "name": "ethanol_liquid_test",
    "smiles": "CCO",                    # Ethanol
    "molecule_count": 100,              # Number of molecules
    "target_density": 0.789,            # g/cm³ (ethanol at 298 K)
    "temperature": 298.15,              # K
    "pressure": 1.0,                    # atm
    "force_field": "openff",            # OpenFF Sage 2.2.1
    "charge_method": "nagl"             # NAGL AM1-BCC
}
```

You can modify these parameters in the script to test different systems.

#### Troubleshooting

**ImportError: OpenFF not available**
```bash
pip install openff-toolkit openff-interchange openff-units
```

**ImportError: RDKit not available**
```bash
pip install rdkit
# or with conda:
conda install -c conda-forge rdkit
```

**NAGL charge assignment fails**
```bash
pip install openff-nagl
```

#### Next Steps

After verifying the test passes, you can:

1. Examine the generated LAMMPS files in `examples/output/`
2. Run the simulation using LAMMPS:
   ```bash
   lmp -in output/ethanol_liquid_test.in
   ```
3. Modify the test parameters to create different systems
4. Use the script as a template for your own organic liquid simulations

## Force Field Information

### OpenFF Sage 2.2.1

This package uses the OpenFF Sage 2.2.1 force field, which is:
- Modern and actively maintained
- Optimized for organic molecules
- Provides superior accuracy compared to legacy GAFF
- Automatically assigns all force field parameters

### NAGL Charges

NAGL (Neural Network Assisted Graph Learning) provides:
- AM1-BCC quality charges from a neural network
- Much faster than quantum mechanical calculations
- More accurate than classical methods (e.g., Gasteiger)
- Automatic charge neutralization

## Additional Resources

- [MCP LAMMPS Documentation](../docs/)
- [OpenFF Toolkit Documentation](https://open-forcefield-toolkit.readthedocs.io/)
- [LAMMPS Documentation](https://docs.lammps.org/)

