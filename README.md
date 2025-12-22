# MCP LAMMPS Server

A Model Context Protocol (MCP) server that enables AI assistants to interact with LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) for molecular dynamics simulations.

## Status

This is still in experimental status. This package is developed in collaboration with AI coder.

### Update 06.12.2025
Note: all tests are performed with Sonnet 4.5

Add support for organic liquids simulations with gaff

simple test prompt:

Organic Liquid Simulation
```
Create a complete organic liquid simulation with these parameters:
- Name: "ethanol_liquid_md"
- Molecules: [{"smiles": "CCO", "count": 500, "name": "ethanol"}]
- Target density: 0.789 g/cm³
- Force field: GAFF
- Temperature: 298.15 K
- Pressure: 1.0 atm
- Timestep: 0.5 fs
- Equilibration: 10000 steps
- Production: 100000 steps
```

Multi-Component Simulation
```
Set up a ternary mixture simulation called "organic_solvent_mix":
- Ethanol (CCO): 30% mole fraction
- Acetone (CC(=O)C): 40% mole fraction  
- Toluene (Cc1ccccc1): 30% mole fraction
- Temperature: 300 K
- Pressure: 1 atm
- Equilibration: 20000 steps
- Production: 200000 steps
```


## Overview

This MCP is part of our workflow for the autonomous computational materials design with LLM. This MCP server provides a standardized interface for controlling LAMMPS molecular dynamics simulations through natural language commands. It enables AI assistants to:

- Set up and configure molecular dynamics simulations
- Run equilibration and production simulations
- Monitor simulation progress in real-time
- Analyze simulation results
- Manage simulation workflows

## Features

### Core Capabilities
- **Simulation Management**: Create, configure, and run LAMMPS simulations
- **Molecular Structure Processing**: Convert SMILES to 3D structures, assign GAFF atom types
- **Force Field Support**: GAFF (General AMBER Force Field) parameter assignment
- **LAMMPS Data File Generation**: Create simulation input files from molecular structures
- **Real-time Monitoring**: Track simulation progress and system properties
- **Analysis Tools**: Process trajectories and calculate thermodynamic properties (with MDAnalysis)
- **Charge Calculation**: Gasteiger, MMFF94, and GAFF charge estimation methods
- **Workflow Automation**: Define and execute multi-step simulation workflows

### Supported Molecular Systems
- Small organic molecules
- Liquid systems and mixtures
- Polymer systems
- Custom molecular structures via SMILES or coordinate files

## Installation

### Prerequisites
- Python 3.9 or higher
- LAMMPS with Python interface
- RDKit (for molecular structure processing)
- **Packmol** (for organic liquid box creation with proper molecular packing)

### System Dependencies

**Packmol Installation** (Required for organic liquid simulations):
```bash
# Ubuntu/Debian
sudo apt install packmol

# macOS (Homebrew)
brew install packmol

# From source
# Download from http://m3g.iqm.unicamp.br/packmol/
```

### Optional Dependencies
- MDAnalysis (for advanced trajectory analysis)
- OpenBabel (for additional molecular format conversion - requires system Open Babel libraries)
- pytraj (for additional trajectory analysis - requires cpptraj)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mcp-lammps/mcp-lammps.git
   cd mcp-lammps
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install in development mode**:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

Start the MCP server:

```bash
python -m mcp_lammps.server
```

### Configuration

The server can be configured through environment variables or configuration files:

```bash
export MCP_LAMMPS_LOG_LEVEL=INFO
export MCP_LAMMPS_WORK_DIR=/path/to/workspace
python -m mcp_lammps.server
```

### Example Prompts

**Basic Organic Molecule Simulation:**
```
Create a complete ethanol liquid simulation:
- SMILES: CCO
- 200 molecules
- Temperature: 298 K
- Pressure: 1 atm
- GAFF force field
- 10,000 equilibration steps, 50,000 production steps
```

**Multi-Component Mixture:**
```
Set up a binary mixture simulation:
- Component 1: Ethanol (CCO, 100 molecules)
- Component 2: Acetone (CC(=O)C, 100 molecules)
- Temperature: 300 K
- NPT ensemble
- GAFF force field
```

**Custom Molecule from SMILES:**
```
Create a simulation for ibuprofen:
- SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
- 50 molecules
- Calculate partial charges using Gasteiger method
- NVT ensemble at 310 K
```

## Development

### Project Structure
```
mcp_lammps/
├── src/mcp_lammps/
│   ├── server.py              # Main MCP server
│   ├── lammps_interface.py    # LAMMPS wrapper and control
│   ├── simulation_manager.py  # Simulation lifecycle management
│   ├── data_handler.py        # Data processing and file I/O
│   ├── tools/                 # MCP tool implementations
│   │   ├── setup_tools.py     # Simulation setup tools
│   │   ├── control_tools.py   # Simulation control tools
│   │   ├── analysis_tools.py  # Analysis and property calculation
│   │   ├── monitoring_tools.py # Real-time monitoring
│   │   └── organic_tools.py   # Organic molecule handling
│   ├── utils/                 # Utility modules
│   │   ├── molecular_utils.py # Molecular structure processing
│   │   ├── forcefield_utils.py # GAFF force field handling
│   │   ├── converters.py      # Format conversion utilities
│   │   └── helpers.py         # General helper functions
│   ├── parsers/               # Parameter file parsers
│   │   └── gaff_parser.py     # GAFF force field parser
│   └── data/                  # Force field data files
│       └── gaff_parameters.json
├── tests/                     # Comprehensive test suite
├── examples/                  # Usage examples and sample data
└── docs/                      # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LAMMPS development team for the molecular dynamics engine
- Model Context Protocol community for the MCP framework
- Scientific computing community for inspiration and feedback
- LLM for writing the code