# MCP LAMMPS Documentation

Welcome to the documentation for the MCP LAMMPS package - a Model Context Protocol (MCP) server that enables AI assistants to interact with LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) for molecular dynamics simulations.

## Quick Start

- **[Installation Guide](installation.md)** - How to install and set up MCP LAMMPS
- **[User Guide](user-guide.md)** - Basic usage and common workflows
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples.md)** - Working examples and tutorials
- **[Configuration](configuration.md)** - Server configuration options

## What is MCP LAMMPS?

MCP LAMMPS is a Model Context Protocol server that provides a standardized interface for controlling LAMMPS molecular dynamics simulations through natural language commands. It enables AI assistants to:

- Set up and configure molecular dynamics simulations
- Run equilibration and production simulations
- Monitor simulation progress in real-time
- Analyze simulation results
- Manage simulation workflows

## Key Features

### 🚀 **Simulation Management**
- Create, configure, and run LAMMPS simulations
- Support for multiple force fields and ensembles
- Automated workflow management

### 📊 **Real-time Monitoring**
- Track simulation progress and system properties
- Monitor energy components and structural properties
- Real-time data visualization capabilities

### 🔬 **Analysis Tools**
- Process trajectories and calculate thermodynamic properties
- Generate plots and export data in various formats
- Comprehensive analysis workflows

### 🛠️ **Structure Handling**
- Load molecular structures from various formats
- Create simple test structures
- Support for complex molecular systems

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Assistant  │◄──►│  MCP LAMMPS     │◄──►│     LAMMPS      │
│                 │    │     Server      │    │   Simulation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Data Handler   │
                       │  & Analysis     │
                       └─────────────────┘
```

## Getting Help

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/mcp-lammps/mcp-lammps/issues)
- **Discussions**: Join the community on [GitHub Discussions](https://github.com/mcp-lammps/mcp-lammps/discussions)
- **Documentation**: This documentation is hosted on [Read the Docs](https://mcp-lammps.readthedocs.io/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on how to get started.

## License

This project is licensed under the Apache License - see the [LICENSE](../LICENSE) file for details. 