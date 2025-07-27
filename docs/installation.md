# Installation Guide

This guide will help you install and set up MCP LAMMPS on your system.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large simulations)
- **Storage**: At least 1GB free space for installation and simulation files

### Required Software

#### 1. Python 3.9+

Check your Python version:
```bash
python3 --version
```

If Python 3.9+ is not installed, install it from [python.org](https://www.python.org/downloads/) or use your system's package manager:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip
```

**macOS (using Homebrew):**
```bash
brew install python@3.9
```

**Windows:**
Download and install from [python.org](https://www.python.org/downloads/)

#### 2. LAMMPS

MCP LAMMPS requires LAMMPS with Python interface support.

**Option A: Install from Source (Recommended)**
```bash
# Download LAMMPS
git clone https://github.com/lammps/lammps.git
cd lammps

# Configure with Python support
mkdir build
cd build
cmake ../cmake -DPKG_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python3)

# Build and install
make -j4
make install-python
```

**Option B: Using Package Managers**

**Ubuntu/Debian:**
```bash
sudo apt install lammps
```

**macOS (using Homebrew):**
```bash
brew install lammps
```

**Option C: Using Conda**
```bash
conda install -c conda-forge lammps
```

#### 3. Verify LAMMPS Installation

Test that LAMMPS Python interface works:
```python
import lammps
lmp = lammps.lammps()
print("LAMMPS version:", lmp.version())
```

## Installation Methods

### Method 1: Installation from Source (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mcp-lammps/mcp-lammps.git
   cd mcp-lammps
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install in development mode:**
   ```bash
   pip install -e .
   ```

### Method 2: Installation from PyPI

```bash
pip install mcp-lammps
```

### Method 3: Using Conda

```bash
conda install -c conda-forge mcp-lammps
```

## Configuration

### Environment Variables

Set up environment variables for optimal performance:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export MCP_LAMMPS_LOG_LEVEL=INFO
export MCP_LAMMPS_WORK_DIR=/path/to/your/workspace
export MCP_LAMMPS_MAX_SIMULATIONS=10
```

### Configuration File

Create a configuration file at `~/.mcp_lammps/config.yaml`:

```yaml
server:
  host: localhost
  port: 8000
  log_level: INFO

simulation:
  default_temperature: 300
  default_pressure: 1.0
  default_timestep: 0.001
  max_simulations: 10

storage:
  work_dir: /path/to/workspace
  backup_enabled: true
  max_file_size: 1GB
```

## Verification

### Test Installation

1. **Start the server:**
   ```bash
   python -m mcp_lammps.server
   ```

2. **Run a simple test:**
   ```bash
   python test_basic.py
   ```

3. **Check server status:**
   ```bash
   curl http://localhost:8000/health
   ```

### Example Test

Create a simple test script `test_installation.py`:

```python
import mcp_lammps
from mcp_lammps.server import LAMMPSServer

# Test server creation
server = LAMMPSServer()
print("✓ MCP LAMMPS server created successfully")

# Test LAMMPS interface
import lammps
lmp = lammps.lammps()
print(f"✓ LAMMPS version: {lmp.version()}")

print("Installation successful!")
```

## Troubleshooting

### Common Issues

#### 1. LAMMPS Python Interface Not Found

**Error:** `ModuleNotFoundError: No module named 'lammps'`

**Solution:**
```bash
# Reinstall LAMMPS with Python support
cd lammps
mkdir build
cd build
cmake ../cmake -DPKG_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python3)
make -j4
make install-python
```

#### 2. Permission Denied

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Check file permissions
ls -la /path/to/workspace

# Fix permissions
chmod 755 /path/to/workspace
```

#### 3. Memory Issues

**Error:** `MemoryError` or simulation crashes

**Solution:**
- Reduce simulation size
- Increase system memory
- Use swap space

#### 4. Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
export MCP_LAMMPS_PORT=8001
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/mcp-lammps/mcp-lammps/issues)
2. Search [GitHub Discussions](https://github.com/mcp-lammps/mcp-lammps/discussions)
3. Create a new issue with:
   - Operating system and version
   - Python version
   - LAMMPS version
   - Complete error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read the [User Guide](user-guide.md) to learn basic usage
2. Try the [Examples](examples.md) to see MCP LAMMPS in action
3. Explore the [API Reference](api-reference.md) for detailed documentation
4. Check the [Configuration](configuration.md) guide for advanced setup 