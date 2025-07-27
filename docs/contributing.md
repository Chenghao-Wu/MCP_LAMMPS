# Contributing to MCP LAMMPS

Thank you for your interest in contributing to MCP LAMMPS! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, make sure you have:

- Python 3.9 or higher
- Git
- LAMMPS with Python interface
- Basic knowledge of molecular dynamics simulations
- Familiarity with the Model Context Protocol (MCP)

### Areas for Contribution

We welcome contributions in the following areas:

- **Core Functionality**: New simulation types, analysis methods, force fields
- **Performance**: Optimization, parallel processing, GPU support
- **User Interface**: Better error handling, user experience improvements
- **Documentation**: Tutorials, examples, API documentation
- **Testing**: Unit tests, integration tests, performance benchmarks
- **Examples**: New simulation examples, use cases
- **Bug Fixes**: Bug reports and fixes

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/your-username/mcp-lammps.git
cd mcp-lammps

# Add the upstream repository
git remote add upstream https://github.com/mcp-lammps/mcp-lammps.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Install Development Tools

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or install individually
pip install pytest pytest-asyncio black flake8 mypy
```

### 4. Verify Setup

```bash
# Run tests to verify setup
pytest tests/

# Check code style
black --check src/
flake8 src/
mypy src/
```

## Code Style and Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **String quotes**: Double quotes for docstrings, single quotes for strings
- **Import order**: Standard library, third-party, local imports

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format code
black src/

# Check formatting
black --check src/
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Run linter
flake8 src/

# Configuration in pyproject.toml
[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
```

### Type Checking

We use [mypy](https://mypy.readthedocs.io/) for type checking:

```bash
# Run type checker
mypy src/

# Configuration in pyproject.toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
disallow_untyped_defs = true
```

### Example Code Style

```python
"""Example of proper code style for MCP LAMMPS."""

from typing import Dict, List, Optional, Tuple
import asyncio
import logging

from mcp_lammps.data_handler import DataHandler
from mcp_lammps.simulation_manager import SimulationManager


class ExampleClass:
    """Example class demonstrating proper code style.
    
    This class shows the proper way to write code for MCP LAMMPS,
    including docstrings, type hints, and error handling.
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None) -> None:
        """Initialize the example class.
        
        Args:
            name: The name of the instance
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    async def process_data(self, data: List[float]) -> Tuple[float, float]:
        """Process a list of data points.
        
        Args:
            data: List of numerical data points
            
        Returns:
            Tuple of (mean, standard deviation)
            
        Raises:
            ValueError: If data list is empty
        """
        if not data:
            raise ValueError("Data list cannot be empty")
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5
        
        self.logger.info(f"Processed {len(data)} data points")
        return mean, std_dev
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mcp_lammps

# Run specific test file
pytest tests/test_simulation_manager.py

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Writing Tests

Follow these guidelines for writing tests:

1. **Test file naming**: `test_<module_name>.py`
2. **Test function naming**: `test_<function_name>_<scenario>`
3. **Use descriptive test names**: Clear what is being tested
4. **Test both success and failure cases**
5. **Use fixtures for common setup**
6. **Mock external dependencies**

### Example Test

```python
"""Tests for the simulation manager module."""

import pytest
from unittest.mock import Mock, patch

from mcp_lammps.simulation_manager import SimulationManager
from mcp_lammps.exceptions import SimulationError


class TestSimulationManager:
    """Test cases for SimulationManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a simulation manager instance for testing."""
        return SimulationManager()
    
    @pytest.fixture
    def mock_lammps(self):
        """Mock LAMMPS instance."""
        mock = Mock()
        mock.version.return_value = "LAMMPS (1 Jan 2023)"
        return mock
    
    def test_create_simulation_success(self, manager):
        """Test successful simulation creation."""
        simulation = manager.create_simulation(
            name="test_sim",
            temperature=300,
            pressure=1.0
        )
        
        assert simulation.name == "test_sim"
        assert simulation.temperature == 300
        assert simulation.pressure == 1.0
        assert simulation.id is not None
    
    def test_create_simulation_invalid_temperature(self, manager):
        """Test simulation creation with invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            manager.create_simulation(
                name="test_sim",
                temperature=-100
            )
    
    @patch('mcp_lammps.simulation_manager.lammps')
    def test_run_simulation_success(self, mock_lammps_class, manager, mock_lammps):
        """Test successful simulation execution."""
        mock_lammps_class.return_value = mock_lammps
        
        simulation = manager.create_simulation(name="test_sim")
        result = manager.run_simulation(simulation.id)
        
        assert result is True
        mock_lammps.command.assert_called()
    
    def test_run_simulation_not_found(self, manager):
        """Test running non-existent simulation."""
        with pytest.raises(SimulationError, match="Simulation not found"):
            manager.run_simulation("non-existent-id")
```

### Test Coverage

We aim for high test coverage:

```bash
# Generate coverage report
pytest --cov=mcp_lammps --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation

### Docstring Standards

Use Google-style docstrings:

```python
def calculate_properties(
    simulation_id: str,
    log_file: Optional[str] = None
) -> ThermodynamicProperties:
    """Calculate thermodynamic properties from simulation data.
    
    This function analyzes the log file from a LAMMPS simulation
    and calculates average thermodynamic properties.
    
    Args:
        simulation_id: The ID of the simulation to analyze
        log_file: Optional path to the log file. If not provided,
                 the function will look for the default log file.
    
    Returns:
        ThermodynamicProperties object containing calculated averages
        and standard deviations.
    
    Raises:
        SimulationError: If the simulation is not found
        FileError: If the log file cannot be read
        ValueError: If the log file format is invalid
    
    Example:
        >>> properties = calculate_properties("sim_123")
        >>> print(f"Average temperature: {properties.avg_temperature} K")
        Average temperature: 300.5 K
    """
    pass
```

### Documentation Updates

When adding new features, update:

1. **API Reference** (`docs/api-reference.md`)
2. **User Guide** (`docs/user-guide.md`) if applicable
3. **Examples** (`docs/examples.md`) with new examples
4. **Configuration Guide** (`docs/configuration.md`) if new options

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### 1. Create a Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### 2. Make Your Changes

- Write your code following the style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "Add new water simulation template

- Add TIP4P water model support
- Include temperature-dependent parameters
- Add comprehensive tests
- Update documentation with examples

Fixes #123"
```

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### 5. Pull Request Guidelines

Your pull request should include:

- **Clear description** of the changes
- **Reference to issues** being addressed
- **Tests** for new functionality
- **Documentation updates** if needed
- **Screenshots** for UI changes

### 6. Review Process

1. **Automated checks** must pass (tests, linting, type checking)
2. **Code review** by maintainers
3. **Address feedback** and make requested changes
4. **Merge** when approved

## Issue Reporting

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for solutions
3. **Try the latest version** of the code
4. **Reproduce the issue** in a minimal example

### Issue Template

Use the provided issue templates:

```markdown
## Bug Report

**Description**
Clear description of the bug.

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- LAMMPS: [e.g., 2023.0.0]
- MCP LAMMPS: [e.g., 0.1.0]

**Additional Information**
Any other context, logs, or screenshots.
```

### Feature Request Template

```markdown
## Feature Request

**Description**
Clear description of the feature you'd like to see.

**Use Case**
How would this feature be used?

**Proposed Solution**
Any ideas for implementation?

**Alternatives Considered**
Other approaches you've considered.

**Additional Information**
Any other context or examples.
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Be patient** with newcomers
- **Be helpful** and supportive

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions
- **Email**: For security issues (see SECURITY.md)

### Recognition

Contributors will be recognized in:

- **Contributors list** on GitHub
- **Release notes** for significant contributions
- **Documentation** for major features
- **Acknowledgments** in papers and presentations

## Development Workflow

### Daily Development

```bash
# Start development session
git checkout main
git pull upstream main
git checkout -b feature/new-feature

# Make changes and test
# ... edit code ...
pytest tests/
black src/
flake8 src/
mypy src/

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature
```

### Release Process

1. **Create release branch** from main
2. **Update version** in `pyproject.toml`
3. **Update changelog** with new features/fixes
4. **Run full test suite** and integration tests
5. **Create release** on GitHub
6. **Publish to PyPI** (maintainers only)

## Getting Help

If you need help with contributing:

1. **Check documentation** first
2. **Search existing issues** and discussions
3. **Ask in GitHub Discussions**
4. **Contact maintainers** for specific questions

## License

By contributing to MCP LAMMPS, you agree that your contributions will be licensed under the same license as the project (MIT License).

Thank you for contributing to MCP LAMMPS! 