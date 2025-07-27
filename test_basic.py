#!/usr/bin/env python3
"""
Basic test script for the MCP LAMMPS server.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_lammps.server import LAMMPSServer


def test_basic_functionality():
    """Test basic server functionality."""
    print("Testing MCP LAMMPS Server...")
    
    # Create server
    server = LAMMPSServer()
    print(f"✅ Server created: {server.name}")
    print(f"✅ Work directory: {server.work_dir}")
    
    # Check components
    print(f"✅ LAMMPS Interface: {server.lammps_interface is not None}")
    print(f"✅ Simulation Manager: {server.simulation_manager is not None}")
    print(f"✅ Data Handler: {server.data_handler is not None}")
    
    # Check LAMMPS availability
    lammps_available = server.lammps_interface.is_available()
    print(f"✅ LAMMPS Available: {lammps_available}")
    
    # Test simulation manager
    status = server.simulation_manager.get_status()
    print(f"✅ Simulation Manager Status: {status}")
    
    # Test data handler
    data_status = server.data_handler.get_status()
    print(f"✅ Data Handler Status: {data_status}")
    
    print("\n🎉 Basic functionality test completed successfully!")
    print("\nNext steps:")
    print("1. Install LAMMPS to enable full functionality")
    print("2. Run the full example: python examples/basic_simulation.py")
    print("3. Test individual tools and features")


if __name__ == "__main__":
    test_basic_functionality() 