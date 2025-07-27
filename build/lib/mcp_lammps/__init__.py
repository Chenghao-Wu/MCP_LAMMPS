"""
MCP LAMMPS Server - Model Context Protocol server for LAMMPS molecular dynamics simulations.
"""

__version__ = "0.1.0"
__author__ = "MCP LAMMPS Development Team"

from .server import LAMMPSServer

__all__ = ["LAMMPSServer"] 