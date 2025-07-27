"""
Tests for the MCP LAMMPS server.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from mcp_lammps.server import LAMMPSServer


class TestLAMMPSServer:
    """Test cases for the LAMMPS MCP server."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        server = LAMMPSServer()
        assert server.name == "LAMMPS MCP Server"
        assert server.work_dir.exists()
        assert server.lammps_interface is not None
        assert server.simulation_manager is not None
        assert server.data_handler is not None
    
    def test_server_with_custom_work_dir(self):
        """Test server initialization with custom work directory."""
        work_dir = Path("/home/zhenghaowu/mcp_lammps/test_lammps_workspace")
        server = LAMMPSServer(work_dir=work_dir)
        assert server.work_dir == work_dir
        assert work_dir.exists()
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        server = LAMMPSServer()
        
        # Mock context
        mock_ctx = Mock()
        
        # Test health check
        result = await server.health_check(mock_ctx)
        
        assert "status" in result
        assert "lammps_interface" in result
        assert "simulation_manager" in result
        assert "data_handler" in result
        assert "work_directory" in result
        assert "active_simulations" in result
    
    def test_server_tool_registration(self):
        """Test that tools are properly registered."""
        server = LAMMPSServer()
        
        # Check that the server has tools registered
        # This is a basic check - the actual tool registration happens in _register_tools
        assert hasattr(server, '_register_tools')
    
    def test_server_run_method(self):
        """Test server run method."""
        server = LAMMPSServer()
        
        # Test that run method exists
        assert hasattr(server, 'run')
        assert callable(server.run)
    
    @pytest.mark.asyncio
    async def test_server_run_async_method(self):
        """Test server async run method."""
        server = LAMMPSServer()
        
        # Test that run_async method exists
        assert hasattr(server, 'run_async')
        assert callable(server.run_async)
    
    def test_server_logging(self):
        """Test server logging configuration."""
        server = LAMMPSServer(log_level="DEBUG")
        
        # Check that logging is configured
        import logging
        assert logging.getLogger().level <= logging.DEBUG


class TestLAMMPSServerIntegration:
    """Integration tests for the LAMMPS MCP server."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete simulation workflow."""
        server = LAMMPSServer()
        
        # Mock context
        mock_ctx = Mock()
        
        # Test health check
        health_result = await server.health_check(mock_ctx)
        assert health_result["status"] in ["healthy", "unhealthy"]
        
        # Test simulation creation
        # This would require more complex mocking of the LAMMPS interface
        # For now, just test that the components exist
        assert server.simulation_manager is not None
        assert server.lammps_interface is not None
        assert server.data_handler is not None


if __name__ == "__main__":
    pytest.main([__file__]) 