"""
MCP LAMMPS Server - Main server implementation for LAMMPS molecular dynamics simulations.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP
from mcp.server.models import InitializationOptions
from mcp.server.fastmcp.server import Context

from .lammps_interface import LAMMPSInterface
from .simulation_manager import SimulationManager
from .data_handler import DataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LAMMPSServer:
    """
    MCP Server for LAMMPS molecular dynamics simulations.
    
    This server provides tools for:
    - Setting up and configuring LAMMPS simulations
    - Running equilibration and production simulations
    - Monitoring simulation progress
    - Analyzing simulation results
    - Managing simulation workflows
    """
    
    def __init__(
        self,
        name: str = "LAMMPS MCP Server",
        instructions: str = "A server for controlling LAMMPS molecular dynamics simulations",
        work_dir: Optional[Path] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the LAMMPS MCP server.
        
        Args:
            name: Server name
            instructions: Server instructions for AI assistants
            work_dir: Working directory for simulations
            log_level: Logging level
        """
        self.name = name
        self.instructions = instructions
        self.work_dir = work_dir or Path.cwd() / "lammps_workspace"
        self.work_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        # Initialize components
        self.lammps_interface = LAMMPSInterface()
        self.simulation_manager = SimulationManager(self.work_dir)
        self.data_handler = DataHandler(self.work_dir)
        
        # Create MCP server
        self.server = FastMCP(
            name=name,
            instructions=instructions,
            tools=[]  # Will be populated by register_tools
        )
        
        # Register all tools
        self._register_tools()
        
        logger.info(f"LAMMPS MCP Server initialized with work directory: {self.work_dir}")
    
    def _register_tools(self) -> None:
        """Register all available tools with the MCP server."""
        from .tools.setup_tools import register_setup_tools
        from .tools.control_tools import register_control_tools
        from .tools.analysis_tools import register_analysis_tools
        from .tools.monitoring_tools import register_monitoring_tools
        
        # Register tools from each module
        register_setup_tools(self.server, self)
        register_control_tools(self.server, self)
        register_analysis_tools(self.server, self)
        register_monitoring_tools(self.server, self)
        
        logger.info("All tools registered successfully")
    
    async def health_check(self, ctx: Context) -> Dict[str, Any]:
        """
        Health check tool to verify server status.
        
        Args:
            ctx: MCP context
            
        Returns:
            Health status information
        """
        try:
            # Check LAMMPS availability
            lammps_status = await self.lammps_interface.check_availability()
            print(lammps_status)
            # Check simulation manager
            sim_manager_status = self.simulation_manager.get_status()
            print(sim_manager_status)
            # Check data handler
            data_handler_status = self.data_handler.get_status()
                        
            return {
                "status": "healthy",
                "lammps_interface": lammps_status,
                "simulation_manager": sim_manager_status,
                "data_handler": data_handler_status,
                "work_directory": str(self.work_dir),
                "active_simulations": len(self.simulation_manager.get_active_simulations())
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def run(self, transport: str = "stdio") -> None:
        """
        Run the MCP server.
        
        Args:
            transport: Transport protocol ("stdio", "sse", or "streamable-http")
        """
        logger.info(f"Starting LAMMPS MCP Server with {transport} transport")
        
        # Register health check tool
        @self.server.tool(
            name="health_check",
            title="Health Check",
            description="Check the health status of the LAMMPS MCP server"
        )
        async def health_check_tool(ctx: Context) -> Dict[str, Any]:
            return await self.health_check(ctx)
        
        # Run the server
        self.server.run(transport=transport)
    
    async def run_async(self, transport: str = "stdio") -> None:
        """
        Run the MCP server asynchronously.
        
        Args:
            transport: Transport protocol ("stdio", "sse", or "streamable-http")
        """
        logger.info(f"Starting LAMMPS MCP Server with {transport} transport (async)")
        
        # Register health check tool
        @self.server.tool(
            name="health_check",
            title="Health Check",
            description="Check the health status of the LAMMPS MCP server"
        )
        async def health_check_tool(ctx: Context) -> Dict[str, Any]:
            return await self.health_check(ctx)
        
        # Run the server asynchronously
        if transport == "stdio":
            await self.server.run_stdio_async()
        elif transport == "sse":
            await self.server.run_sse_async()
        elif transport == "streamable-http":
            await self.server.run_streamable_http_async()
        else:
            raise ValueError(f"Unsupported transport: {transport}")


def main():
    """Main entry point for the LAMMPS MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LAMMPS MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory for simulations"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Create and run server
    server = LAMMPSServer(
        work_dir=args.work_dir,
        log_level=args.log_level
    )
    
    try:
        server.run(transport=args.transport)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main() 