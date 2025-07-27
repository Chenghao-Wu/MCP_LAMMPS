"""
Analysis Tools - Tools for analyzing LAMMPS simulation results.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mcp.server.fastmcp.server import Context

logger = logging.getLogger(__name__)


def register_analysis_tools(server: Any, lammps_server: Any) -> None:
    """
    Register analysis tools with the MCP server.
    
    Args:
        server: MCP server instance
        lammps_server: LAMMPS server instance
    """
    
    @server.tool(
        name="analyze_trajectory",
        title="Analyze Trajectory",
        description="Analyze trajectory data from a simulation"
    )
    async def analyze_trajectory(
        ctx: Context,
        simulation_id: str,
        trajectory_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze trajectory data from a simulation.
        
        Args:
            simulation_id: Simulation ID
            trajectory_file: Path to trajectory file (optional)
            
        Returns:
            Trajectory analysis results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Find trajectory file
            if trajectory_file:
                traj_path = Path(trajectory_file)
            else:
                # Look for trajectory file in simulation directory
                traj_files = list(sim.work_dir.glob("*.dump"))
                if traj_files:
                    traj_path = traj_files[0]
                else:
                    raise ValueError("No trajectory file found")
            
            if not traj_path.exists():
                raise ValueError(f"Trajectory file not found: {traj_path}")
            
            # Read trajectory data
            trajectory_data = lammps_server.data_handler.read_trajectory_data(traj_path)
            
            if not trajectory_data:
                raise ValueError("Failed to read trajectory data")
            
            # Perform analysis
            analysis_results = {}
            
            # Basic statistics
            if trajectory_data.get('positions'):
                positions = trajectory_data['positions']
                analysis_results['num_frames'] = len(positions)
                analysis_results['num_atoms'] = positions[0].shape[0] if positions else 0
                
                # Calculate RMSD if multiple frames
                if len(positions) > 1:
                    rmsd_values = []
                    ref_pos = positions[0]
                    for pos in positions[1:]:
                        rmsd = np.sqrt(np.mean(np.sum((pos - ref_pos) ** 2, axis=1)))
                        rmsd_values.append(rmsd)
                    
                    analysis_results['rmsd'] = {
                        'mean': float(np.mean(rmsd_values)),
                        'std': float(np.std(rmsd_values)),
                        'min': float(np.min(rmsd_values)),
                        'max': float(np.max(rmsd_values)),
                        'values': [float(x) for x in rmsd_values]
                    }
            
            # Box analysis
            if trajectory_data.get('box'):
                box_data = trajectory_data['box']
                volumes = []
                for box in box_data:
                    # Calculate volume from box dimensions
                    lx = box[0][1] - box[0][0]
                    ly = box[1][1] - box[1][0]
                    lz = box[2][1] - box[2][0]
                    volume = lx * ly * lz
                    volumes.append(volume)
                
                analysis_results['box_analysis'] = {
                    'mean_volume': float(np.mean(volumes)),
                    'volume_std': float(np.std(volumes)),
                    'volume_change': float((volumes[-1] - volumes[0]) / volumes[0] * 100) if len(volumes) > 1 else 0.0
                }
            
            # Save analysis results
            sim.add_result("trajectory_analysis", analysis_results)
            
            ctx.info(f"Analyzed trajectory for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "trajectory_file": str(traj_path),
                "analysis_results": analysis_results
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trajectory: {e}")
            raise
    
    @server.tool(
        name="calculate_properties",
        title="Calculate Properties",
        description="Calculate thermodynamic properties from simulation data"
    )
    async def calculate_properties(
        ctx: Context,
        simulation_id: str,
        log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate thermodynamic properties from simulation data.
        
        Args:
            simulation_id: Simulation ID
            log_file: Path to log file (optional)
            
        Returns:
            Calculated properties
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Find log file
            if log_file:
                log_path = Path(log_file)
            else:
                # Look for log file in simulation directory
                log_files = list(sim.work_dir.glob("*.log"))
                if log_files:
                    log_path = log_files[0]
                else:
                    raise ValueError("No log file found")
            
            if not log_path.exists():
                raise ValueError(f"Log file not found: {log_path}")
            
            # Read thermodynamic data
            thermo_df = lammps_server.data_handler.read_thermo_data(log_path)
            
            if thermo_df.empty:
                raise ValueError("No thermodynamic data found in log file")
            
            # Calculate properties
            properties = {}
            
            # Basic statistics for each column
            for column in thermo_df.columns:
                if column != 'Step' and thermo_df[column].dtype in ['float64', 'int64']:
                    properties[column] = {
                        'mean': float(thermo_df[column].mean()),
                        'std': float(thermo_df[column].std()),
                        'min': float(thermo_df[column].min()),
                        'max': float(thermo_df[column].max()),
                        'median': float(thermo_df[column].median())
                    }
            
            # Specific property calculations
            if 'Temp' in thermo_df.columns:
                properties['temperature_analysis'] = {
                    'target_temp': float(thermo_df['Temp'].iloc[-1]),  # Assume last value is target
                    'temp_fluctuation': float(thermo_df['Temp'].std()),
                    'temp_convergence': float(thermo_df['Temp'].iloc[-100:].std()) if len(thermo_df) > 100 else float(thermo_df['Temp'].std())
                }
            
            if 'Press' in thermo_df.columns:
                properties['pressure_analysis'] = {
                    'target_pressure': float(thermo_df['Press'].iloc[-1]),
                    'pressure_fluctuation': float(thermo_df['Press'].std()),
                    'pressure_convergence': float(thermo_df['Press'].iloc[-100:].std()) if len(thermo_df) > 100 else float(thermo_df['Press'].std())
                }
            
            if 'PotEng' in thermo_df.columns:
                properties['energy_analysis'] = {
                    'final_potential_energy': float(thermo_df['PotEng'].iloc[-1]),
                    'energy_convergence': float(thermo_df['PotEng'].iloc[-100:].std()) if len(thermo_df) > 100 else float(thermo_df['PotEng'].std()),
                    'energy_trend': 'decreasing' if thermo_df['PotEng'].iloc[-1] < thermo_df['PotEng'].iloc[0] else 'increasing'
                }
            
            # Save properties
            sim.add_result("calculated_properties", properties)
            
            ctx.info(f"Calculated properties for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "log_file": str(log_path),
                "properties": properties
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate properties: {e}")
            raise
    
    @server.tool(
        name="plot_results",
        title="Plot Results",
        description="Generate plots from simulation results"
    )
    async def plot_results(
        ctx: Context,
        simulation_id: str,
        plot_type: str = "thermo",
        log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate plots from simulation results.
        
        Args:
            simulation_id: Simulation ID
            plot_type: Type of plot to generate (thermo, trajectory, etc.)
            log_file: Path to log file (optional)
            
        Returns:
            Plot information
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Create plots directory
            plots_dir = sim.work_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_files = []
            
            if plot_type == "thermo":
                # Find log file
                if log_file:
                    log_path = Path(log_file)
                else:
                    log_files = list(sim.work_dir.glob("*.log"))
                    if log_files:
                        log_path = log_files[0]
                    else:
                        raise ValueError("No log file found")
                
                if not log_path.exists():
                    raise ValueError(f"Log file not found: {log_path}")
                
                # Read thermodynamic data
                thermo_df = lammps_server.data_handler.read_thermo_data(log_path)
                
                if thermo_df.empty:
                    raise ValueError("No thermodynamic data found")
                
                # Create temperature plot
                if 'Temp' in thermo_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(thermo_df['Step'], thermo_df['Temp'])
                    plt.xlabel('Step')
                    plt.ylabel('Temperature (K)')
                    plt.title(f'Temperature vs Time - {sim.name}')
                    plt.grid(True)
                    
                    temp_plot_file = plots_dir / f"{simulation_id}_temperature.png"
                    plt.savefig(temp_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(str(temp_plot_file))
                
                # Create pressure plot
                if 'Press' in thermo_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(thermo_df['Step'], thermo_df['Press'])
                    plt.xlabel('Step')
                    plt.ylabel('Pressure (atm)')
                    plt.title(f'Pressure vs Time - {sim.name}')
                    plt.grid(True)
                    
                    press_plot_file = plots_dir / f"{simulation_id}_pressure.png"
                    plt.savefig(press_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(str(press_plot_file))
                
                # Create energy plot
                if 'PotEng' in thermo_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(thermo_df['Step'], thermo_df['PotEng'])
                    plt.xlabel('Step')
                    plt.ylabel('Potential Energy')
                    plt.title(f'Potential Energy vs Time - {sim.name}')
                    plt.grid(True)
                    
                    energy_plot_file = plots_dir / f"{simulation_id}_energy.png"
                    plt.savefig(energy_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(str(energy_plot_file))
            
            elif plot_type == "trajectory":
                # Find trajectory file
                traj_files = list(sim.work_dir.glob("*.dump"))
                if not traj_files:
                    raise ValueError("No trajectory file found")
                
                traj_path = traj_files[0]
                trajectory_data = lammps_server.data_handler.read_trajectory_data(traj_path)
                
                if trajectory_data.get('positions'):
                    positions = trajectory_data['positions']
                    
                    # Plot RMSD
                    if len(positions) > 1:
                        rmsd_values = []
                        ref_pos = positions[0]
                        for pos in positions[1:]:
                            rmsd = np.sqrt(np.mean(np.sum((pos - ref_pos) ** 2, axis=1)))
                            rmsd_values.append(rmsd)
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(range(1, len(rmsd_values) + 1), rmsd_values)
                        plt.xlabel('Frame')
                        plt.ylabel('RMSD (Å)')
                        plt.title(f'RMSD vs Frame - {sim.name}')
                        plt.grid(True)
                        
                        rmsd_plot_file = plots_dir / f"{simulation_id}_rmsd.png"
                        plt.savefig(rmsd_plot_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        plot_files.append(str(rmsd_plot_file))
            
            # Save plot information
            sim.add_result("generated_plots", {
                "plot_type": plot_type,
                "plot_files": plot_files
            })
            
            ctx.info(f"Generated {len(plot_files)} plots for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "plot_type": plot_type,
                "plot_files": plot_files,
                "plots_directory": str(plots_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
            raise
    
    @server.tool(
        name="export_data",
        title="Export Data",
        description="Export simulation data in various formats"
    )
    async def export_data(
        ctx: Context,
        simulation_id: str,
        data_type: str = "thermo",
        format: str = "csv",
        log_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export simulation data in various formats.
        
        Args:
            simulation_id: Simulation ID
            data_type: Type of data to export (thermo, trajectory, etc.)
            format: Export format (csv, json, npy, txt)
            log_file: Path to log file (optional)
            
        Returns:
            Export information
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            exported_files = []
            
            if data_type == "thermo":
                # Find log file
                if log_file:
                    log_path = Path(log_file)
                else:
                    log_files = list(sim.work_dir.glob("*.log"))
                    if log_files:
                        log_path = log_files[0]
                    else:
                        raise ValueError("No log file found")
                
                if not log_path.exists():
                    raise ValueError(f"Log file not found: {log_path}")
                
                # Read thermodynamic data
                thermo_df = lammps_server.data_handler.read_thermo_data(log_path)
                
                if thermo_df.empty:
                    raise ValueError("No thermodynamic data found")
                
                # Export data
                filename = f"{simulation_id}_thermo_data"
                export_file = lammps_server.data_handler.export_data(
                    thermo_df, filename, format
                )
                exported_files.append(str(export_file))
            
            elif data_type == "trajectory":
                # Find trajectory file
                traj_files = list(sim.work_dir.glob("*.dump"))
                if not traj_files:
                    raise ValueError("No trajectory file found")
                
                traj_path = traj_files[0]
                trajectory_data = lammps_server.data_handler.read_trajectory_data(traj_path)
                
                if not trajectory_data:
                    raise ValueError("No trajectory data found")
                
                # Export trajectory data
                filename = f"{simulation_id}_trajectory_data"
                export_file = lammps_server.data_handler.export_data(
                    trajectory_data, filename, format
                )
                exported_files.append(str(export_file))
            
            elif data_type == "results":
                # Export simulation results
                results = sim.results
                if results:
                    filename = f"{simulation_id}_results"
                    export_file = lammps_server.data_handler.export_data(
                        results, filename, format
                    )
                    exported_files.append(str(export_file))
                else:
                    raise ValueError("No results data found")
            
            # Save export information
            sim.add_result("exported_data", {
                "data_type": data_type,
                "format": format,
                "exported_files": exported_files
            })
            
            ctx.info(f"Exported {len(exported_files)} files for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "data_type": data_type,
                "format": format,
                "exported_files": exported_files
            }
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
    
    @server.tool(
        name="get_simulation_results",
        title="Get Simulation Results",
        description="Get all results from a simulation"
    )
    async def get_simulation_results(
        ctx: Context,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Get all results from a simulation.
        
        Args:
            simulation_id: Simulation ID
            
        Returns:
            Simulation results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Get results from data handler
            results = lammps_server.data_handler.load_results(simulation_id)
            
            # Combine with simulation results
            all_results = {
                "simulation_results": sim.results,
                "saved_results": results,
                "logs": sim.logs[-50:]  # Last 50 log entries
            }
            
            ctx.info(f"Retrieved results for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "results": all_results
            }
            
        except Exception as e:
            logger.error(f"Failed to get simulation results: {e}")
            raise 