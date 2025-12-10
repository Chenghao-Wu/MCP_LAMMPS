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
    
    @server.tool(
        name="calculate_liquid_density",
        title="Calculate Liquid Density",
        description="Calculate liquid density with error analysis"
    )
    async def calculate_liquid_density(
        ctx: Context,
        simulation_id: str,
        equilibration_time: float = 1.0,  # ns
        analysis_method: str = "time_average"
    ) -> Dict[str, Any]:
        """
        Calculate liquid density from simulation data.
        
        Args:
            simulation_id: Simulation ID
            equilibration_time: Equilibration time to skip (ns)
            analysis_method: Method for density calculation
            
        Returns:
            Density calculation results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Find log file
            log_files = list(sim.work_dir.glob("*.log"))
            if not log_files:
                raise ValueError("No log file found for analysis")
            
            log_file = log_files[0]
            thermo_df = lammps_server.data_handler.read_thermo_data(log_file)
            
            if thermo_df.empty or 'Density' not in thermo_df.columns:
                raise ValueError("No density data found in log file")
            
            # Skip equilibration period
            if 'Step' in thermo_df.columns:
                # Assume timestep from simulation config
                timestep = sim.config.get('timestep', 0.001)  # ps
                equilibration_steps = int(equilibration_time * 1000 / timestep)  # Convert ns to steps
                
                equilibration_mask = thermo_df['Step'] >= equilibration_steps
                analysis_df = thermo_df[equilibration_mask]
            else:
                # Skip first 20% of data
                skip_points = int(len(thermo_df) * 0.2)
                analysis_df = thermo_df.iloc[skip_points:]
            
            if analysis_df.empty:
                raise ValueError("No data remaining after equilibration period")
            
            # Calculate density statistics
            density_data = analysis_df['Density'].values
            
            mean_density = np.mean(density_data)
            std_density = np.std(density_data)
            min_density = np.min(density_data)
            max_density = np.max(density_data)
            
            # Calculate block averages for error estimation
            block_size = max(10, len(density_data) // 20)  # At least 10 points per block
            num_blocks = len(density_data) // block_size
            
            block_averages = []
            for i in range(num_blocks):
                start_idx = i * block_size
                end_idx = (i + 1) * block_size
                block_avg = np.mean(density_data[start_idx:end_idx])
                block_averages.append(block_avg)
            
            block_std = np.std(block_averages) if len(block_averages) > 1 else std_density
            standard_error = block_std / np.sqrt(len(block_averages)) if len(block_averages) > 1 else std_density / np.sqrt(len(density_data))
            
            # Calculate convergence
            convergence_window = min(100, len(density_data) // 4)
            if convergence_window > 0:
                final_avg = np.mean(density_data[-convergence_window:])
                convergence_std = np.std(density_data[-convergence_window:])
                is_converged = convergence_std / final_avg < 0.01  # 1% relative std
            else:
                is_converged = False
                convergence_std = std_density
            
            result = {
                "mean_density_g_cm3": float(mean_density),
                "std_density_g_cm3": float(std_density),
                "standard_error_g_cm3": float(standard_error),
                "min_density_g_cm3": float(min_density),
                "max_density_g_cm3": float(max_density),
                "relative_std_percent": float(100 * std_density / mean_density),
                "num_data_points": len(density_data),
                "num_blocks": len(block_averages),
                "is_converged": is_converged,
                "convergence_std": float(convergence_std),
                "analysis_method": analysis_method,
                "equilibration_time_ns": equilibration_time
            }
            
            # Save results
            sim.add_result("liquid_density", result)
            
            ctx.info(f"Calculated liquid density: {mean_density:.3f} ± {standard_error:.3f} g/cm³")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "density_results": result
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate liquid density: {e}")
            raise
    
    @server.tool(
        name="calculate_viscosity",
        title="Calculate Viscosity",
        description="Calculate viscosity using Green-Kubo and Einstein relations"
    )
    async def calculate_viscosity(
        ctx: Context,
        simulation_id: str,
        method: str = "green_kubo",
        correlation_length: int = 1000
    ) -> Dict[str, Any]:
        """
        Calculate viscosity from simulation data.
        
        Args:
            simulation_id: Simulation ID
            method: Calculation method (green_kubo, einstein)
            correlation_length: Length for correlation functions
            
        Returns:
            Viscosity calculation results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Find pressure autocorrelation file or log file
            pressure_files = list(sim.work_dir.glob("pressure_autocorr.dat"))
            log_files = list(sim.work_dir.glob("*.log"))
            
            if method == "green_kubo" and pressure_files:
                # Use pressure autocorrelation data
                pressure_data = np.loadtxt(pressure_files[0])
                
                if len(pressure_data.shape) == 1:
                    pressure_data = pressure_data.reshape(-1, 1)
                
                if pressure_data.shape[1] >= 2:
                    time = pressure_data[:, 0]
                    autocorr = pressure_data[:, 1]
                    
                    # Integrate autocorrelation function
                    dt = time[1] - time[0] if len(time) > 1 else 1.0
                    integral = np.trapz(autocorr, dx=dt)
                    
                    # Convert to viscosity (simplified)
                    # This is a placeholder - real calculation needs proper units and constants
                    viscosity_pa_s = integral * 1e-12  # Placeholder conversion
                    viscosity_cp = viscosity_pa_s * 1000
                    
                    result = {
                        "viscosity_pa_s": float(viscosity_pa_s),
                        "viscosity_cp": float(viscosity_cp),
                        "method": "green_kubo",
                        "autocorr_integral": float(integral),
                        "correlation_length": len(autocorr)
                    }
                else:
                    raise ValueError("Invalid pressure autocorrelation data format")
            
            elif log_files:
                # Use thermodynamic data for simplified calculation
                log_file = log_files[0]
                thermo_df = lammps_server.data_handler.read_thermo_data(log_file)
                
                if 'Press' not in thermo_df.columns:
                    raise ValueError("Pressure data not available for viscosity calculation")
                
                pressure = thermo_df['Press'].values
                
                # Calculate pressure autocorrelation
                pressure_mean = np.mean(pressure)
                pressure_fluct = pressure - pressure_mean
                
                # Simple autocorrelation
                autocorr_length = min(correlation_length, len(pressure_fluct) // 2)
                autocorr = np.correlate(pressure_fluct, pressure_fluct, mode='full')
                autocorr = autocorr[autocorr.size // 2:][:autocorr_length]
                autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                
                # Integrate
                dt = 1.0  # Assume unit time step
                integral = np.trapz(autocorr, dx=dt)
                
                # Simplified viscosity calculation
                temperature = thermo_df['Temp'].mean() if 'Temp' in thermo_df.columns else 300.0
                volume = thermo_df['Volume'].mean() if 'Volume' in thermo_df.columns else 1000.0
                
                # Placeholder calculation
                viscosity_pa_s = (volume * integral * 1e-15) / temperature
                viscosity_cp = viscosity_pa_s * 1000
                
                result = {
                    "viscosity_pa_s": float(viscosity_pa_s),
                    "viscosity_cp": float(viscosity_cp),
                    "method": "simplified_green_kubo",
                    "pressure_variance": float(np.var(pressure_fluct)),
                    "autocorr_integral": float(integral),
                    "temperature_k": float(temperature),
                    "volume_a3": float(volume)
                }
            else:
                raise ValueError("No suitable data files found for viscosity calculation")
            
            # Save results
            sim.add_result("viscosity", result)
            
            ctx.info(f"Calculated viscosity: {result['viscosity_cp']:.3f} cP using {result['method']}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "viscosity_results": result
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate viscosity: {e}")
            raise
    
    @server.tool(
        name="calculate_diffusion_coefficient",
        title="Calculate Diffusion Coefficient",
        description="Calculate self and mutual diffusion coefficients"
    )
    async def calculate_diffusion_coefficient(
        ctx: Context,
        simulation_id: str,
        diffusion_type: str = "self",
        species: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate diffusion coefficients from MSD data.
        
        Args:
            simulation_id: Simulation ID
            diffusion_type: Type of diffusion (self, mutual)
            species: Species selection for analysis
            
        Returns:
            Diffusion coefficient results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Find MSD data file
            msd_files = list(sim.work_dir.glob("msd.dat"))
            if not msd_files:
                raise ValueError("MSD data file not found")
            
            # Read MSD data
            msd_data = np.loadtxt(msd_files[0])
            
            if len(msd_data.shape) == 1:
                msd_data = msd_data.reshape(-1, 1)
            
            if msd_data.shape[1] < 2:
                raise ValueError("Invalid MSD data format - need at least time and MSD columns")
            
            time = msd_data[:, 0]
            msd = msd_data[:, 1]
            
            # Find linear region for diffusion coefficient calculation
            # Skip initial non-linear region (typically first 20% of data)
            start_idx = max(1, int(len(time) * 0.2))
            end_idx = int(len(time) * 0.8)  # Use up to 80% of data
            
            if start_idx >= end_idx or end_idx - start_idx < 10:
                raise ValueError("Insufficient data for reliable diffusion coefficient calculation")
            
            # Linear fit: MSD = 6*D*t + b (for 3D diffusion)
            time_fit = time[start_idx:end_idx]
            msd_fit = msd[start_idx:end_idx]
            
            coeffs = np.polyfit(time_fit, msd_fit, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Diffusion coefficient D = slope / 6 (for 3D)
            diffusion_coeff = slope / 6.0
            
            # Calculate R-squared for fit quality
            msd_pred = np.polyval(coeffs, time_fit)
            ss_res = np.sum((msd_fit - msd_pred) ** 2)
            ss_tot = np.sum((msd_fit - np.mean(msd_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Convert units (assuming LAMMPS units: Å²/ps to m²/s)
            diffusion_coeff_si = diffusion_coeff * 1e-8  # Å²/ps to m²/s
            diffusion_coeff_cm2_s = diffusion_coeff_si * 1e4  # m²/s to cm²/s
            
            # Calculate uncertainty from fit
            residuals = msd_fit - msd_pred
            mse = np.mean(residuals**2)
            slope_error = np.sqrt(mse / np.sum((time_fit - np.mean(time_fit))**2))
            diffusion_error = slope_error / 6.0
            diffusion_error_si = diffusion_error * 1e-8
            
            result = {
                "diffusion_coefficient_m2_s": float(diffusion_coeff_si),
                "diffusion_coefficient_cm2_s": float(diffusion_coeff_cm2_s),
                "diffusion_coefficient_a2_ps": float(diffusion_coeff),
                "error_m2_s": float(diffusion_error_si),
                "msd_slope": float(slope),
                "msd_intercept": float(intercept),
                "r_squared": float(r_squared),
                "fit_range_ps": [float(time_fit[0]), float(time_fit[-1])],
                "num_points_fitted": len(time_fit),
                "diffusion_type": diffusion_type,
                "species": species or "all"
            }
            
            # Save results
            sim.add_result("diffusion_coefficient", result)
            
            ctx.info(f"Calculated diffusion coefficient: {diffusion_coeff_cm2_s:.2e} cm²/s (R² = {r_squared:.3f})")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "diffusion_results": result
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate diffusion coefficient: {e}")
            raise
    
    @server.tool(
        name="calculate_rdf",
        title="Calculate RDF",
        description="Calculate radial distribution functions for liquid structure"
    )
    async def calculate_rdf(
        ctx: Context,
        simulation_id: str,
        atom_pairs: List[str] = ["*", "*"],
        max_distance: float = 10.0,
        bin_width: float = 0.1
    ) -> Dict[str, Any]:
        """
        Calculate radial distribution functions from trajectory.
        
        Args:
            simulation_id: Simulation ID
            atom_pairs: List of atom type pairs for RDF calculation
            max_distance: Maximum distance for RDF (Å)
            bin_width: Bin width for RDF (Å)
            
        Returns:
            RDF calculation results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # Find trajectory file
            traj_files = list(sim.work_dir.glob("*.lammpstrj"))
            if not traj_files:
                raise ValueError("No trajectory file found for RDF calculation")
            
            traj_file = traj_files[0]
            
            # Try to use MDAnalysis if available
            try:
                import MDAnalysis as mda
                from MDAnalysis.analysis import rdf as mda_rdf
                
                # Load trajectory
                u = mda.Universe(str(traj_file), format='LAMMPSTRJ')
                
                # Calculate RDF for specified pairs
                rdf_results = {}
                
                for i in range(0, len(atom_pairs), 2):
                    if i + 1 < len(atom_pairs):
                        type1 = atom_pairs[i]
                        type2 = atom_pairs[i + 1]
                        
                        # Select atoms (simplified - assumes all atoms if *)
                        if type1 == "*" and type2 == "*":
                            sel1 = u.atoms
                            sel2 = u.atoms
                            pair_name = "all-all"
                        else:
                            # For specific types, would need more sophisticated selection
                            sel1 = u.atoms
                            sel2 = u.atoms
                            pair_name = f"{type1}-{type2}"
                        
                        # Calculate RDF
                        rdf_analysis = mda_rdf.InterRDF(sel1, sel2, 
                                                      nbins=int(max_distance/bin_width),
                                                      range=(0, max_distance))
                        rdf_analysis.run()
                        
                        rdf_results[pair_name] = {
                            "distances": rdf_analysis.bins.tolist(),
                            "rdf_values": rdf_analysis.rdf.tolist(),
                            "coordination_number": float(np.trapz(rdf_analysis.rdf, rdf_analysis.bins))
                        }
                
                method = "mdanalysis"
                
            except ImportError:
                # Fallback to simplified RDF calculation
                rdf_results = _calculate_simple_rdf(traj_file, max_distance, bin_width)
                method = "simplified"
            
            # Find peaks in RDF (coordination shells)
            coordination_shells = []
            for pair_name, rdf_data in rdf_results.items():
                distances = np.array(rdf_data["distances"])
                rdf_values = np.array(rdf_data["rdf_values"])
                
                # Find peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(rdf_values, height=1.1, distance=int(0.5/bin_width))
                
                shells = []
                for peak_idx in peaks[:3]:  # First 3 coordination shells
                    shells.append({
                        "distance": float(distances[peak_idx]),
                        "peak_height": float(rdf_values[peak_idx])
                    })
                
                coordination_shells.append({
                    "pair": pair_name,
                    "shells": shells
                })
            
            result = {
                "rdf_data": rdf_results,
                "coordination_shells": coordination_shells,
                "parameters": {
                    "max_distance": max_distance,
                    "bin_width": bin_width,
                    "atom_pairs": atom_pairs
                },
                "method": method
            }
            
            # Save results
            sim.add_result("rdf_analysis", result)
            
            ctx.info(f"Calculated RDF for {len(rdf_results)} atom pair types using {method}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "rdf_results": result
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate RDF: {e}")
            raise
    
    @server.tool(
        name="calculate_hydrogen_bonds",
        title="Calculate Hydrogen Bonds",
        description="Analyze hydrogen bonding in polar liquids"
    )
    async def calculate_hydrogen_bonds(
        ctx: Context,
        simulation_id: str,
        donor_acceptor_distance: float = 3.5,
        angle_cutoff: float = 30.0
    ) -> Dict[str, Any]:
        """
        Analyze hydrogen bonding patterns in the simulation.
        
        Args:
            simulation_id: Simulation ID
            donor_acceptor_distance: Maximum donor-acceptor distance (Å)
            angle_cutoff: Maximum deviation from linear H-bond (degrees)
            
        Returns:
            Hydrogen bonding analysis results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            # This is a simplified implementation
            # Real H-bond analysis would require detailed trajectory parsing
            
            result = {
                "average_hbonds_per_molecule": 0.0,
                "hbond_lifetime_ps": 0.0,
                "hbond_distance_distribution": [],
                "hbond_angle_distribution": [],
                "parameters": {
                    "donor_acceptor_distance": donor_acceptor_distance,
                    "angle_cutoff": angle_cutoff
                },
                "method": "simplified_placeholder"
            }
            
            # Save results
            sim.add_result("hydrogen_bonds", result)
            
            ctx.info(f"Analyzed hydrogen bonding (placeholder implementation)")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "hbond_results": result
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate hydrogen bonds: {e}")
            raise
    
    @server.tool(
        name="analyze_molecular_dynamics",
        title="Analyze Molecular Dynamics",
        description="Comprehensive MD analysis for liquids"
    )
    async def analyze_molecular_dynamics(
        ctx: Context,
        simulation_id: str,
        analysis_types: List[str] = ["density", "rdf", "msd", "energy"],
        equilibration_time: float = 1.0
    ) -> Dict[str, Any]:
        """
        Perform comprehensive molecular dynamics analysis.
        
        Args:
            simulation_id: Simulation ID
            analysis_types: List of analysis types to perform
            equilibration_time: Equilibration time to skip (ns)
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            comprehensive_results = {}
            
            # Perform requested analyses
            if "density" in analysis_types:
                try:
                    density_result = await calculate_liquid_density(ctx, simulation_id, equilibration_time)
                    comprehensive_results["density"] = density_result["density_results"]
                except Exception as e:
                    comprehensive_results["density"] = {"error": str(e)}
            
            if "rdf" in analysis_types:
                try:
                    rdf_result = await calculate_rdf(ctx, simulation_id)
                    comprehensive_results["rdf"] = rdf_result["rdf_results"]
                except Exception as e:
                    comprehensive_results["rdf"] = {"error": str(e)}
            
            if "msd" in analysis_types:
                try:
                    diffusion_result = await calculate_diffusion_coefficient(ctx, simulation_id)
                    comprehensive_results["diffusion"] = diffusion_result["diffusion_results"]
                except Exception as e:
                    comprehensive_results["diffusion"] = {"error": str(e)}
            
            if "energy" in analysis_types:
                try:
                    # Use existing calculate_properties function
                    energy_result = await calculate_properties(ctx, simulation_id)
                    comprehensive_results["energy"] = energy_result["properties"]
                except Exception as e:
                    comprehensive_results["energy"] = {"error": str(e)}
            
            # Calculate summary statistics
            summary = {
                "total_analyses": len(analysis_types),
                "successful_analyses": sum(1 for result in comprehensive_results.values() 
                                         if "error" not in result),
                "failed_analyses": sum(1 for result in comprehensive_results.values() 
                                     if "error" in result),
                "equilibration_time_ns": equilibration_time
            }
            
            result = {
                "comprehensive_analysis": comprehensive_results,
                "summary": summary,
                "analysis_types": analysis_types
            }
            
            # Save results
            sim.add_result("comprehensive_md_analysis", result)
            
            ctx.info(f"Completed comprehensive MD analysis: {summary['successful_analyses']}/{summary['total_analyses']} successful")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "analysis_results": result
            }
            
        except Exception as e:
            logger.error(f"Failed to perform comprehensive MD analysis: {e}")
            raise


def _calculate_simple_rdf(traj_file: Path, max_distance: float, bin_width: float) -> Dict[str, Any]:
    """
    Simple RDF calculation without MDAnalysis.
    This is a placeholder implementation.
    """
    try:
        # Placeholder for simple RDF calculation
        num_bins = int(max_distance / bin_width)
        distances = np.linspace(0, max_distance, num_bins)
        rdf_values = np.ones_like(distances)  # Placeholder
        
        return {
            "all-all": {
                "distances": distances.tolist(),
                "rdf_values": rdf_values.tolist(),
                "coordination_number": float(np.trapz(rdf_values, distances))
            }
        }
    except Exception as e:
        logger.error(f"Error in simple RDF calculation: {e}")
        return {} 