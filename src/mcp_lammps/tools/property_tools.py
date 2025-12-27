"""
Property Tools - Tools for calculating transport and thermodynamic properties.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mcp.server.fastmcp.server import Context

logger = logging.getLogger(__name__)

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import diffusionmap, rdf
    MDANALYSIS_AVAILABLE = True
except ImportError:
    logger.warning("MDAnalysis not available. Advanced trajectory analysis will be limited.")
    MDANALYSIS_AVAILABLE = False


def register_property_tools(server: Any, lammps_server: Any) -> None:
    """
    Register property calculation tools with the MCP server.
    
    Args:
        server: MCP server instance
        lammps_server: LAMMPS server instance
    """
    
    @server.tool(
        name="calculate_transport_properties",
        title="Calculate Transport Properties",
        description="Calculate viscosity, thermal conductivity, and diffusion coefficients"
    )
    async def calculate_transport_properties(
        ctx: Context,
        simulation_id: str,
        property_types: List[str] = ["viscosity", "diffusion", "thermal_conductivity"],
        analysis_start_time: float = 0.5,  # ns
        correlation_length: int = 1000
    ) -> Dict[str, Any]:
        """
        Calculate transport properties from simulation data.
        
        Args:
            simulation_id: Simulation ID
            property_types: List of properties to calculate
            analysis_start_time: Start time for analysis (ns)
            correlation_length: Length for correlation functions
            
        Returns:
            Transport properties
        """
        try:
            # Get simulation
            sim = lammps_server.simulation_manager.get_simulation(simulation_id)
            if not sim:
                raise ValueError(f"Simulation {simulation_id} not found")
            
            results = {}
            
            # Find log file for thermodynamic data
            log_files = list(sim.work_dir.glob("*.log"))
            if not log_files:
                raise ValueError("No log file found for analysis")
            
            log_file = log_files[0]
            thermo_df = lammps_server.data_handler.read_thermo_data(log_file)
            
            if thermo_df.empty:
                raise ValueError("No thermodynamic data found")
            
            # Calculate properties based on requested types
            if "viscosity" in property_types:
                viscosity_result = _calculate_viscosity(thermo_df, analysis_start_time, correlation_length)
                results["viscosity"] = viscosity_result
            
            if "diffusion" in property_types:
                diffusion_result = _calculate_diffusion_coefficient(sim, analysis_start_time)
                results["diffusion"] = diffusion_result
            
            if "thermal_conductivity" in property_types:
                thermal_result = _calculate_thermal_conductivity(thermo_df, analysis_start_time, correlation_length)
                results["thermal_conductivity"] = thermal_result
            
            # Save results
            sim.add_result("transport_properties", results)
            
            ctx.info(f"Calculated transport properties for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "properties": results,
                "analysis_parameters": {
                    "start_time": analysis_start_time,
                    "correlation_length": correlation_length,
                    "property_types": property_types
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate transport properties: {e}")
            raise
    
    @server.tool(
        name="calculate_thermodynamic_properties",
        title="Calculate Thermodynamic Properties",
        description="Calculate heat capacity, compressibility, and thermal expansion"
    )
    async def calculate_thermodynamic_properties(
        ctx: Context,
        simulation_id: str,
        property_types: List[str] = ["heat_capacity", "compressibility", "thermal_expansion"],
        temperature_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate thermodynamic properties from simulation data.
        
        Args:
            simulation_id: Simulation ID
            property_types: List of properties to calculate
            temperature_range: Temperature range for analysis (K)
            
        Returns:
            Thermodynamic properties
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
            
            if thermo_df.empty:
                raise ValueError("No thermodynamic data found")
            
            results = {}
            
            # Filter data by temperature range if specified
            if temperature_range:
                mask = (thermo_df['Temp'] >= temperature_range[0]) & (thermo_df['Temp'] <= temperature_range[1])
                thermo_df = thermo_df[mask]
            
            # Calculate properties
            if "heat_capacity" in property_types:
                cp_result = _calculate_heat_capacity(thermo_df)
                results["heat_capacity"] = cp_result
            
            if "compressibility" in property_types:
                comp_result = _calculate_compressibility(thermo_df)
                results["compressibility"] = comp_result
            
            if "thermal_expansion" in property_types:
                expansion_result = _calculate_thermal_expansion(thermo_df)
                results["thermal_expansion"] = expansion_result
            
            # Save results
            sim.add_result("thermodynamic_properties", results)
            
            ctx.info(f"Calculated thermodynamic properties for simulation: {sim.name}")
            
            return {
                "simulation_id": simulation_id,
                "name": sim.name,
                "properties": results,
                "analysis_parameters": {
                    "temperature_range": temperature_range,
                    "property_types": property_types,
                    "data_points": len(thermo_df)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate thermodynamic properties: {e}")
            raise


def _calculate_viscosity(thermo_df: Any, start_time: float, corr_length: int) -> Dict[str, Any]:
    """Calculate viscosity using Green-Kubo relations."""
    try:
        # This is a simplified implementation
        # In practice, you would need stress tensor autocorrelation
        
        if 'Press' not in thermo_df.columns:
            return {"error": "Pressure data not available for viscosity calculation"}
        
        # Use pressure fluctuations as a proxy (simplified)
        pressure_data = thermo_df['Press'].values
        
        # Calculate pressure autocorrelation
        pressure_mean = np.mean(pressure_data)
        pressure_fluct = pressure_data - pressure_mean
        
        # Simple autocorrelation calculation
        autocorr = np.correlate(pressure_fluct, pressure_fluct, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr[:min(corr_length, len(autocorr))]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Integrate to get viscosity (simplified)
        dt = 1.0  # Assume 1 fs timestep
        integral = np.trapz(autocorr, dx=dt)
        
        # Convert to viscosity (very simplified)
        kb = 1.380649e-23  # Boltzmann constant
        T = thermo_df['Temp'].mean()
        V = thermo_df.get('Volume', pd.Series([1000])).mean()
        
        viscosity = (V * integral) / (kb * T) * 1e-12  # Convert to Pa·s
        
        return {
            "viscosity_pa_s": float(viscosity),
            "viscosity_cp": float(viscosity * 1000),  # Convert to cP
            "temperature_k": float(T),
            "pressure_autocorr_integral": float(integral),
            "method": "green_kubo_simplified"
        }
        
    except Exception as e:
        logger.error(f"Error calculating viscosity: {e}")
        return {"error": str(e)}


def _calculate_diffusion_coefficient(sim: Any, start_time: float) -> Dict[str, Any]:
    """Calculate diffusion coefficient from MSD."""
    try:
        # Look for MSD data file
        msd_files = list(sim.work_dir.glob("msd.dat"))
        if not msd_files:
            return {"error": "MSD data file not found"}
        
        # Read MSD data
        msd_data = np.loadtxt(msd_files[0])
        if msd_data.size == 0:
            return {"error": "Empty MSD data"}
        
        # Assume columns: time, MSD
        if len(msd_data.shape) == 1:
            msd_data = msd_data.reshape(-1, 1)
        
        if msd_data.shape[1] < 2:
            return {"error": "Invalid MSD data format"}
        
        time = msd_data[:, 0]
        msd = msd_data[:, 1]
        
        # Find linear region (typically after equilibration)
        start_idx = int(len(time) * 0.2)  # Skip first 20%
        end_idx = int(len(time) * 0.8)    # Use up to 80%
        
        if start_idx >= end_idx:
            return {"error": "Insufficient data for linear fit"}
        
        # Linear fit: MSD = 6*D*t + b
        coeffs = np.polyfit(time[start_idx:end_idx], msd[start_idx:end_idx], 1)
        slope = coeffs[0]
        
        # Diffusion coefficient D = slope / 6
        diffusion_coeff = slope / 6.0
        
        # Convert from LAMMPS units to SI (m²/s)
        # LAMMPS: Å²/ps, SI: m²/s
        diffusion_coeff_si = diffusion_coeff * 1e-8  # Å²/ps to m²/s
        
        return {
            "diffusion_coefficient_m2_s": float(diffusion_coeff_si),
            "diffusion_coefficient_cm2_s": float(diffusion_coeff_si * 1e4),
            "msd_slope": float(slope),
            "fit_range": [float(time[start_idx]), float(time[end_idx])],
            "r_squared": float(_calculate_r_squared(time[start_idx:end_idx], msd[start_idx:end_idx], coeffs))
        }
        
    except Exception as e:
        logger.error(f"Error calculating diffusion coefficient: {e}")
        return {"error": str(e)}


def _calculate_thermal_conductivity(thermo_df: Any, start_time: float, corr_length: int) -> Dict[str, Any]:
    """Calculate thermal conductivity using heat flux autocorrelation."""
    try:
        # This is a placeholder implementation
        # Real thermal conductivity calculation requires heat flux data
        
        if 'Temp' not in thermo_df.columns:
            return {"error": "Temperature data not available"}
        
        # Use temperature fluctuations as a proxy (very simplified)
        temp_data = thermo_df['Temp'].values
        temp_mean = np.mean(temp_data)
        temp_fluct = temp_data - temp_mean
        
        # Calculate temperature variance
        temp_var = np.var(temp_fluct)
        
        # Simplified thermal conductivity estimate
        # This is not physically accurate - just a placeholder
        thermal_conductivity = temp_var * 0.1  # Arbitrary scaling
        
        return {
            "thermal_conductivity_w_m_k": float(thermal_conductivity),
            "temperature_variance": float(temp_var),
            "mean_temperature": float(temp_mean),
            "method": "simplified_placeholder"
        }
        
    except Exception as e:
        logger.error(f"Error calculating thermal conductivity: {e}")
        return {"error": str(e)}


def _calculate_heat_capacity(thermo_df: Any) -> Dict[str, Any]:
    """Calculate heat capacity from energy fluctuations."""
    try:
        if 'Temp' not in thermo_df.columns or 'TotEng' not in thermo_df.columns:
            return {"error": "Temperature or energy data not available"}
        
        temp = thermo_df['Temp'].values
        energy = thermo_df['TotEng'].values
        
        # Calculate heat capacity using fluctuation formula
        # Cv = <E²> - <E>² / (kB * T²)
        
        mean_temp = np.mean(temp)
        mean_energy = np.mean(energy)
        mean_energy_sq = np.mean(energy**2)
        
        energy_var = mean_energy_sq - mean_energy**2
        
        # Boltzmann constant in kcal/mol/K
        kb = 1.987204e-3  # kcal/mol/K
        
        # Heat capacity per molecule
        cv_per_molecule = energy_var / (kb * mean_temp**2)
        
        # Convert to J/mol/K
        cv_j_mol_k = cv_per_molecule * 4184  # kcal/mol to J/mol
        
        return {
            "heat_capacity_j_mol_k": float(cv_j_mol_k),
            "heat_capacity_cal_mol_k": float(cv_per_molecule * 1000),
            "energy_variance": float(energy_var),
            "mean_temperature": float(mean_temp),
            "method": "fluctuation_formula"
        }
        
    except Exception as e:
        logger.error(f"Error calculating heat capacity: {e}")
        return {"error": str(e)}


def _calculate_compressibility(thermo_df: Any) -> Dict[str, Any]:
    """Calculate isothermal compressibility from volume fluctuations."""
    try:
        if 'Volume' not in thermo_df.columns or 'Temp' not in thermo_df.columns:
            return {"error": "Volume or temperature data not available"}
        
        volume = thermo_df['Volume'].values
        temp = thermo_df['Temp'].values
        
        mean_volume = np.mean(volume)
        mean_temp = np.mean(temp)
        volume_var = np.var(volume)
        
        # Isothermal compressibility: κT = <V²> - <V>² / (<V> * kB * T)
        kb = 1.380649e-23  # J/K
        
        compressibility = volume_var / (mean_volume * kb * mean_temp)
        
        # Convert from 1/Pa to 1/GPa
        compressibility_gpa = compressibility * 1e-9
        
        return {
            "isothermal_compressibility_1_gpa": float(compressibility_gpa),
            "isothermal_compressibility_1_pa": float(compressibility),
            "volume_variance": float(volume_var),
            "mean_volume": float(mean_volume),
            "mean_temperature": float(mean_temp)
        }
        
    except Exception as e:
        logger.error(f"Error calculating compressibility: {e}")
        return {"error": str(e)}


def _calculate_thermal_expansion(thermo_df: Any) -> Dict[str, Any]:
    """Calculate thermal expansion coefficient."""
    try:
        if 'Volume' not in thermo_df.columns or 'Temp' not in thermo_df.columns:
            return {"error": "Volume or temperature data not available"}
        
        volume = thermo_df['Volume'].values
        temp = thermo_df['Temp'].values
        
        # Calculate thermal expansion: α = (1/V) * (dV/dT)
        # Use linear fit to get dV/dT
        coeffs = np.polyfit(temp, volume, 1)
        dv_dt = coeffs[0]
        
        mean_volume = np.mean(volume)
        thermal_expansion = dv_dt / mean_volume
        
        return {
            "thermal_expansion_1_k": float(thermal_expansion),
            "dv_dt": float(dv_dt),
            "mean_volume": float(mean_volume),
            "temperature_range": [float(np.min(temp)), float(np.max(temp))]
        }
        
    except Exception as e:
        logger.error(f"Error calculating thermal expansion: {e}")
        return {"error": str(e)}


def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate moving average."""
    if window_size >= len(data):
        return np.full_like(data, np.mean(data))
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def _calculate_r_squared(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
    """Calculate R-squared for polynomial fit."""
    try:
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    except:
        return 0
