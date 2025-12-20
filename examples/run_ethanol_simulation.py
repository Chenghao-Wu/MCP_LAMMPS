#!/usr/bin/env python3
"""
Script to create and run an ethanol liquid simulation using OpenFF.

This script:
1. Creates ethanol molecules from SMILES (CCO)
2. Builds a liquid box with target density
3. Generates LAMMPS input files with OpenFF force field
4. Creates a simulation configuration

Ethanol properties:
- SMILES: CCO
- Density: ~0.789 g/cm³ at 298 K
- Molecular formula: C₂H₆O
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import mcp_lammps
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mcp_lammps.data_handler import DataHandler
    from mcp_lammps.simulation_manager import SimulationManager
    from mcp_lammps.lammps_interface import LAMMPSInterface
except ImportError as e:
    logger.error(f"Failed to import mcp_lammps modules: {e}")
    logger.error("Make sure the package is installed or in PYTHONPATH")
    sys.exit(1)


def create_ethanol_simulation(
    num_molecules: int = 100,
    target_density: float = 0.789,
    temperature: float = 298.0,
    pressure: float = 1.0,
    equilibration_steps: int = 5000,
    production_steps: int = 50000,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Create a complete ethanol liquid simulation setup.
    
    Args:
        num_molecules: Number of ethanol molecules
        target_density: Target density in g/cm³ (default: 0.789 for ethanol at 298K)
        temperature: Simulation temperature in K
        pressure: Simulation pressure in atm
        equilibration_steps: Number of equilibration steps
        production_steps: Number of production steps
        output_dir: Output directory (default: examples directory)
        
    Returns:
        Dictionary with simulation information
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Creating Ethanol Liquid Simulation")
    logger.info("=" * 70)
    logger.info(f"Number of molecules: {num_molecules}")
    logger.info(f"Target density: {target_density} g/cm³")
    logger.info(f"Temperature: {temperature} K")
    logger.info(f"Pressure: {pressure} atm")
    logger.info(f"Equilibration steps: {equilibration_steps}")
    logger.info(f"Production steps: {production_steps}")
    
    #try:
    # Initialize handlers
    logger.info("\nInitializing MCP LAMMPS components...")
    data_handler = DataHandler(output_dir)
    simulation_manager = SimulationManager(output_dir)
    lammps_interface = LAMMPSInterface()
    
    # Define ethanol molecule
    ethanol_smiles = "CCO"
    molecule_name = "ethanol"
    
    logger.info(f"\nCreating liquid box with {num_molecules} {molecule_name} molecules...")
    logger.info(f"SMILES: {ethanol_smiles}")
    
    # Create molecules list for liquid box
    molecules = [
        {
            "smiles": ethanol_smiles,
            "count": num_molecules,
            "name": molecule_name
        }
    ]
    
    # Create liquid box using OpenFF
    box_file = data_handler.create_liquid_box_file(
        molecules=molecules,
        target_density=target_density,
        box_type="cubic"
    )
    
    logger.info(f"✓ Liquid box created: {box_file.name}")
    
    # Load metadata
    metadata_file = box_file.with_suffix(".json")
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"✓ System metadata:")
        logger.info(f"  - Total atoms: {metadata.get('total_atoms', 'N/A')}")
        logger.info(f"  - Total molecules: {metadata.get('total_molecules', 'N/A')}")
        logger.info(f"  - Box dimensions: {metadata.get('box_dimensions', 'N/A')}")
        
    #     # Check for auto-generated LAMMPS script
    #     script_file = box_file.with_suffix(".in")
    #     if script_file.exists():
    #         logger.info(f"✓ LAMMPS script generated: {script_file.name}")
    #     else:
    #         logger.warning(f"⚠ LAMMPS script not found: {script_file.name}")
    #         logger.info("  (Script may be generated during simulation run)")
        
    #     # Create simulation configuration
    #     logger.info("\nCreating simulation configuration...")
    #     config = {
    #         "name": "ethanol_liquid",
    #         "type": "organic_liquid_simulation",
    #         "molecules": molecules,
    #         "target_density": target_density,
    #         "force_field": "openff-sage-2.2.0",
    #         "charge_method": "am1bcc",
    #         "temperature": temperature,
    #         "pressure": pressure,
    #         "timestep": 0.001,
    #         "equilibration_steps": equilibration_steps,
    #         "production_steps": production_steps,
    #         "structure_file": str(box_file),
    #         "script_file": str(script_file) if script_file.exists() else None
    #     }
        
    #     # Save configuration
    #     config_file = output_dir / "ethanol_simulation_config.json"
    #     with open(config_file, 'w') as f:
    #         json.dump(config, f, indent=2)
        
    #     logger.info(f"✓ Configuration saved: {config_file.name}")
        
    #     # Create simulation in manager
    #     simulation_id = simulation_manager.create_simulation("ethanol_liquid", config)
    #     logger.info(f"✓ Simulation created with ID: {simulation_id}")
        
    #     # Summary
    #     logger.info("\n" + "=" * 70)
    #     logger.info("SIMULATION SETUP COMPLETE")
    #     logger.info("=" * 70)
    #     logger.info(f"\nFiles created:")
    #     logger.info(f"  - Structure file: {box_file.name}")
    #     if script_file.exists():
    #         logger.info(f"  - LAMMPS script: {script_file.name}")
    #     logger.info(f"  - Configuration: {config_file.name}")
    #     logger.info(f"  - Metadata: {metadata_file.name}")
        
    #     logger.info(f"\nTo run the simulation:")
    #     if script_file.exists():
    #         logger.info(f"  lmp -in {script_file.name}")
    #     else:
    #         logger.info(f"  (Use the MCP server tools to run the simulation)")
        
    #     return {
    #         "simulation_id": simulation_id,
    #         "name": "ethanol_liquid",
    #         "status": "created",
    #         "structure_file": str(box_file),
    #         "script_file": str(script_file) if script_file.exists() else None,
    #         "config_file": str(config_file),
    #         "metadata": metadata,
    #         "config": config
    #     }
        
    # except Exception as e:
    #     logger.error(f"\n✗ Failed to create ethanol simulation: {e}")
    #     logger.exception("Error details:")
    #     raise


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create an ethanol liquid simulation using OpenFF"
    )
    parser.add_argument(
        "-n", "--num-molecules",
        type=int,
        default=100,
        help="Number of ethanol molecules (default: 100)"
    )
    parser.add_argument(
        "-d", "--density",
        type=float,
        default=0.789,
        help="Target density in g/cm³ (default: 0.789 for ethanol at 298K)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=298.0,
        help="Temperature in K (default: 298.0)"
    )
    parser.add_argument(
        "-p", "--pressure",
        type=float,
        default=1.0,
        help="Pressure in atm (default: 1.0)"
    )
    parser.add_argument(
        "--equilibration-steps",
        type=int,
        default=5000,
        help="Number of equilibration steps (default: 5000)"
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=50000,
        help="Number of production steps (default: 50000)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: examples directory)"
    )
    
    args = parser.parse_args()
    
    # Convert output_dir to Path if provided
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        result = create_ethanol_simulation(
            num_molecules=args.num_molecules,
            target_density=args.density,
            temperature=args.temperature,
            pressure=args.pressure,
            equilibration_steps=args.equilibration_steps,
            production_steps=args.production_steps,
            output_dir=output_dir
        )
        
        logger.info("\n✓ Ethanol simulation setup completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Simulation setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


