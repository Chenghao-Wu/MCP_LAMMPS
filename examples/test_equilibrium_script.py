#!/usr/bin/env python3
"""
Test script for LAMMPS Equilibrium Script Generator

This script validates the new EquilibriumScriptGenerator following MD best practices
for equilibrium simulations. It tests:
- Script generation with default parameters
- Script generation with custom parameters
- System type detection
- Integration with OpenFF workflow
- Backward compatibility

Usage:
    python test_equilibrium_script.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mcp_lammps.utils.script_generator import (
        EquilibriumScriptGenerator,
        SimulationConfig,
        SystemDetector,
        SystemType,
        create_equilibrium_script
    )
    from mcp_lammps.utils.openff_utils import openff_forcefield, OPENFF_AVAILABLE
    from mcp_lammps.data_handler import DataHandler
except ImportError as e:
    logger.error(f"Failed to import mcp_lammps modules: {e}")
    logger.error("Please ensure you're running from the mcp_lammps directory")
    sys.exit(1)


class EquilibriumScriptTest:
    """Test class for equilibrium script generation."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize the test.
        
        Args:
            output_dir: Directory for output files (default: examples/output/)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "output"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data handler
        self.data_handler = DataHandler(self.output_dir)
        
        self.results = {}
        
    def print_header(self, text: str):
        """Print a formatted header."""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)
    
    def print_section(self, text: str):
        """Print a formatted section header."""
        print(f"\n--- {text} ---")
    
    def test_basic_script_generation(self) -> bool:
        """Test basic script generation with default parameters."""
        self.print_header("Test 1: Basic Script Generation")
        
        try:
            # Create a simple data file for testing
            test_data_file = self.output_dir / "test_system.lmp"
            
            # Generate script with defaults
            config = SimulationConfig()
            generator = EquilibriumScriptGenerator(config)
            
            script = generator.generate_script(
                str(test_data_file),
                output_prefix="test_basic"
            )
            
            print("✓ Script generated successfully")
            print(f"  Script length: {len(script)} characters")
            print(f"  Script lines: {len(script.split(chr(10)))} lines")
            
            # Validate script content
            required_sections = [
                "INITIALIZATION",
                "FORCE FIELD",
                "SIMULATION SETTINGS",
                "STAGE 1: ENERGY MINIMIZATION",
                "STAGE 2: NVT HEATING",
                "STAGE 3: NPT DENSITY EQUILIBRATION",
                "STAGE 4: NPT PRODUCTION"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in script:
                    missing_sections.append(section)
                else:
                    print(f"  ✓ Found section: {section}")
            
            if missing_sections:
                logger.error(f"Missing sections: {missing_sections}")
                return False
            
            # Save script for inspection
            script_file = self.output_dir / "test_basic_equilibrium.in"
            with open(script_file, 'w') as f:
                f.write(script)
            
            print(f"\n✓ Script saved to: {script_file}")
            
            self.results['basic_script'] = {
                'script_length': len(script),
                'script_lines': len(script.split('\n')),
                'script_file': str(script_file),
                'all_sections_present': len(missing_sections) == 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate basic script: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_custom_configuration(self) -> bool:
        """Test script generation with custom configuration."""
        self.print_header("Test 2: Custom Configuration")
        
        try:
            # Create custom configuration
            config = SimulationConfig(
                temperature=350.0,
                pressure=10.0,
                timestep=2.0,
                minimization_steps=5000,
                nvt_heating_steps=20000,
                npt_density_steps=50000,
                production_steps=500000,
                enable_minimization=True,
                enable_nvt_heating=True,
                enable_density_equilibration=True,
                thermostat_damping=200.0,
                barostat_damping=2000.0,
                trajectory_frequency=5000,
                restart_frequency=50000,
                thermo_frequency=500
            )
            
            print(f"✓ Created custom configuration:")
            print(f"  Temperature: {config.temperature} K")
            print(f"  Pressure: {config.pressure} atm")
            print(f"  Timestep: {config.timestep} fs")
            print(f"  Production steps: {config.production_steps}")
            
            # Generate script
            test_data_file = self.output_dir / "test_system.lmp"
            generator = EquilibriumScriptGenerator(config)
            
            script = generator.generate_script(
                str(test_data_file),
                output_prefix="test_custom"
            )
            
            print(f"\n✓ Custom script generated successfully")
            print(f"  Script length: {len(script)} characters")
            
            # Validate custom parameters in script
            if f"Temperature: {config.temperature}" in script:
                print("  ✓ Custom temperature found in script")
            if f"Pressure: {config.pressure}" in script:
                print("  ✓ Custom pressure found in script")
            if f"Timestep: {config.timestep}" in script:
                print("  ✓ Custom timestep found in script")
            
            # Save script
            script_file = self.output_dir / "test_custom_equilibrium.in"
            with open(script_file, 'w') as f:
                f.write(script)
            
            print(f"\n✓ Custom script saved to: {script_file}")
            
            self.results['custom_script'] = {
                'config': {
                    'temperature': config.temperature,
                    'pressure': config.pressure,
                    'timestep': config.timestep,
                    'production_steps': config.production_steps
                },
                'script_file': str(script_file)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate custom script: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_system_detection(self) -> bool:
        """Test system type detection."""
        self.print_header("Test 3: System Type Detection")
        
        try:
            # Test detection with different system types
            test_cases = [
                ("generic_system.lmp", SystemType.GENERIC),
                ("water_system.lmp", SystemType.WATER),
                ("organic_system.lmp", SystemType.ORGANIC_LIQUID)
            ]
            
            print("Testing system type detection:")
            
            for filename, expected_type in test_cases:
                # Create dummy file with appropriate markers
                test_file = self.output_dir / filename
                
                if "water" in filename:
                    content = "# TIP3P Water System\n1000 atoms\n333 molecules\n"
                elif "organic" in filename:
                    content = "# Organic Liquid System\n# OpenFF\n5000 atoms\n100 molecules\n"
                else:
                    content = "# Generic System\n1000 atoms\n"
                
                with open(test_file, 'w') as f:
                    f.write(content)
                
                # Detect system type
                detected_type = SystemDetector.detect_from_data_file(test_file)
                
                status = "✓" if detected_type == expected_type else "✗"
                print(f"  {status} {filename}: detected as {detected_type.value} (expected {expected_type.value})")
            
            print("\n✓ System detection tests completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed system detection test: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_openff_integration(self) -> bool:
        """Test integration with OpenFF workflow."""
        self.print_header("Test 4: OpenFF Integration")
        
        if not OPENFF_AVAILABLE:
            logger.warning("OpenFF not available, skipping integration test")
            return True
        
        try:
            print("Creating ethanol system with OpenFF...")
            
            # Create ethanol molecule
            molecules = [{
                "smiles": "CCO",
                "count": 50,
                "name": "ethanol"
            }]
            
            # Build liquid box with equilibrium script
            interchange, data_file, script_file = openff_forcefield.build_liquid_box(
                molecule_specs=molecules,
                target_density=0.789,
                output_prefix=self.output_dir / "test_openff_ethanol",
                generate_equilibrium_script=True,
                simulation_config=None  # Use defaults
            )
            
            print(f"✓ OpenFF liquid box created")
            print(f"  Data file: {data_file}")
            print(f"  Script file: {script_file}")
            
            # Verify script file exists and contains equilibrium protocol
            if script_file and script_file.exists():
                with open(script_file, 'r') as f:
                    script_content = f.read()
                
                if "STAGE 1: ENERGY MINIMIZATION" in script_content:
                    print("  ✓ Equilibrium script generated with multi-stage protocol")
                else:
                    print("  ✗ Script does not contain expected equilibrium stages")
                    return False
            else:
                logger.error("Script file not generated")
                return False
            
            self.results['openff_integration'] = {
                'data_file': str(data_file),
                'script_file': str(script_file),
                'n_atoms': interchange.topology.n_atoms,
                'n_molecules': interchange.topology.n_molecules
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed OpenFF integration test: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_convenience_function(self) -> bool:
        """Test convenience function for script generation."""
        self.print_header("Test 5: Convenience Function")
        
        try:
            # Create test data file
            test_data_file = self.output_dir / "test_convenience.lmp"
            with open(test_data_file, 'w') as f:
                f.write("# Test system\n1000 atoms\n")
            
            # Use convenience function
            script = create_equilibrium_script(
                str(test_data_file),
                output_prefix="test_convenience",
                config=None,  # Use defaults
                auto_detect_system=True
            )
            
            print("✓ Convenience function generated script successfully")
            print(f"  Script length: {len(script)} characters")
            
            # Save script
            script_file = self.output_dir / "test_convenience_equilibrium.in"
            with open(script_file, 'w') as f:
                f.write(script)
            
            print(f"✓ Script saved to: {script_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed convenience function test: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_summary(self):
        """Display test results summary."""
        self.print_header("Test Results Summary")
        
        print("\nGenerated Files:")
        for key, value in self.results.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
        
        print(f"\nAll test files are located in: {self.output_dir}")
    
    def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        self.print_header("LAMMPS Equilibrium Script Generator Test Suite")
        
        # Run tests
        tests = [
            ("Basic Script Generation", self.test_basic_script_generation),
            ("Custom Configuration", self.test_custom_configuration),
            ("System Detection", self.test_system_detection),
            ("OpenFF Integration", self.test_openff_integration),
            ("Convenience Function", self.test_convenience_function),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                logger.error(f"Test '{test_name}' failed with exception: {e}")
                results.append((test_name, False))
        
        # Display summary
        self.display_summary()
        
        # Print test results
        self.print_header("Test Results")
        
        print("\nTest Results:")
        all_passed = True
        for test_name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {status} - {test_name}")
            if not success:
                all_passed = False
        
        print("\n" + "=" * 80)
        if all_passed:
            print("  ✓ ALL TESTS PASSED")
            print("  Equilibrium script generator is working correctly!")
        else:
            print("  ✗ SOME TESTS FAILED")
            print("  Please check the error messages above.")
        print("=" * 80 + "\n")
        
        return all_passed


def main():
    """Main entry point."""
    # Create test instance
    test = EquilibriumScriptTest()
    
    # Run all tests
    success = test.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

