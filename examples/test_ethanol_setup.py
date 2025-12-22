#!/usr/bin/env python3
"""
Test script for MCP LAMMPS Ethanol System Setup

This script demonstrates the capability of mcp_lammps to set up an organic liquid
system (ethanol) and generate LAMMPS data files ready for simulation using the
modern OpenFF Sage 2.2.1 force field with NAGL AM1-BCC charges.

Features tested:
- SMILES to 3D structure conversion using OpenFF Toolkit
- Automatic atom typing via OpenFF Sage 2.2.1
- NAGL AM1-BCC charge assignment (neural network-based)
- Multi-component liquid box creation using OpenFF Interchange
- LAMMPS data and input file generation
- System validation and information display

Note: This tests the modern OpenFF-based system (migrated December 2025),
not the legacy GAFF implementation.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mcp_lammps.data_handler import DataHandler
    from mcp_lammps.utils.openff_utils import openff_forcefield, OPENFF_AVAILABLE
except ImportError as e:
    logger.error(f"Failed to import mcp_lammps modules: {e}")
    logger.error("Please ensure you're running from the mcp_lammps directory")
    sys.exit(1)


class EthanolSystemTest:
    """Test class for ethanol liquid system setup."""
    
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
        
        # Test parameters
        self.test_params = {
            "name": "ethanol_liquid_test",
            "smiles": "CCO",
            "molecule_count": 100,
            "target_density": 0.789,  # g/cm³ for ethanol at 298 K
            "temperature": 298.15,  # K
            "pressure": 1.0,  # atm
            "force_field": "openff",
            "charge_method": "nagl"
        }
        
        self.results = {}
        
    def print_header(self, text: str):
        """Print a formatted header."""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)
    
    def print_section(self, text: str):
        """Print a formatted section header."""
        print(f"\n--- {text} ---")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        self.print_header("Checking Dependencies")
        
        if not OPENFF_AVAILABLE:
            logger.error("OpenFF Toolkit is not available!")
            logger.error("Please install: pip install openff-toolkit openff-interchange")
            return False
        
        print("✓ OpenFF Toolkit available")
        
        try:
            from openff.toolkit import __version__ as openff_version
            print(f"  Version: {openff_version}")
        except:
            pass
        
        try:
            from openff.interchange import __version__ as interchange_version
            print(f"✓ OpenFF Interchange available")
            print(f"  Version: {interchange_version}")
        except ImportError:
            logger.error("OpenFF Interchange is not available!")
            return False
        
        return True
    
    def test_single_molecule(self) -> bool:
        """Test creating a single ethanol molecule."""
        self.print_header("Test 1: Single Ethanol Molecule Creation")
        
        try:
            print(f"Creating ethanol molecule from SMILES: {self.test_params['smiles']}")
            
            # Create molecule from SMILES
            molecule = openff_forcefield.from_smiles(
                self.test_params['smiles'],
                "ethanol_single",
                optimize=True
            )
            
            print(f"✓ Molecule created successfully")
            print(f"  Number of atoms: {molecule.n_atoms}")
            print(f"  Number of bonds: {molecule.n_bonds}")
            
            # Assign charges
            print("\nAssigning NAGL AM1-BCC charges...")
            openff_forcefield.assign_charges(molecule)
            
            from openff.units import unit
            charges = [float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges]
            total_charge = sum(charges)
            
            print(f"✓ Charges assigned successfully")
            print(f"  Total charge: {total_charge:.6f} e")
            print(f"  Charge range: [{min(charges):.4f}, {max(charges):.4f}] e")
            
            # Store results
            self.results['single_molecule'] = {
                'n_atoms': molecule.n_atoms,
                'n_bonds': molecule.n_bonds,
                'charges': charges,
                'total_charge': total_charge
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create single molecule: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_liquid_box_creation(self) -> bool:
        """Test creating a liquid box with multiple ethanol molecules."""
        self.print_header("Test 2: Liquid Box Creation")
        
        try:
            molecules = [{
                "smiles": self.test_params['smiles'],
                "count": self.test_params['molecule_count'],
                "name": "ethanol"
            }]
            
            print(f"Creating liquid box with {self.test_params['molecule_count']} ethanol molecules")
            print(f"Target density: {self.test_params['target_density']} g/cm³")
            
            # Create liquid box
            data_file = self.data_handler.create_liquid_box_file(
                molecules=molecules,
                target_density=self.test_params['target_density'],
                box_type="cubic"
            )
            
            print(f"\n✓ Liquid box created successfully")
            print(f"  Data file: {data_file}")
            
            # Check if files exist
            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}")
                return False
            
            # Look for input script
            script_file = data_file.with_suffix('.in')
            if script_file.exists():
                print(f"  Input script: {script_file}")
            
            # Look for metadata
            metadata_file = data_file.with_suffix('.json')
            if metadata_file.exists():
                print(f"  Metadata: {metadata_file}")
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.results['liquid_box'] = {
                    'data_file': str(data_file),
                    'script_file': str(script_file) if script_file.exists() else None,
                    'metadata_file': str(metadata_file),
                    'metadata': metadata
                }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create liquid box: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_data_file(self) -> bool:
        """Validate the generated LAMMPS data file."""
        self.print_header("Test 3: Data File Validation")
        
        if 'liquid_box' not in self.results:
            logger.error("No liquid box data to validate")
            return False
        
        data_file = Path(self.results['liquid_box']['data_file'])
        
        try:
            print(f"Validating LAMMPS data file: {data_file.name}")
            
            with open(data_file, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = ['Atoms', 'Bonds', 'Angles', 'Dihedrals']
            found_sections = []
            
            for section in required_sections:
                if section in content:
                    found_sections.append(section)
                    print(f"  ✓ Found {section} section")
                else:
                    print(f"  ✗ Missing {section} section")
            
            # Extract statistics from header
            lines = content.split('\n')
            stats = {}
            
            for line in lines[:50]:  # Check first 50 lines for header info
                if 'atoms' in line and 'atom types' not in line:
                    stats['n_atoms'] = int(line.split()[0])
                elif 'bonds' in line and 'bond types' not in line:
                    stats['n_bonds'] = int(line.split()[0])
                elif 'angles' in line and 'angle types' not in line:
                    stats['n_angles'] = int(line.split()[0])
                elif 'dihedrals' in line and 'dihedral types' not in line:
                    stats['n_dihedrals'] = int(line.split()[0])
                elif 'atom types' in line:
                    stats['n_atom_types'] = int(line.split()[0])
                elif 'bond types' in line:
                    stats['n_bond_types'] = int(line.split()[0])
            
            print("\nSystem Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Validate charges
            print("\nValidating charges...")
            atom_charges = self._extract_atom_charges(content)
            
            if atom_charges:
                total_charge = sum(atom_charges)
                print(f"  Total system charge: {total_charge:.6f} e")
                print(f"  Number of atoms with charges: {len(atom_charges)}")
                print(f"  Charge range: [{min(atom_charges):.4f}, {max(atom_charges):.4f}] e")
                
                if abs(total_charge) < 1e-4:
                    print("  ✓ System is electrically neutral")
                else:
                    print(f"  ⚠ System has net charge: {total_charge:.6f} e")
            
            self.results['validation'] = {
                'found_sections': found_sections,
                'statistics': stats,
                'total_charge': total_charge if atom_charges else None,
                'n_charged_atoms': len(atom_charges) if atom_charges else 0
            }
            
            return len(found_sections) >= 3  # At least Atoms, Bonds, Angles
            
        except Exception as e:
            logger.error(f"Failed to validate data file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_atom_charges(self, content: str) -> List[float]:
        """Extract atom charges from LAMMPS data file."""
        charges = []
        in_atoms_section = False
        
        for line in content.split('\n'):
            if line.strip() == 'Atoms':
                in_atoms_section = True
                continue
            elif in_atoms_section and line.strip() and not line.startswith('#'):
                if line.strip().startswith('Velocities') or line.strip().startswith('Bonds'):
                    break
                
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        # LAMMPS full atom style: atom-ID mol-ID atom-type charge x y z
                        charge = float(parts[3])
                        charges.append(charge)
                    except (ValueError, IndexError):
                        continue
        
        return charges
    
    def display_system_info(self):
        """Display comprehensive system information."""
        self.print_header("System Information Summary")
        
        if 'liquid_box' not in self.results or 'metadata' not in self.results['liquid_box']:
            logger.warning("No metadata available to display")
            return
        
        metadata = self.results['liquid_box']['metadata']
        
        print("\nForce Field Information:")
        print(f"  Force field: {metadata.get('force_field', 'N/A')}")
        print(f"  Charge method: {metadata.get('charge_method', 'N/A')}")
        
        if 'system_info' in metadata:
            sys_info = metadata['system_info']
            print("\nSystem Composition:")
            print(f"  Total molecules: {sys_info.get('n_molecules', 'N/A')}")
            print(f"  Total atoms: {sys_info.get('n_atoms', 'N/A')}")
            print(f"  Total bonds: {sys_info.get('n_bonds', 'N/A')}")
            
            if 'box_vectors' in sys_info:
                box = sys_info['box_vectors']
                print(f"\nBox Dimensions:")
                print(f"  Box size: {box}")
        
        if 'charge_information' in metadata:
            charge_info = metadata['charge_information']
            print("\nCharge Information:")
            print(f"  Total system charge: {charge_info.get('total_system_charge', 'N/A'):.6f} e")
            print(f"  Charge method: {charge_info.get('charge_method', 'N/A')}")
            
            if 'molecule_charges' in charge_info:
                print("\n  Per-molecule charges:")
                for mol_charge in charge_info['molecule_charges']:
                    print(f"    {mol_charge.get('name', 'Unknown')}: {mol_charge.get('charge_per_molecule', 0):.6f} e")
        
        if 'validation' in self.results:
            val = self.results['validation']
            print("\nValidation Results:")
            print(f"  Sections found: {', '.join(val['found_sections'])}")
            if val.get('total_charge') is not None:
                print(f"  System neutrality: {'✓ PASS' if abs(val['total_charge']) < 1e-4 else '✗ FAIL'}")
    
    def display_file_summary(self):
        """Display summary of generated files."""
        self.print_header("Generated Files Summary")
        
        if 'liquid_box' not in self.results:
            logger.warning("No files generated")
            return
        
        files = []
        
        data_file = Path(self.results['liquid_box']['data_file'])
        if data_file.exists():
            size = data_file.stat().st_size / 1024  # KB
            files.append(('LAMMPS Data File', data_file, size))
        
        if self.results['liquid_box'].get('script_file'):
            script_file = Path(self.results['liquid_box']['script_file'])
            if script_file.exists():
                size = script_file.stat().st_size / 1024
                files.append(('LAMMPS Input Script', script_file, size))
        
        if self.results['liquid_box'].get('metadata_file'):
            metadata_file = Path(self.results['liquid_box']['metadata_file'])
            if metadata_file.exists():
                size = metadata_file.stat().st_size / 1024
                files.append(('Metadata File', metadata_file, size))
        
        print("\nGenerated Files:")
        for file_type, file_path, size in files:
            print(f"\n  {file_type}:")
            print(f"    Path: {file_path}")
            print(f"    Size: {size:.2f} KB")
        
        print(f"\nAll files are located in: {self.output_dir}")
    
    def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        self.print_header("MCP LAMMPS Ethanol System Setup Test")
        
        print("\nTest Configuration:")
        for key, value in self.test_params.items():
            print(f"  {key}: {value}")
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Run tests
        tests = [
            ("Single Molecule Creation", self.test_single_molecule),
            ("Liquid Box Creation", self.test_liquid_box_creation),
            ("Data File Validation", self.validate_data_file),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                logger.error(f"Test '{test_name}' failed with exception: {e}")
                results.append((test_name, False))
        
        # Display information
        self.display_system_info()
        self.display_file_summary()
        
        # Print summary
        self.print_header("Test Results Summary")
        
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
            print("  Ethanol system setup is working correctly with OpenFF!")
        else:
            print("  ✗ SOME TESTS FAILED")
            print("  Please check the error messages above.")
        print("=" * 80 + "\n")
        
        return all_passed


def main():
    """Main entry point."""
    # Create test instance
    test = EthanolSystemTest()
    
    # Run all tests
    success = test.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

