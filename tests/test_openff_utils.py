"""
Tests for OpenFF utilities and integration.
"""

import pytest
import numpy as np
from pathlib import Path

# Check if OpenFF is available
try:
    from src.mcp_lammps.utils.openff_utils import openff_utils, OpenFFUtils, OPENFF_AVAILABLE
    from openff.units import unit
    OPENFF_TESTS_ENABLED = OPENFF_AVAILABLE
except ImportError:
    OPENFF_TESTS_ENABLED = False

pytestmark = pytest.mark.skipif(not OPENFF_TESTS_ENABLED, reason="OpenFF Toolkit not available")


class TestOpenFFUtils:
    """Test OpenFF utilities functionality."""
    
    def test_initialization(self):
        """Test OpenFFUtils initialization."""
        utils = OpenFFUtils()
        assert utils.force_field is not None
        assert utils.charge_model == 'openff-gnn-am1bcc-1.0.0.pt'
        assert 'sage' in utils.ff_name.lower() or 'openff' in utils.ff_name.lower()
    
    def test_smiles_to_molecule_ethanol(self):
        """Test SMILES to OpenFF molecule conversion for ethanol."""
        smiles = "CCO"
        name = "ethanol"
        
        molecule = openff_utils.smiles_to_openff_molecule(smiles, name, optimize=True)
        
        assert molecule is not None
        assert molecule.name == name
        assert molecule.n_atoms == 9  # 2C + 1O + 6H
        assert molecule.n_bonds == 8
        assert len(molecule.conformers) > 0
    
    def test_smiles_to_molecule_water(self):
        """Test SMILES to OpenFF molecule conversion for water."""
        smiles = "O"
        name = "water"
        
        molecule = openff_utils.smiles_to_openff_molecule(smiles, name)
        
        assert molecule is not None
        assert molecule.name == name
        assert molecule.n_atoms == 3  # 1O + 2H
    
    def test_charge_assignment(self):
        """Test NAGL charge assignment."""
        # Create a simple molecule
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        
        # Assign charges
        openff_utils.assign_charges_nagl(molecule)
        
        assert molecule.partial_charges is not None
        assert len(molecule.partial_charges) == molecule.n_atoms
        
        # Check charge neutrality
        total_charge = sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges])
        assert abs(total_charge) < 1e-5, f"Total charge should be near zero, got {total_charge}"
    
    def test_round_and_renormalize_charges(self):
        """Test charge rounding and renormalization."""
        # Create test charges
        test_charges = np.array([0.123456789, -0.234567890, 0.111111111]) * unit.elementary_charge
        
        # Renormalize
        normalized = openff_utils.round_and_renormalize_charges(test_charges, decimals=6)
        
        # Check rounding
        for charge in normalized:
            charge_val = float(charge.m_as(unit.elementary_charge))
            # Should be rounded to 6 decimals
            assert abs(charge_val - round(charge_val, 6)) < 1e-7
        
        # Check sum to zero
        total = sum([float(c.m_as(unit.elementary_charge)) for c in normalized])
        assert abs(total) < 1e-6
    
    def test_create_topology_single_molecule(self):
        """Test topology creation for single molecule."""
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        openff_utils.assign_charges_nagl(molecule)
        
        topology = openff_utils.create_topology([molecule])
        
        assert topology is not None
        assert topology.n_molecules == 1
        assert topology.n_atoms == molecule.n_atoms
    
    def test_create_topology_multiple_molecules(self):
        """Test topology creation for multiple molecules."""
        mol1 = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        mol2 = openff_utils.smiles_to_openff_molecule("O", "water")
        
        openff_utils.assign_charges_nagl(mol1)
        openff_utils.assign_charges_nagl(mol2)
        
        topology = openff_utils.create_topology([mol1, mol1, mol2])
        
        assert topology is not None
        assert topology.n_molecules == 3
        assert topology.n_atoms == 2 * mol1.n_atoms + mol2.n_atoms
    
    def test_create_interchange_system(self):
        """Test Interchange system creation."""
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        openff_utils.assign_charges_nagl(molecule)
        
        topology = openff_utils.create_topology([molecule])
        interchange = openff_utils.create_interchange_system(topology, [molecule])
        
        assert interchange is not None
        assert interchange.topology.n_atoms == molecule.n_atoms
    
    def test_calculate_box_size(self):
        """Test box size calculation."""
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        
        box_size = openff_utils.calculate_box_size([molecule], [100], target_density=0.789)
        
        assert box_size > 0
        assert box_size < 1000  # Reasonable range for 100 molecules
    
    def test_generate_random_positions(self):
        """Test random position generation."""
        molecule = openff_utils.smiles_to_openff_molecule("O", "water")
        
        positions = openff_utils.generate_random_positions([molecule], [10], box_size=30.0)
        
        assert positions is not None
        assert len(positions) == 10 * molecule.n_atoms
        assert positions.shape[1] == 3  # x, y, z coordinates
    
    def test_interchange_to_lammps(self, tmp_path):
        """Test Interchange export to LAMMPS files."""
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        openff_utils.assign_charges_nagl(molecule)
        
        topology = openff_utils.create_topology([molecule])
        interchange = openff_utils.create_interchange_system(topology, [molecule])
        
        output_prefix = tmp_path / "test_ethanol"
        data_file, script_file = openff_utils.interchange_to_lammps(interchange, str(output_prefix))
        
        assert data_file.exists()
        assert script_file.exists()
        assert data_file.suffix == ".data"
        assert script_file.suffix == ".in"
        
        # Check data file has content
        with open(data_file, 'r') as f:
            content = f.read()
            assert "atoms" in content.lower()
            assert "bonds" in content.lower()
    
    def test_build_liquid_box(self, tmp_path):
        """Test complete liquid box building."""
        molecule_specs = [
            {'smiles': 'CCO', 'count': 10, 'name': 'ethanol'},
            {'smiles': 'O', 'count': 10, 'name': 'water'}
        ]
        
        output_prefix = tmp_path / "test_mixture"
        
        interchange, data_file, script_file = openff_utils.build_liquid_box(
            molecule_specs=molecule_specs,
            target_density=1.0,
            box_type='cubic',
            output_prefix=str(output_prefix)
        )
        
        assert interchange is not None
        assert data_file.exists()
        assert script_file.exists()
        assert interchange.topology.n_molecules == 20  # 10 ethanol + 10 water
    
    def test_get_system_info(self):
        """Test system information extraction."""
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        openff_utils.assign_charges_nagl(molecule)
        
        topology = openff_utils.create_topology([molecule])
        interchange = openff_utils.create_interchange_system(topology, [molecule])
        
        info = openff_utils.get_system_info(interchange)
        
        assert "n_atoms" in info
        assert "n_molecules" in info
        assert "force_field" in info
        assert "charge_model" in info
        assert info["n_atoms"] == molecule.n_atoms
        assert info["n_molecules"] == 1


class TestForceFieldUtils:
    """Test ForceFieldUtils OpenFF integration."""
    
    def test_forcefield_utils_initialization(self):
        """Test ForceFieldUtils initialization."""
        from src.mcp_lammps.utils.forcefield_utils import ForceFieldUtils
        
        ff_utils = ForceFieldUtils()
        assert ff_utils.openff_utils is not None
        assert ff_utils.ff_name is not None
    
    def test_assign_charges(self):
        """Test charge assignment through ForceFieldUtils."""
        from src.mcp_lammps.utils.forcefield_utils import forcefield_utils
        
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        forcefield_utils.assign_charges(molecule)
        
        assert molecule.partial_charges is not None
    
    def test_create_system_from_smiles(self):
        """Test complete system creation from SMILES."""
        from src.mcp_lammps.utils.forcefield_utils import forcefield_utils
        
        system = forcefield_utils.create_system_from_smiles("CCO", "ethanol")
        
        assert system is not None
        assert system.topology.n_molecules == 1
    
    def test_validate_molecule(self):
        """Test molecule validation."""
        from src.mcp_lammps.utils.forcefield_utils import forcefield_utils
        
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        openff_utils.assign_charges_nagl(molecule)
        
        is_valid, issues = forcefield_utils.validate_molecule(molecule)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_get_force_field_info(self):
        """Test force field information retrieval."""
        from src.mcp_lammps.utils.forcefield_utils import forcefield_utils
        
        info = forcefield_utils.get_force_field_info()
        
        assert "force_field" in info
        assert "type" in info
        assert info["type"] == "OpenFF"


class TestOpenFFValidation:
    """Test OpenFF validation functions."""
    
    def test_validate_force_field(self):
        """Test force field validation."""
        from src.mcp_lammps.utils.validators import validate_force_field
        
        # Valid OpenFF force fields
        assert validate_force_field("openff")
        assert validate_force_field("openff-sage")
        assert validate_force_field("openff-2.2.1.offxml")
        assert validate_force_field("openff_unconstrained-2.2.1.offxml")
        
        # Invalid force field
        assert not validate_force_field("invalid_ff")
        assert not validate_force_field("")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

