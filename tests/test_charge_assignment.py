"""
Integration tests for charge assignment in the force field module.

This module tests the complete workflow from SMILES to charged molecules:
- SMILES to OpenFF Molecule conversion
- NAGL charge assignment
- Charge normalization and validation
- ForceFieldUtils integration
"""

import pytest
import numpy as np
from pathlib import Path

# Check if OpenFF is available
try:
    from src.mcp_lammps.utils.openff_utils import openff_utils, OpenFFUtils, OPENFF_AVAILABLE
    from src.mcp_lammps.utils.forcefield_utils import forcefield_utils, ForceFieldUtils
    from openff.units import unit
    OPENFF_TESTS_ENABLED = OPENFF_AVAILABLE
except ImportError:
    OPENFF_TESTS_ENABLED = False

pytestmark = pytest.mark.skipif(not OPENFF_TESTS_ENABLED, reason="OpenFF Toolkit not available")


@pytest.mark.skipif(not OPENFF_TESTS_ENABLED, reason="OpenFF not available")
class TestChargeAssignmentWorkflow:
    """Integration tests for charge assignment workflow."""
    
    def test_water_charge_assignment(self):
        """Test charge assignment for water molecule (H2O)."""
        # SMILES for water
        smiles = "O"
        name = "water"
        expected_atoms = 3  # 1 O + 2 H
        
        # Step 1: Convert SMILES to molecule
        molecule = openff_utils.smiles_to_openff_molecule(smiles, name, optimize=True)
        
        assert molecule is not None, "Molecule creation failed"
        assert molecule.name == name
        assert molecule.n_atoms == expected_atoms, f"Expected {expected_atoms} atoms, got {molecule.n_atoms}"
        assert len(molecule.conformers) > 0, "No conformers generated"
        
        # Step 2: Assign charges using NAGL
        openff_utils.assign_charges_nagl(molecule)
        
        # Step 3: Validate charges
        assert molecule.partial_charges is not None, "Charges were not assigned"
        assert len(molecule.partial_charges) == expected_atoms, \
            f"Expected {expected_atoms} charges, got {len(molecule.partial_charges)}"
        
        # Step 4: Check charge neutrality
        total_charge = sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges])
        assert abs(total_charge) < 1e-6, \
            f"Charges should sum to zero, got {total_charge:.8f}"
        
        # Step 5: Check charge values are reasonable
        for charge in molecule.partial_charges:
            charge_val = float(charge.m_as(unit.elementary_charge))
            assert -2.0 <= charge_val <= 2.0, \
                f"Charge {charge_val} is outside reasonable range [-2.0, 2.0]"
    
    def test_methane_charge_assignment(self):
        """Test charge assignment for methane molecule (CH4)."""
        # SMILES for methane
        smiles = "C"
        name = "methane"
        expected_atoms = 5  # 1 C + 4 H
        
        # Step 1: Convert SMILES to molecule with geometry optimization
        molecule = openff_utils.smiles_to_openff_molecule(smiles, name, optimize=True)
        
        assert molecule is not None, "Molecule creation failed"
        assert molecule.name == name
        assert molecule.n_atoms == expected_atoms, f"Expected {expected_atoms} atoms, got {molecule.n_atoms}"
        assert len(molecule.conformers) > 0, "No conformers generated"
        
        # Step 2: Assign charges using NAGL
        openff_utils.assign_charges_nagl(molecule)
        
        # Step 3: Validate charges
        assert molecule.partial_charges is not None, "Charges were not assigned"
        assert len(molecule.partial_charges) == expected_atoms, \
            f"Expected {expected_atoms} charges, got {len(molecule.partial_charges)}"
        
        # Step 4: Check charge neutrality
        total_charge = sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges])
        assert abs(total_charge) < 1e-6, \
            f"Charges should sum to zero, got {total_charge:.8f}"
        
        # Step 5: Check charge values are reasonable
        for charge in molecule.partial_charges:
            charge_val = float(charge.m_as(unit.elementary_charge))
            assert -2.0 <= charge_val <= 2.0, \
                f"Charge {charge_val} is outside reasonable range [-2.0, 2.0]"
        
        # Step 6: Verify methane has small charges (symmetric molecule)
        charge_magnitudes = [abs(float(c.m_as(unit.elementary_charge))) for c in molecule.partial_charges]
        max_charge = max(charge_magnitudes)
        assert max_charge < 0.5, \
            f"Methane charges should be small due to symmetry, max charge: {max_charge}"
    
    def test_ethanol_charge_assignment(self):
        """Test charge assignment for ethanol molecule (C2H5OH)."""
        # SMILES for ethanol
        smiles = "CCO"
        name = "ethanol"
        expected_atoms = 9  # 2 C + 1 O + 6 H
        
        # Step 1: Convert SMILES to molecule
        molecule = openff_utils.smiles_to_openff_molecule(smiles, name, optimize=True)
        
        assert molecule is not None, "Molecule creation failed"
        assert molecule.name == name
        assert molecule.n_atoms == expected_atoms, f"Expected {expected_atoms} atoms, got {molecule.n_atoms}"
        assert len(molecule.conformers) > 0, "No conformers generated"
        
        # Step 2: Assign charges using NAGL with normalization
        openff_utils.assign_charges_nagl(molecule)
        
        # Step 3: Validate charges
        assert molecule.partial_charges is not None, "Charges were not assigned"
        assert len(molecule.partial_charges) == expected_atoms, \
            f"Expected {expected_atoms} charges, got {len(molecule.partial_charges)}"
        
        # Step 4: Check charge neutrality (should be exact after normalization)
        total_charge = sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges])
        assert abs(total_charge) < 1e-6, \
            f"Charges should sum to zero after normalization, got {total_charge:.8f}"
        
        # Step 5: Check charge values are reasonable
        for charge in molecule.partial_charges:
            charge_val = float(charge.m_as(unit.elementary_charge))
            assert -2.0 <= charge_val <= 2.0, \
                f"Charge {charge_val} is outside reasonable range [-2.0, 2.0]"
        
        # Step 6: Check that oxygen has significant negative charge
        # (This is a chemical property test - oxygen should be electronegative)
        charges_list = [float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges]
        min_charge = min(charges_list)
        assert min_charge < -0.3, \
            f"Oxygen in ethanol should have significant negative charge, got min: {min_charge}"
    
    def test_charge_neutrality(self):
        """Test charge normalization and neutrality enforcement."""
        # Create a test molecule
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        
        # Create test charges that don't sum to zero
        test_charges = np.array([0.123456789, -0.234567890, 0.111111111, 
                                 0.050000000, -0.030000000, 0.020000000,
                                 -0.015000000, -0.010000000, -0.005000000]) * unit.elementary_charge
        
        # Apply normalization
        normalized = openff_utils.round_and_renormalize_charges(test_charges, decimals=6)
        
        # Verify properties
        assert len(normalized) == len(test_charges), "Charge count changed during normalization"
        
        # Check rounding to 6 decimals
        for charge in normalized:
            charge_val = float(charge.m_as(unit.elementary_charge))
            rounded_val = round(charge_val, 6)
            assert abs(charge_val - rounded_val) < 1e-7, \
                f"Charge {charge_val} not properly rounded to 6 decimals"
        
        # Check sum to zero (use slightly relaxed tolerance for floating point)
        total = sum([float(c.m_as(unit.elementary_charge)) for c in normalized])
        assert abs(total) < 2e-6, \
            f"Normalized charges should sum to zero, got {total:.8f}"
    
    def test_forcefield_utils_integration(self):
        """Test ForceFieldUtils integration with charge assignment."""
        # Test 1: assign_charges method
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        
        # Use ForceFieldUtils to assign charges
        forcefield_utils.assign_charges(molecule)
        
        assert molecule.partial_charges is not None, "ForceFieldUtils.assign_charges failed"
        
        total_charge = sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges])
        assert abs(total_charge) < 1e-6, \
            f"Charges from ForceFieldUtils should sum to zero, got {total_charge:.8f}"
        
        # Test 2: validate_molecule method
        is_valid, issues = forcefield_utils.validate_molecule(molecule)
        
        assert isinstance(is_valid, bool), "validate_molecule should return boolean"
        assert isinstance(issues, list), "validate_molecule should return list of issues"
        assert is_valid, f"Molecule validation failed: {issues}"
        assert len(issues) == 0, f"Validation issues found: {issues}"
        
        # Test 3: validate molecule without charges
        molecule_no_charges = openff_utils.smiles_to_openff_molecule("O", "water")
        is_valid, issues = forcefield_utils.validate_molecule(molecule_no_charges)
        
        assert not is_valid, "Molecule without charges should fail validation"
        assert len(issues) > 0, "Should report issues for molecule without charges"
        assert any("charge" in issue.lower() for issue in issues), \
            "Should report missing charges"
    
    def test_multiple_molecules(self):
        """Test charge assignment for multiple molecules in sequence."""
        # Define test molecules
        test_molecules = [
            {"smiles": "O", "name": "water", "expected_atoms": 3},
            {"smiles": "C", "name": "methane", "expected_atoms": 5},
            {"smiles": "CCO", "name": "ethanol", "expected_atoms": 9}
        ]
        
        charged_molecules = []
        
        # Process each molecule
        for mol_spec in test_molecules:
            # Create molecule
            molecule = openff_utils.smiles_to_openff_molecule(
                mol_spec["smiles"], 
                mol_spec["name"],
                optimize=True
            )
            
            assert molecule.n_atoms == mol_spec["expected_atoms"], \
                f"{mol_spec['name']}: Expected {mol_spec['expected_atoms']} atoms, got {molecule.n_atoms}"
            
            # Assign charges
            openff_utils.assign_charges_nagl(molecule)
            
            # Validate charges
            assert molecule.partial_charges is not None, \
                f"{mol_spec['name']}: Charges not assigned"
            
            total_charge = sum([float(c.m_as(unit.elementary_charge)) for c in molecule.partial_charges])
            assert abs(total_charge) < 1e-6, \
                f"{mol_spec['name']}: Charges should sum to zero, got {total_charge:.8f}"
            
            charged_molecules.append(molecule)
        
        # Verify all molecules are independent
        assert len(charged_molecules) == len(test_molecules), \
            "Not all molecules were processed"
        
        # Verify each molecule has correct properties
        for i, molecule in enumerate(charged_molecules):
            assert molecule.name == test_molecules[i]["name"], \
                f"Molecule {i} has wrong name"
            assert molecule.n_atoms == test_molecules[i]["expected_atoms"], \
                f"Molecule {i} has wrong atom count"
            assert molecule.partial_charges is not None, \
                f"Molecule {i} lost charges"
    
    def test_charge_assignment_with_create_system(self):
        """Test charge assignment through create_system_from_smiles workflow."""
        # Use the high-level method from ForceFieldUtils
        system = forcefield_utils.create_system_from_smiles("CCO", "ethanol", optimize=True)
        
        assert system is not None, "System creation failed"
        assert system.topology is not None, "System has no topology"
        assert system.topology.n_atoms == 9, "System has wrong number of atoms"
        assert system.topology.n_molecules == 1, "System should have 1 molecule"
        
        # Verify the system can be used (has proper force field parameters)
        info = openff_utils.get_system_info(system)
        assert "n_atoms" in info, "System info missing atom count"
        assert info["n_atoms"] == 9, "System info has wrong atom count"
        assert "charge_model" in info, "System info missing charge model"
    
    def test_charge_values_are_finite(self):
        """Test that all assigned charges are finite (not NaN or Inf)."""
        molecule = openff_utils.smiles_to_openff_molecule("CCO", "ethanol")
        openff_utils.assign_charges_nagl(molecule)
        
        for i, charge in enumerate(molecule.partial_charges):
            charge_val = float(charge.m_as(unit.elementary_charge))
            assert np.isfinite(charge_val), \
                f"Charge {i} is not finite: {charge_val}"
            assert not np.isnan(charge_val), \
                f"Charge {i} is NaN"
            assert not np.isinf(charge_val), \
                f"Charge {i} is infinite"
    
    def test_charge_consistency_across_runs(self):
        """Test that charge assignment is consistent across multiple runs."""
        smiles = "CCO"
        name = "ethanol"
        
        # First run
        molecule1 = openff_utils.smiles_to_openff_molecule(smiles, name)
        openff_utils.assign_charges_nagl(molecule1)
        charges1 = [float(c.m_as(unit.elementary_charge)) for c in molecule1.partial_charges]
        
        # Second run
        molecule2 = openff_utils.smiles_to_openff_molecule(smiles, name)
        openff_utils.assign_charges_nagl(molecule2)
        charges2 = [float(c.m_as(unit.elementary_charge)) for c in molecule2.partial_charges]
        
        # Charges should be identical (or very close due to conformer generation)
        assert len(charges1) == len(charges2), "Different number of charges"
        
        # Check that charges are similar (allowing for small conformer differences)
        for i, (c1, c2) in enumerate(zip(charges1, charges2)):
            assert abs(c1 - c2) < 0.01, \
                f"Charge {i} differs significantly: {c1} vs {c2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

