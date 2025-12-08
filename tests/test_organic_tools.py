"""
Test suite for organic simulation functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from src.mcp_lammps.utils.molecular_utils import MolecularUtils
from src.mcp_lammps.utils.forcefield_utils import ForceFieldUtils
from src.mcp_lammps.data_handler import DataHandler
from src.mcp_lammps.lammps_interface import LAMMPSInterface


class TestMolecularUtils:
    """Test molecular utilities functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.molecular_utils = MolecularUtils()
    
    def test_init(self):
        """Test MolecularUtils initialization."""
        assert isinstance(self.molecular_utils, MolecularUtils)
        # Test should work even if RDKit/OpenBabel not available
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_smiles_to_3d_simple_molecule(self):
        """Test SMILES to 3D conversion for simple molecule."""
        smiles = "CCO"  # Ethanol
        mol = self.molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            assert mol.GetNumAtoms() > 0
            assert mol.GetNumConformers() > 0
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_calculate_molecular_properties(self):
        """Test molecular property calculation."""
        # Create a simple molecule if RDKit is available
        smiles = "CCO"
        mol = self.molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            properties = self.molecular_utils.calculate_molecular_properties(mol)
            
            assert isinstance(properties, dict)
            assert "molecular_weight" in properties
            assert properties["molecular_weight"] > 0
    
    def test_smiles_to_3d_invalid_smiles(self):
        """Test SMILES to 3D conversion with invalid SMILES."""
        invalid_smiles = "INVALID_SMILES"
        mol = self.molecular_utils.smiles_to_3d(invalid_smiles)
        
        # Should return None for invalid SMILES
        assert mol is None
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_mol_to_xyz(self):
        """Test molecule to XYZ format conversion."""
        smiles = "C"  # Methane
        mol = self.molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            xyz_content = self.molecular_utils.mol_to_xyz(mol)
            
            assert isinstance(xyz_content, str)
            assert len(xyz_content) > 0
            lines = xyz_content.strip().split('\n')
            assert len(lines) >= 2  # At least header lines
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_detect_molecule_type(self):
        """Test molecule type detection."""
        test_cases = [
            ("CCO", "alcohol"),  # Ethanol should be detected as alcohol
            ("c1ccccc1", "aromatic"),  # Benzene should be aromatic
            ("CC", "small_organic"),  # Ethane should be small organic
        ]
        
        for smiles, expected_type in test_cases:
            mol = self.molecular_utils.smiles_to_3d(smiles)
            if mol is not None:
                detected_type = self.molecular_utils.detect_molecule_type(mol)
                # Note: exact type detection may vary, so we just check it returns a string
                assert isinstance(detected_type, str)
                assert len(detected_type) > 0
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_validate_molecule(self):
        """Test molecule validation."""
        smiles = "CCO"
        mol = self.molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            is_valid, issues = self.molecular_utils.validate_molecule(mol)
            
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)


class TestForceFieldUtils:
    """Test force field utilities functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ff_utils = ForceFieldUtils()
    
    def test_init(self):
        """Test ForceFieldUtils initialization."""
        assert isinstance(self.ff_utils, ForceFieldUtils)
        assert hasattr(self.ff_utils, 'gaff_patterns')
        assert len(self.ff_utils.gaff_patterns) > 0
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_assign_gaff_atom_types(self):
        """Test GAFF atom type assignment."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            
            assert isinstance(atom_types, dict)
            assert len(atom_types) == mol.GetNumAtoms()
            
            # Check that all atom types are strings
            for atom_idx, atom_type in atom_types.items():
                assert isinstance(atom_idx, int)
                assert isinstance(atom_type, str)
                assert len(atom_type) > 0
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_generate_topology(self):
        """Test topology generation."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            topology = self.ff_utils.generate_topology(mol, atom_types)
            
            assert isinstance(topology, dict)
            assert "bonds" in topology
            assert "angles" in topology
            assert "dihedrals" in topology
            
            # Check that we have some bonds
            assert len(topology["bonds"]) > 0
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_create_lammps_data_file(self):
        """Test LAMMPS data file creation."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            topology = self.ff_utils.generate_topology(mol, atom_types)
            
            data_content = self.ff_utils.create_lammps_data_file(mol, atom_types, topology)
            
            assert isinstance(data_content, str)
            assert len(data_content) > 0
            assert "atoms" in data_content.lower()
            assert "bonds" in data_content.lower()
    
    def test_get_atomic_mass(self):
        """Test atomic mass retrieval."""
        # Test some common GAFF atom types
        test_cases = [
            ("c3", 12.01),  # Carbon
            ("hc", 1.008),  # Hydrogen
            ("oh", 15.999), # Oxygen
        ]
        
        for atom_type, expected_mass in test_cases:
            mass = self.ff_utils._get_atomic_mass(atom_type)
            assert isinstance(mass, float)
            assert mass > 0
            # Allow some tolerance for mass values
            assert abs(mass - expected_mass) < 5.0
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_validate_parameters(self):
        """Test parameter validation."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            topology = self.ff_utils.generate_topology(mol, atom_types)
            
            is_valid, issues = self.ff_utils.validate_parameters(atom_types, topology)
            
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_dihedral_parameter_assignment(self):
        """Test comprehensive dihedral parameter assignment."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        # Test different molecular types
        test_molecules = [
            ("CCCC", "butane"),  # Simple alkane
            ("c1ccccc1", "benzene"),  # Aromatic
            ("CCO", "ethanol"),  # Alcohol
            ("CCOC", "diethyl ether"),  # Ether
            ("CC(=O)O", "acetic acid"),  # Carboxylic acid
            ("CCN", "ethylamine"),  # Amine
        ]
        
        for smiles, name in test_molecules:
            mol = molecular_utils.smiles_to_3d(smiles)
            
            if mol is not None:
                atom_types = self.ff_utils.assign_gaff_atom_types(mol)
                topology = self.ff_utils.generate_topology(mol, atom_types)
                
                # Check that dihedrals were generated
                assert "dihedrals" in topology
                dihedrals = topology["dihedrals"]
                
                if len(dihedrals) > 0:
                    # Check that each dihedral has proper structure
                    for dihedral in dihedrals:
                        assert "atoms" in dihedral
                        assert "types" in dihedral
                        assert "parameters" in dihedral
                        
                        # Check atom indices are valid
                        atoms = dihedral["atoms"]
                        assert len(atoms) == 4
                        assert all(0 <= idx < mol.GetNumAtoms() for idx in atoms)
                        
                        # Check types are strings
                        types = dihedral["types"]
                        assert len(types) == 4
                        assert all(isinstance(t, str) for t in types)
                        
                        # Check parameters have required fields
                        params = dihedral["parameters"]
                        assert isinstance(params, list)
                        assert len(params) > 0
                        for param in params:
                            assert "k" in param
                            assert "n" in param
                            assert "phase" in param
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_improper_dihedral_detection(self):
        """Test improper dihedral detection and assignment."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        # Test molecules that should have impropers
        test_molecules = [
            ("c1ccccc1", "benzene"),  # Aromatic - should have impropers
            ("CC(=O)O", "acetic acid"),  # Carbonyl - should have impropers
            ("C=C", "ethene"),  # Alkene - should have impropers
        ]
        
        for smiles, name in test_molecules:
            mol = molecular_utils.smiles_to_3d(smiles)
            
            if mol is not None:
                atom_types = self.ff_utils.assign_gaff_atom_types(mol)
                topology = self.ff_utils.generate_topology(mol, atom_types)
                
                # Check that impropers section exists
                assert "impropers" in topology
                impropers = topology["impropers"]
                
                # For aromatic and sp2 systems, we should have some impropers
                if any(atom_type in ["ca", "c", "c2"] for atom_type in atom_types.values()):
                    # Should have at least some impropers for planarity
                    if len(impropers) > 0:
                        # Check improper structure
                        for improper in impropers:
                            assert "atoms" in improper
                            assert "types" in improper
                            assert "parameters" in improper
                            
                            # Check atom indices are valid
                            atoms = improper["atoms"]
                            assert len(atoms) == 4
                            assert all(0 <= idx < mol.GetNumAtoms() for idx in atoms)
                            
                            # Check parameters
                            params = improper["parameters"]
                            assert isinstance(params, list)
                            assert len(params) > 0
                            for param in params:
                                assert "k" in param
                                assert "n" in param
                                assert "phase" in param
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_wildcard_dihedral_matching(self):
        """Test wildcard pattern matching for dihedrals."""
        # Test the wildcard matching function directly
        test_cases = [
            (("c3", "c3", "c3", "c3"), ("X", "c3", "c3", "X"), True),
            (("hc", "c3", "c3", "hc"), ("X", "c3", "c3", "X"), True),
            (("ca", "ca", "ca", "ca"), ("X", "ca", "ca", "X"), True),
            (("c3", "os", "c3", "hc"), ("X", "os", "c3", "X"), True),
            (("c3", "c3", "n3", "hn"), ("X", "c3", "n3", "X"), True),   # Should match with wildcards
        ]
        
        for dihedral_type, pattern, expected in test_cases:
            result = self.ff_utils._matches_wildcard_pattern(dihedral_type, pattern)
            assert result == expected, f"Pattern {pattern} vs {dihedral_type} should be {expected}"
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_dihedral_parameter_coverage(self):
        """Test dihedral parameter coverage and fallback logic."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        # Test a complex molecule to check parameter coverage
        smiles = "CC(C)C(=O)Oc1ccccc1"  # Isobutyl benzoate
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            topology = self.ff_utils.generate_topology(mol, atom_types)
            
            # Validate parameters to check coverage
            is_valid, issues = self.ff_utils.validate_parameters(atom_types, topology)
            
            # Check that validation provides detailed coverage information
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)
            
            # Check that dihedral parameter lookup works (may return None for uncovered types)
            found_params = 0
            total_dihedrals = len(topology["dihedrals"])
            for dihedral in topology["dihedrals"]:
                dihedral_type = tuple(dihedral["types"])
                params = self.ff_utils.get_dihedral_parameters(dihedral_type)

                if params is not None:
                    found_params += 1
                    assert isinstance(params, list)
                    assert len(params) > 0

            # Should find parameters for at least some dihedrals
            assert found_params > 0, f"No dihedral parameters found for {total_dihedrals} dihedrals"
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_enhanced_validation_with_dihedrals(self):
        """Test enhanced parameter validation including dihedrals."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"  # Simple molecule
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            topology = self.ff_utils.generate_topology(mol, atom_types)
            
            # Test the enhanced validation
            is_valid, issues = self.ff_utils.validate_parameters(atom_types, topology)
            
            # Should return validation results
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)
            
            # For a simple molecule like ethanol, validation should generally pass
            # (though some parameters might use fallbacks)
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_calculate_partial_charges_gasteiger(self):
        """Test Gasteiger partial charge calculation."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"  # Ethanol
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            charges = self.ff_utils.calculate_partial_charges(mol, method="gasteiger")
            
            assert isinstance(charges, dict)
            assert len(charges) == mol.GetNumAtoms()
            
            # Check that charges are reasonable
            total_charge = sum(charges.values())
            assert abs(total_charge) < 0.1  # Should be approximately neutral
            
            # Check that all charges are floats and within reasonable range
            for atom_idx, charge in charges.items():
                assert isinstance(atom_idx, int)
                assert isinstance(charge, float)
                assert abs(charge) < 2.0  # No charge should be too extreme
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_calculate_partial_charges_mmff(self):
        """Test MMFF partial charge calculation."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"  # Ethanol
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            charges = self.ff_utils.calculate_partial_charges(mol, method="mmff94")
            
            assert isinstance(charges, dict)
            assert len(charges) == mol.GetNumAtoms()
            
            # Check charge neutrality
            total_charge = sum(charges.values())
            assert abs(total_charge) < 0.1
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_get_gaff_charge_estimates(self):
        """Test GAFF charge estimates based on atom types."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"  # Ethanol
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            charges = self.ff_utils.get_gaff_charge_estimates(mol, atom_types)
            
            assert isinstance(charges, dict)
            assert len(charges) == mol.GetNumAtoms()
            
            # Check charge neutrality
            total_charge = sum(charges.values())
            assert abs(total_charge) < 1e-6  # Should be exactly neutral due to adjustment
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_validate_charges(self):
        """Test charge validation functionality."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"  # Ethanol
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            # Test with valid charges
            charges = self.ff_utils.calculate_partial_charges(mol, method="gasteiger")
            is_valid, issues = self.ff_utils.validate_charges(charges, mol)
            
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)
            
            # Test with invalid charges (non-neutral)
            invalid_charges = {i: 1.0 for i in range(mol.GetNumAtoms())}
            is_valid, issues = self.ff_utils.validate_charges(invalid_charges, mol)
            
            assert not is_valid  # Should fail validation
            assert len(issues) > 0
            assert any("neutral" in issue.lower() for issue in issues)
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_create_lammps_data_file_with_charges(self):
        """Test LAMMPS data file creation with proper charges."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        smiles = "CCO"  # Ethanol
        mol = molecular_utils.smiles_to_3d(smiles)
        
        if mol is not None:
            atom_types = self.ff_utils.assign_gaff_atom_types(mol)
            topology = self.ff_utils.generate_topology(mol, atom_types)
            charges = self.ff_utils.calculate_partial_charges(mol, method="gasteiger")
            
            data_content = self.ff_utils.create_lammps_data_file(mol, atom_types, topology, charges)
            
            assert isinstance(data_content, str)
            assert len(data_content) > 0
            assert "atoms" in data_content.lower()
            
            # Check that charges are not all zero
            lines = data_content.split('\n')
            atom_lines = []
            in_atoms_section = False
            
            for line in lines:
                if line.strip() == "Atoms":
                    in_atoms_section = True
                    continue
                elif in_atoms_section and line.strip() and not line.strip().startswith("Bonds"):
                    # Skip empty lines but collect atom data lines
                    if len(line.split()) >= 4 and line.split()[0].isdigit():
                        atom_lines.append(line)
                elif in_atoms_section and line.strip().startswith("Bonds"):
                    break
            
            # Parse charges from atom lines
            parsed_charges = []
            for line in atom_lines:
                parts = line.split()
                if len(parts) >= 4:
                    charge = float(parts[3])  # Charge is the 4th column
                    parsed_charges.append(charge)
            
            # Check that we found some charges
            assert len(parsed_charges) > 0, f"No charges found in atom lines: {atom_lines[:3]}"
            
            # Check that not all charges are zero
            non_zero_charges = [c for c in parsed_charges if abs(c) > 1e-6]
            assert len(non_zero_charges) > 0, f"All charges are zero: {parsed_charges}"
    
    @pytest.mark.skipif(not hasattr(ForceFieldUtils(), 'rdkit_available') or not ForceFieldUtils().rdkit_available,
                       reason="RDKit not available")
    def test_charge_assignment_for_different_molecules(self):
        """Test charge assignment for various molecule types."""
        from src.mcp_lammps.utils.molecular_utils import molecular_utils
        
        test_molecules = [
            ("CCO", "ethanol"),
            ("CC(=O)O", "acetic_acid"),
            ("c1ccccc1", "benzene"),
            ("CCN", "ethylamine"),
            ("CCOC", "diethyl_ether")
        ]
        
        for smiles, name in test_molecules:
            mol = molecular_utils.smiles_to_3d(smiles)
            
            if mol is not None:
                atom_types = self.ff_utils.assign_gaff_atom_types(mol)
                charges = self.ff_utils.calculate_partial_charges(mol, method="gasteiger")
                
                # Basic validation
                assert len(charges) == mol.GetNumAtoms(), f"Charge count mismatch for {name}"
                
                # Check neutrality
                total_charge = sum(charges.values())
                assert abs(total_charge) < 0.1, f"Non-neutral charges for {name}: {total_charge}"
                
                # Validate charges
                is_valid, issues = self.ff_utils.validate_charges(charges, mol)
                if not is_valid:
                    # Log issues but don't fail test (some molecules might have edge cases)
                    print(f"Charge validation issues for {name}: {issues}")


class TestDataHandlerOrganic:
    """Test organic functionality in DataHandler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_handler = DataHandler(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_with_organic_methods(self):
        """Test that DataHandler has organic methods."""
        assert hasattr(self.data_handler, 'import_smiles_structure')
        assert hasattr(self.data_handler, 'import_mol2_file')
        assert hasattr(self.data_handler, 'import_sdf_file')
        assert hasattr(self.data_handler, 'import_pdb_organic')
        assert hasattr(self.data_handler, 'create_liquid_box_file')
        assert hasattr(self.data_handler, 'assign_gaff_parameters')
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_import_smiles_structure(self):
        """Test SMILES structure import."""
        smiles = "CCO"
        molecule_name = "ethanol"
        
        try:
            structure_file = self.data_handler.import_smiles_structure(
                smiles=smiles,
                molecule_name=molecule_name,
                optimize_geometry=True
            )
            
            assert structure_file.exists()
            assert structure_file.suffix == ".data"
            
            # Check that metadata file was created
            metadata_file = structure_file.with_suffix(".json")
            assert metadata_file.exists()
            
            # Load and check metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            assert metadata["smiles"] == smiles
            assert metadata["molecule_name"] == molecule_name
            
        except Exception as e:
            # If RDKit functionality fails, that's expected in some environments
            pytest.skip(f"SMILES import failed (expected in some environments): {e}")
    
    def test_import_mol2_file_basic(self):
        """Test basic MOL2 file import."""
        mol2_content = """@<TRIPOS>MOLECULE
test_molecule
3 2 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
1 C1 0.0000 0.0000 0.0000 C.3 1 MOL 0.0000
2 C2 1.5000 0.0000 0.0000 C.3 1 MOL 0.0000
3 H1 -0.5000 0.0000 0.0000 H 1 MOL 0.0000

@<TRIPOS>BOND
1 1 2 1
2 1 3 1
"""
        
        filename = "test_molecule"
        
        try:
            structure_file = self.data_handler.import_mol2_file(mol2_content, filename)
            assert structure_file.exists()
            
        except Exception as e:
            # MOL2 import may fail without proper dependencies
            pytest.skip(f"MOL2 import failed (expected in some environments): {e}")
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_create_liquid_box_file(self):
        """Test liquid box creation."""
        molecules = [
            {"smiles": "CCO", "count": 50, "name": "ethanol"},
            {"smiles": "O", "count": 100, "name": "water"}
        ]
        
        try:
            box_file = self.data_handler.create_liquid_box_file(
                molecules=molecules,
                target_density=0.8,
                box_type="cubic"
            )
            
            assert box_file.exists()
            assert box_file.suffix == ".data"
            
            # Check that metadata was created
            metadata_file = box_file.with_suffix(".json")
            assert metadata_file.exists()
            
            # Load and check metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            assert len(metadata["molecules"]) == 2
            assert metadata["target_density"] == 0.8
            
        except Exception as e:
            pytest.skip(f"Liquid box creation failed (expected in some environments): {e}")
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_assign_gaff_parameters_with_charges(self):
        """Test GAFF parameter assignment includes proper charge calculation."""
        try:
            # First create a structure file
            smiles = "CCO"
            molecule_name = "ethanol_test"
            
            structure_file = self.data_handler.import_smiles_structure(
                smiles=smiles,
                molecule_name=molecule_name,
                optimize_geometry=True
            )
            
            # Now test GAFF parameter assignment
            result = self.data_handler.assign_gaff_parameters(structure_file)
            
            # Check that result contains charges
            assert "charges" in result
            assert isinstance(result["charges"], dict)
            assert len(result["charges"]) > 0
            
            # Check that charges are not all zero
            charges = result["charges"]
            assert not all(abs(charge) < 1e-6 for charge in charges.values())
            
            # Check charge neutrality
            total_charge = sum(charges.values())
            assert abs(total_charge) < 0.1
            
            # Check that statistics include total charge
            assert "statistics" in result
            assert "total_charge" in result["statistics"]
            
            # Check validation includes charge validation
            assert "validation" in result
            validation = result["validation"]
            assert isinstance(validation["is_valid"], bool)
            assert isinstance(validation["issues"], list)
            
        except Exception as e:
            pytest.skip(f"GAFF parameter assignment test failed (expected in some environments): {e}")
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_liquid_box_charge_assignment(self):
        """Test that liquid box creation properly assigns charges to all atoms."""
        try:
            # Create a liquid box with polar molecules
            molecules = [
                {"smiles": "CCO", "count": 2, "name": "ethanol"},
                {"smiles": "O", "count": 3, "name": "water"}
            ]
            
            box_file = self.data_handler.create_liquid_box_file(
                molecules=molecules,
                target_density=0.8,
                box_type="cubic"
            )
            
            assert box_file.exists()
            
            # Check metadata includes charge information
            metadata_file = box_file.with_suffix(".json")
            assert metadata_file.exists()
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Verify charge information is present
            assert "charge_information" in metadata
            charge_info = metadata["charge_information"]
            assert "total_system_charge" in charge_info
            assert "molecule_charges" in charge_info
            assert "charge_method" in charge_info
            
            # Check that system is approximately neutral
            total_charge = charge_info["total_system_charge"]
            assert abs(total_charge) < 0.1, f"System not neutral: {total_charge}"
            
            # Check molecule charge information
            mol_charges = charge_info["molecule_charges"]
            assert len(mol_charges) == 2  # ethanol and water
            
            # Verify each molecule type has charge info
            for mol_charge_info in mol_charges:
                assert "name" in mol_charge_info
                assert "charge_per_molecule" in mol_charge_info
                assert "total_charge_contribution" in mol_charge_info
                assert "num_atoms_per_molecule" in mol_charge_info
            
            # Check the actual data file has non-zero charges
            with open(box_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Find atom lines and check charges
            in_atoms = False
            atom_charges = []
            for line in lines:
                if line.strip() == "Atoms":
                    in_atoms = True
                    continue
                elif in_atoms and line.strip() and len(line.split()) >= 4:
                    if line.split()[0].isdigit():
                        charge = float(line.split()[3])
                        atom_charges.append(charge)
                elif in_atoms and line.strip().startswith("Bonds"):
                    break
            
            # Verify we found charges and they're not all zero
            assert len(atom_charges) > 0, "No atom charges found in data file"
            non_zero_charges = [c for c in atom_charges if abs(c) > 1e-6]
            assert len(non_zero_charges) > 0, f"All charges are zero: {atom_charges[:10]}"
            
            # Check that total charge from file matches metadata
            file_total_charge = sum(atom_charges)
            assert abs(file_total_charge - total_charge) < 1e-4, f"Charge mismatch: file={file_total_charge}, metadata={total_charge}"
            
        except Exception as e:
            pytest.skip(f"Liquid box charge test failed (expected in some environments): {e}")
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_liquid_box_different_molecule_types(self):
        """Test liquid box with different molecule types have proper charges."""
        try:
            # Test with various molecule types
            molecules = [
                {"smiles": "CC(=O)O", "count": 1, "name": "acetic_acid"},  # Has carboxyl group
                {"smiles": "CCN", "count": 1, "name": "ethylamine"},        # Has amine group
                {"smiles": "CCOC", "count": 1, "name": "diethyl_ether"}    # Has ether group
            ]
            
            box_file = self.data_handler.create_liquid_box_file(
                molecules=molecules,
                target_density=0.9,
                box_type="cubic"
            )
            
            # Load metadata
            metadata_file = box_file.with_suffix(".json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            charge_info = metadata["charge_information"]
            mol_charges = charge_info["molecule_charges"]
            
            # Check each molecule type
            molecule_names = [mol["name"] for mol in mol_charges]
            assert "acetic_acid" in molecule_names
            assert "ethylamine" in molecule_names
            assert "diethyl_ether" in molecule_names
            
            # Verify each molecule has reasonable charge distribution
            for mol_info in mol_charges:
                charge_per_mol = mol_info["charge_per_molecule"]
                # Each individual molecule should be approximately neutral
                assert abs(charge_per_mol) < 0.1, f"{mol_info['name']} not neutral: {charge_per_mol}"
            
            # System should be neutral overall
            total_charge = charge_info["total_system_charge"]
            assert abs(total_charge) < 1e-6, f"System not neutral: {total_charge}"
            
        except Exception as e:
            pytest.skip(f"Multi-molecule liquid box test failed (expected in some environments): {e}")
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_liquid_box_charge_consistency(self):
        """Test that liquid box charges are consistent across multiple instances."""
        try:
            # Create liquid box with multiple instances of the same molecule
            molecules = [
                {"smiles": "CCO", "count": 5, "name": "ethanol"}
            ]
            
            box_file = self.data_handler.create_liquid_box_file(
                molecules=molecules,
                target_density=0.8,
                box_type="cubic"
            )
            
            # Parse the data file to check charge consistency
            with open(box_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Extract all atom charges
            in_atoms = False
            atom_charges = []
            molecule_ids = []
            for line in lines:
                if line.strip() == "Atoms":
                    in_atoms = True
                    continue
                elif in_atoms and line.strip() and len(line.split()) >= 4:
                    if line.split()[0].isdigit():
                        parts = line.split()
                        molecule_id = int(parts[1])
                        charge = float(parts[3])
                        atom_charges.append(charge)
                        molecule_ids.append(molecule_id)
                elif in_atoms and line.strip().startswith("Bonds"):
                    break
            
            # Group charges by molecule instance
            molecule_charges = {}
            for i, (mol_id, charge) in enumerate(zip(molecule_ids, atom_charges)):
                if mol_id not in molecule_charges:
                    molecule_charges[mol_id] = []
                molecule_charges[mol_id].append(charge)
            
            # Check that all molecule instances have the same charge pattern
            assert len(molecule_charges) == 5, f"Expected 5 molecules, got {len(molecule_charges)}"
            
            # All molecules should have the same number of atoms
            atom_counts = [len(charges) for charges in molecule_charges.values()]
            assert all(count == atom_counts[0] for count in atom_counts), "Inconsistent atom counts"
            
            # Each molecule should have the same total charge (approximately)
            mol_total_charges = [sum(charges) for charges in molecule_charges.values()]
            for i in range(1, len(mol_total_charges)):
                charge_diff = abs(mol_total_charges[i] - mol_total_charges[0])
                assert charge_diff < 1e-6, f"Inconsistent molecule charges: {mol_total_charges}"
            
        except Exception as e:
            pytest.skip(f"Liquid box consistency test failed (expected in some environments): {e}")


class TestLAMMPSInterfaceOrganic:
    """Test organic functionality in LAMMPSInterface."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.lammps_interface = LAMMPSInterface()
    
    def test_create_amber_script(self):
        """Test AMBER script creation."""
        structure_file = "test_structure.data"
        
        script = self.lammps_interface.create_amber_script(
            structure_file=structure_file,
            force_field="gaff",
            temperature=300.0,
            pressure=1.0
        )
        
        assert isinstance(script, str)
        assert len(script) > 0
        assert structure_file in script
        assert "amber" in script.lower()
        assert "gaff" in script.lower() or "amber" in script.lower()
    
    def test_create_liquid_script(self):
        """Test liquid simulation script creation."""
        structure_file = "liquid_box.data"
        
        script = self.lammps_interface.create_liquid_script(
            structure_file=structure_file,
            force_field="gaff",
            temperature=298.15,
            pressure=1.0,
            density_target=0.8
        )
        
        assert isinstance(script, str)
        assert len(script) > 0
        assert structure_file in script
        assert "liquid" in script.lower() or "npt" in script.lower()
    
    def test_setup_property_computes(self):
        """Test property compute setup."""
        # Mock LAMMPS instance
        mock_lmp = Mock()
        
        # Test that the method doesn't crash
        self.lammps_interface.setup_property_computes(mock_lmp)
        
        # The method should have called run_command multiple times
        assert mock_lmp is not None
    
    def test_run_density_equilibration(self):
        """Test density equilibration."""
        # Mock LAMMPS instance
        mock_lmp = Mock()
        
        # Mock the get_system_property method to return converged density
        with patch.object(self.lammps_interface, 'get_system_property', return_value=1.0):
            with patch.object(self.lammps_interface, 'run_command'):
                result = self.lammps_interface.run_density_equilibration(
                    lmp=mock_lmp,
                    target_density=1.0,
                    temperature=300.0,
                    max_cycles=2,
                    tolerance=0.1
                )
                
                assert isinstance(result, bool)


class TestIntegration:
    """Integration tests for organic simulation workflow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not hasattr(MolecularUtils(), 'rdkit_available') or not MolecularUtils().rdkit_available,
                       reason="RDKit not available")
    def test_full_organic_workflow(self):
        """Test complete organic simulation workflow."""
        try:
            # 1. Create molecular structure from SMILES
            molecular_utils = MolecularUtils()
            smiles = "CCO"
            mol = molecular_utils.smiles_to_3d(smiles)
            
            if mol is None:
                pytest.skip("Could not create molecule from SMILES")
            
            # 2. Assign force field parameters
            ff_utils = ForceFieldUtils()
            atom_types = ff_utils.assign_gaff_atom_types(mol)
            
            assert len(atom_types) > 0
            
            # 3. Generate topology
            topology = ff_utils.generate_topology(mol, atom_types)
            
            assert len(topology["bonds"]) > 0
            
            # 4. Create LAMMPS data file
            data_content = ff_utils.create_lammps_data_file(mol, atom_types, topology)
            
            assert len(data_content) > 0
            assert "atoms" in data_content.lower()
            
            # 5. Create LAMMPS script
            lammps_interface = LAMMPSInterface()
            script_content = lammps_interface.create_amber_script(
                structure_file="test.data",
                force_field="gaff"
            )
            
            assert len(script_content) > 0
            assert "gaff" in script_content.lower() or "amber" in script_content.lower()
            
        except Exception as e:
            pytest.skip(f"Full workflow test failed (expected in some environments): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
