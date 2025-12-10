#!/usr/bin/env python3
"""
Test script for the _generate_gaff_parameters function in LAMMPSInterface.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_lammps.lammps_interface import LAMMPSInterface
from mcp_lammps.utils.forcefield_utils import forcefield_utils

def test_generate_gaff_parameters():
    """Test the _generate_gaff_parameters function with various inputs."""

    print("Testing _generate_gaff_parameters function...")
    print("=" * 50)

    lammps_interface = LAMMPSInterface()

    # Test Case 1: Simple ethanol molecule
    print("\n1. Testing with ethanol molecule...")

    # Ethanol atom types (from GAFF assignment)
    ethanol_atom_types = {
        0: 'c3',  # CH3 carbon
        1: 'c3',  # CH2 carbon
        2: 'oh',  # OH oxygen
        3: 'hc',  # CH3 hydrogen
        4: 'hc',  # CH3 hydrogen
        5: 'hc',  # CH3 hydrogen
        6: 'hc',  # CH2 hydrogen
        7: 'hc',  # CH2 hydrogen
        8: 'ho'   # OH hydrogen
    }

    # Ethanol topology
    ethanol_topology = {
        "bonds": [
            {"atoms": [0, 1], "types": ["c3", "c3"]},
            {"atoms": [1, 2], "types": ["c3", "oh"]},
            {"atoms": [0, 3], "types": ["c3", "hc"]},
            {"atoms": [0, 4], "types": ["c3", "hc"]},
            {"atoms": [0, 5], "types": ["c3", "hc"]},
            {"atoms": [1, 6], "types": ["c3", "hc"]},
            {"atoms": [1, 7], "types": ["c3", "hc"]},
            {"atoms": [2, 8], "types": ["oh", "ho"]}
        ],
        "angles": [
            {"atoms": [0, 1, 2], "types": ["c3", "c3", "oh"]},
            {"atoms": [1, 2, 8], "types": ["c3", "oh", "ho"]},
            {"atoms": [3, 0, 1], "types": ["hc", "c3", "c3"]},
            {"atoms": [4, 0, 1], "types": ["hc", "c3", "c3"]},
            {"atoms": [5, 0, 1], "types": ["hc", "c3", "c3"]},
            {"atoms": [0, 1, 6], "types": ["c3", "c3", "hc"]},
            {"atoms": [0, 1, 7], "types": ["c3", "c3", "hc"]},
            {"atoms": [6, 1, 2], "types": ["hc", "c3", "oh"]},
            {"atoms": [7, 1, 2], "types": ["hc", "c3", "oh"]}
        ],
        "dihedrals": [
            {"atoms": [3, 0, 1, 2], "types": ["hc", "c3", "c3", "oh"]},
            {"atoms": [4, 0, 1, 2], "types": ["hc", "c3", "c3", "oh"]},
            {"atoms": [5, 0, 1, 2], "types": ["hc", "c3", "c3", "oh"]},
            {"atoms": [0, 1, 2, 8], "types": ["c3", "c3", "oh", "ho"]},
            {"atoms": [6, 1, 2, 8], "types": ["hc", "c3", "oh", "ho"]},
            {"atoms": [7, 1, 2, 8], "types": ["hc", "c3", "oh", "ho"]}
        ]
    }

    # Create topology types for ethanol
    ethanol_topology_types = create_topology_types(ethanol_topology)

    try:
        parameters = lammps_interface._generate_gaff_parameters(
            ethanol_atom_types,
            ethanol_topology,
            ethanol_topology_types
        )

        print(f"   Generated {len(parameters)} parameter lines")

        # Check that we have the expected sections
        sections_found = []
        for line in parameters:
            if '# Pair coefficients' in line:
                sections_found.append('pair')
            elif '# Bond coefficients' in line:
                sections_found.append('bond')
            elif '# Angle coefficients' in line:
                sections_found.append('angle')
            elif '# Dihedral coefficients' in line:
                sections_found.append('dihedral')

        print(f"   Sections found: {sections_found}")

        # Check pair coefficients
        pair_coeffs = [line for line in parameters if 'pair_coeff' in line]
        print(f"   Pair coefficients: {len(pair_coeffs)}")

        # Check bond coefficients
        bond_coeffs = [line for line in parameters if 'bond_coeff' in line]
        print(f"   Bond coefficients: {len(bond_coeffs)}")

        # Check angle coefficients
        angle_coeffs = [line for line in parameters if 'angle_coeff' in line]
        print(f"   Angle coefficients: {len(angle_coeffs)}")

        # Check dihedral coefficients
        dihedral_coeffs = [line for line in parameters if 'dihedral_coeff' in line]
        print(f"   Dihedral coefficients: {len(dihedral_coeffs)}")

        # Validate parameter format and values
        issues = validate_parameter_format(parameters)
        if issues:
            print(f"   ❌ Parameter format issues: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"      - {issue}")
        else:
            print("   ✅ All parameter formats valid")

    except Exception as e:
        print(f"   ❌ Error generating ethanol parameters: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Case 2: Benzene molecule (aromatic)
    print("\n2. Testing with benzene molecule...")

    benzene_atom_types = {
        0: 'ca', 1: 'ca', 2: 'ca', 3: 'ca', 4: 'ca', 5: 'ca',  # carbons
        6: 'ha', 7: 'ha', 8: 'ha', 9: 'ha', 10: 'ha', 11: 'ha'  # hydrogens
    }

    benzene_topology = {
        "bonds": [
            {"atoms": [0, 1], "types": ["ca", "ca"]},
            {"atoms": [1, 2], "types": ["ca", "ca"]},
            {"atoms": [2, 3], "types": ["ca", "ca"]},
            {"atoms": [3, 4], "types": ["ca", "ca"]},
            {"atoms": [4, 5], "types": ["ca", "ca"]},
            {"atoms": [5, 0], "types": ["ca", "ca"]},
            {"atoms": [0, 6], "types": ["ca", "ha"]},
            {"atoms": [1, 7], "types": ["ca", "ha"]},
            {"atoms": [2, 8], "types": ["ca", "ha"]},
            {"atoms": [3, 9], "types": ["ca", "ha"]},
            {"atoms": [4, 10], "types": ["ca", "ha"]},
            {"atoms": [5, 11], "types": ["ca", "ha"]}
        ],
        "angles": [
            # Ring angles
            {"atoms": [5, 0, 1], "types": ["ca", "ca", "ca"]},
            {"atoms": [0, 1, 2], "types": ["ca", "ca", "ca"]},
            {"atoms": [1, 2, 3], "types": ["ca", "ca", "ca"]},
            {"atoms": [2, 3, 4], "types": ["ca", "ca", "ca"]},
            {"atoms": [3, 4, 5], "types": ["ca", "ca", "ca"]},
            {"atoms": [4, 5, 0], "types": ["ca", "ca", "ca"]},
            # C-H angles
            {"atoms": [1, 0, 6], "types": ["ca", "ca", "ha"]},
            {"atoms": [5, 0, 6], "types": ["ca", "ca", "ha"]},
            {"atoms": [0, 1, 7], "types": ["ca", "ca", "ha"]},
            {"atoms": [2, 1, 7], "types": ["ca", "ca", "ha"]},
            {"atoms": [1, 2, 8], "types": ["ca", "ca", "ha"]},
            {"atoms": [3, 2, 8], "types": ["ca", "ca", "ha"]},
            {"atoms": [2, 3, 9], "types": ["ca", "ca", "ha"]},
            {"atoms": [4, 3, 9], "types": ["ca", "ca", "ha"]},
            {"atoms": [3, 4, 10], "types": ["ca", "ca", "ha"]},
            {"atoms": [5, 4, 10], "types": ["ca", "ca", "ha"]},
            {"atoms": [4, 5, 11], "types": ["ca", "ca", "ha"]},
            {"atoms": [0, 5, 11], "types": ["ca", "ca", "ha"]}
        ],
        "dihedrals": [
            # Ring dihedrals
            {"atoms": [4, 5, 0, 1], "types": ["ca", "ca", "ca", "ca"]},
            {"atoms": [5, 0, 1, 2], "types": ["ca", "ca", "ca", "ca"]},
            {"atoms": [0, 1, 2, 3], "types": ["ca", "ca", "ca", "ca"]},
            {"atoms": [1, 2, 3, 4], "types": ["ca", "ca", "ca", "ca"]},
            {"atoms": [2, 3, 4, 5], "types": ["ca", "ca", "ca", "ca"]},
            {"atoms": [3, 4, 5, 0], "types": ["ca", "ca", "ca", "ca"]},
            # C-H dihedrals
            {"atoms": [2, 1, 0, 6], "types": ["ca", "ca", "ca", "ha"]},
            {"atoms": [4, 5, 0, 6], "types": ["ca", "ca", "ca", "ha"]},
            {"atoms": [0, 1, 2, 8], "types": ["ca", "ca", "ca", "ha"]},
            {"atoms": [6, 0, 5, 4], "types": ["ha", "ca", "ca", "ca"]}
        ]
    }

    benzene_topology_types = create_topology_types(benzene_topology)

    try:
        benzene_parameters = lammps_interface._generate_gaff_parameters(
            benzene_atom_types,
            benzene_topology,
            benzene_topology_types
        )

        print(f"   Generated {len(benzene_parameters)} parameter lines")

        # Check for benzene-specific parameters
        benzene_pair_coeffs = [line for line in benzene_parameters if 'pair_coeff' in line and 'ca' in line]
        benzene_bond_coeffs = [line for line in benzene_parameters if 'bond_coeff' in line and 'ca-ca' in line]

        print(f"   Benzene CA-CA bond coefficients: {len(benzene_bond_coeffs)}")
        print(f"   Benzene CA pair coefficients: {len(benzene_pair_coeffs)}")

        benzene_issues = validate_parameter_format(benzene_parameters)
        if benzene_issues:
            print(f"   ❌ Benzene parameter issues: {len(benzene_issues)}")
        else:
            print("   ✅ All benzene parameter formats valid")

    except Exception as e:
        print(f"   ❌ Error generating benzene parameters: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Case 3: Edge cases
    print("\n3. Testing edge cases...")

    # Empty topology
    try:
        empty_params = lammps_interface._generate_gaff_parameters(
            {}, {}, {"bond_type_mapping": {}, "angle_type_mapping": {}, "dihedral_type_mapping": {}}
        )
        print(f"   Empty input: {len(empty_params)} lines generated")
    except Exception as e:
        print(f"   ❌ Empty input error: {e}")

    # Missing parameters
    try:
        missing_params = lammps_interface._generate_gaff_parameters(
            {"unknown": "xx"},  # Unknown atom type
            {"bonds": [{"atoms": [0, 1], "types": ["xx", "yy"]}]},
            {"bond_type_mapping": {("xx", "yy"): 1}, "angle_type_mapping": {}, "dihedral_type_mapping": {}}
        )
        print(f"   Missing parameters: {len(missing_params)} lines (should include fallbacks)")
        fallbacks = [line for line in missing_params if 'fallback' in line.lower()]
        print(f"   Fallback parameters: {len(fallbacks)}")
    except Exception as e:
        print(f"   ❌ Missing parameters error: {e}")

    print("\n" + "=" * 50)
    print("✅ _generate_gaff_parameters function test completed!")
    return True

def create_topology_types(topology: Dict[str, Any]) -> Dict[str, Any]:
    """Create topology type mappings from topology data."""

    # Collect unique types
    bond_types = set()
    for bond in topology.get("bonds", []):
        if "types" in bond:
            bond_types.add(tuple(sorted(bond["types"])))

    angle_types = set()
    for angle in topology.get("angles", []):
        if "types" in angle:
            angle_types.add(tuple(angle["types"]))

    dihedral_types = set()
    for dihedral in topology.get("dihedrals", []):
        if "types" in dihedral:
            dihedral_types.add(tuple(dihedral["types"]))

    # Create mappings
    bond_type_mapping = {bt: i+1 for i, bt in enumerate(sorted(bond_types))}
    angle_type_mapping = {at: i+1 for i, at in enumerate(sorted(angle_types))}
    dihedral_type_mapping = {dt: i+1 for i, dt in enumerate(sorted(dihedral_types))}

    return {
        "unique_bond_types": list(bond_types),
        "unique_angle_types": list(angle_types),
        "unique_dihedral_types": list(dihedral_types),
        "bond_type_mapping": bond_type_mapping,
        "angle_type_mapping": angle_type_mapping,
        "dihedral_type_mapping": dihedral_type_mapping
    }

def validate_parameter_format(parameters: List[str]) -> List[str]:
    """Validate that parameter lines have correct format."""

    issues = []

    for line in parameters:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if not parts:
            continue

        try:
            if 'pair_coeff' in line:
                # pair_coeff type1 type2 style epsilon sigma
                if len(parts) < 6:
                    issues.append(f"Invalid pair_coeff format: {line}")
                else:
                    type1, type2 = int(parts[1]), int(parts[2])
                    epsilon, sigma = float(parts[4]), float(parts[5])
                    if type1 <= 0 or type2 <= 0 or epsilon < 0 or sigma < 0:
                        issues.append(f"Invalid pair_coeff values: {line}")

            elif 'bond_coeff' in line:
                # bond_coeff type style k r0
                if len(parts) < 5:
                    issues.append(f"Invalid bond_coeff format: {line}")
                else:
                    coeff_type = int(parts[1])
                    k, r0 = float(parts[3]), float(parts[4])
                    if coeff_type <= 0 or k < 0 or r0 < 0:
                        issues.append(f"Invalid bond_coeff values: {line}")

            elif 'angle_coeff' in line:
                # angle_coeff type style k theta0
                if len(parts) < 5:
                    issues.append(f"Invalid angle_coeff format: {line}")
                else:
                    coeff_type = int(parts[1])
                    k, theta0 = float(parts[3]), float(parts[4])
                    if coeff_type <= 0 or k < 0 or theta0 < 0:
                        issues.append(f"Invalid angle_coeff values: {line}")

            elif 'dihedral_coeff' in line:
                # dihedral_coeff type style n_terms k n phase
                if len(parts) < 7:
                    issues.append(f"Invalid dihedral_coeff format: {line}")
                else:
                    coeff_type = int(parts[1])
                    n_terms, k, n, phase = int(parts[3]), float(parts[4]), int(parts[5]), float(parts[6])
                    if coeff_type <= 0 or n_terms < 1 or k < 0:
                        issues.append(f"Invalid dihedral_coeff values: {line}")

        except (ValueError, IndexError) as e:
            issues.append(f"Parse error in line '{line}': {e}")

    return issues

if __name__ == "__main__":
    test_generate_gaff_parameters()
