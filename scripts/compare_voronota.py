#!/usr/bin/env python3
"""
Compare outputs from two voronota binaries to verify identical results.

Uses the same two-step pipeline as VoroArea.py:
  1. voronota get-balls-from-atoms-file  (PDB -> balls)
  2. voronota calculate-contacts         (balls -> contacts)

Usage:
    python compare_voronota.py <pdb_file> <voronota_v1> <voronota_v2>

Example:
    python compare_voronota.py test.pdb ./voronota_old ./voronota_new
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_voronota_pipeline(voronota_exec: str, pdb_path: str) -> tuple[bytes, bytes]:
    """
    Run the two-step voronota pipeline.

    Returns:
        (balls_output, contacts_output) as raw bytes
    """
    with open(pdb_path, "rb") as f:
        pdb_content = f.read()

    # Step 1: get-balls-from-atoms-file
    balls_proc = subprocess.Popen(
        [voronota_exec, "get-balls-from-atoms-file"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    balls_output, balls_stderr = balls_proc.communicate(input=pdb_content)

    if balls_proc.returncode != 0:
        raise RuntimeError(
            f"get-balls-from-atoms-file failed for {voronota_exec}:\n{balls_stderr.decode()}"
        )

    # Step 2: calculate-contacts
    contacts_proc = subprocess.Popen(
        [voronota_exec, "calculate-contacts"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    contacts_output, contacts_stderr = contacts_proc.communicate(input=balls_output)

    if contacts_proc.returncode != 0:
        raise RuntimeError(
            f"calculate-contacts failed for {voronota_exec}:\n{contacts_stderr.decode()}"
        )

    return balls_output, contacts_output


def parse_balls(balls_data: bytes) -> dict[int, tuple]:
    """
    Parse balls output: 'x y z r # atomID chainID resSeq resName atomName'

    Returns dict mapping index -> (chainID, resSeq, resName, atomName)
    """
    lines = balls_data.decode("utf-8")
    balls = {}
    for index, line in enumerate(lines.strip().split("\n")):
        if not line.strip():
            continue
        # Format: x y z r # atomID chainID resSeq resName atomName
        parts = line.split("#")
        if len(parts) < 2:
            continue
        atom_info = parts[-1].strip().split()
        if len(atom_info) >= 5:
            # (chainID, resSeq, resName, atomName)
            balls[index] = (atom_info[1], atom_info[2], atom_info[3], atom_info[4])
    return balls


def parse_contacts(contacts_data: bytes) -> dict[tuple[int, int], float]:
    """
    Parse contacts output: 'b1 b2 area'

    Returns dict mapping (atom1_idx, atom2_idx) -> contact_area
    """
    lines = contacts_data.decode("utf-8")
    contacts = {}
    for line in lines.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 3:
            idx1 = int(parts[0])
            idx2 = int(parts[1])
            area = float(parts[2])
            contacts[(idx1, idx2)] = area
    return contacts


def compare_outputs(
    balls1: bytes, contacts1: bytes,
    balls2: bytes, contacts2: bytes,
    tolerance: float = 1e-6
) -> dict:
    """
    Compare outputs from two voronota runs.

    Returns dict with comparison results.
    """
    results = {
        "balls_identical": balls1 == balls2,
        "contacts_identical": contacts1 == contacts2,
        "balls1_lines": len(balls1.decode().strip().split("\n")),
        "balls2_lines": len(balls2.decode().strip().split("\n")),
        "contacts1_lines": len(contacts1.decode().strip().split("\n")),
        "contacts2_lines": len(contacts2.decode().strip().split("\n")),
        "differences": [],
    }

    # Parse and compare structured data
    parsed_balls1 = parse_balls(balls1)
    parsed_balls2 = parse_balls(balls2)
    parsed_contacts1 = parse_contacts(contacts1)
    parsed_contacts2 = parse_contacts(contacts2)

    results["balls1_count"] = len(parsed_balls1)
    results["balls2_count"] = len(parsed_balls2)
    results["contacts1_count"] = len(parsed_contacts1)
    results["contacts2_count"] = len(parsed_contacts2)

    # Compare balls
    if parsed_balls1 != parsed_balls2:
        all_keys = set(parsed_balls1.keys()) | set(parsed_balls2.keys())
        for k in sorted(all_keys):
            v1 = parsed_balls1.get(k)
            v2 = parsed_balls2.get(k)
            if v1 != v2:
                results["differences"].append(
                    f"Ball {k}: v1={v1} vs v2={v2}"
                )

    # Compare contacts
    all_contact_keys = set(parsed_contacts1.keys()) | set(parsed_contacts2.keys())
    for k in sorted(all_contact_keys):
        area1 = parsed_contacts1.get(k)
        area2 = parsed_contacts2.get(k)
        if area1 is None:
            results["differences"].append(
                f"Contact {k}: missing in v1, v2={area2}"
            )
        elif area2 is None:
            results["differences"].append(
                f"Contact {k}: v1={area1}, missing in v2"
            )
        elif abs(area1 - area2) > tolerance:
            results["differences"].append(
                f"Contact {k}: v1={area1:.6f} vs v2={area2:.6f} (diff={abs(area1-area2):.2e})"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare outputs from two voronota binaries"
    )
    parser.add_argument("pdb_file", help="Path to PDB file")
    parser.add_argument("voronota_v1", help="Path to first voronota binary")
    parser.add_argument("voronota_v2", help="Path to second voronota binary")
    parser.add_argument(
        "--tolerance", "-t", type=float, default=1e-6,
        help="Tolerance for floating-point comparison (default: 1e-6)"
    )
    parser.add_argument(
        "--save-outputs", "-s", action="store_true",
        help="Save raw outputs to files for inspection"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show all differences (not just summary)"
    )
    args = parser.parse_args()

    # Validate inputs
    pdb_path = Path(args.pdb_file)
    v1_path = Path(args.voronota_v1)
    v2_path = Path(args.voronota_v2)

    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}")
        sys.exit(1)
    if not v1_path.exists():
        print(f"Error: voronota v1 not found: {v1_path}")
        sys.exit(1)
    if not v2_path.exists():
        print(f"Error: voronota v2 not found: {v2_path}")
        sys.exit(1)

    print(f"PDB file: {pdb_path}")
    print(f"Voronota v1: {v1_path}")
    print(f"Voronota v2: {v2_path}")
    print()

    # Run pipelines
    print("Running voronota v1...")
    try:
        balls1, contacts1 = run_voronota_pipeline(str(v1_path), str(pdb_path))
        print(f"  balls: {len(balls1)} bytes, contacts: {len(contacts1)} bytes")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Running voronota v2...")
    try:
        balls2, contacts2 = run_voronota_pipeline(str(v2_path), str(pdb_path))
        print(f"  balls: {len(balls2)} bytes, contacts: {len(contacts2)} bytes")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Save outputs if requested
    if args.save_outputs:
        pdb_stem = pdb_path.stem
        Path(f"{pdb_stem}_v1_balls.txt").write_bytes(balls1)
        Path(f"{pdb_stem}_v1_contacts.txt").write_bytes(contacts1)
        Path(f"{pdb_stem}_v2_balls.txt").write_bytes(balls2)
        Path(f"{pdb_stem}_v2_contacts.txt").write_bytes(contacts2)
        print(f"\nSaved outputs to {pdb_stem}_v{{1,2}}_{{balls,contacts}}.txt")

    # Compare
    print("\nComparing outputs...")
    results = compare_outputs(balls1, contacts1, balls2, contacts2, args.tolerance)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Balls byte-identical:    {results['balls_identical']}")
    print(f"Contacts byte-identical: {results['contacts_identical']}")
    print()
    print(f"Balls count:    v1={results['balls1_count']}, v2={results['balls2_count']}")
    print(f"Contacts count: v1={results['contacts1_count']}, v2={results['contacts2_count']}")

    if results["differences"]:
        print()
        print(f"Found {len(results['differences'])} difference(s):")
        if args.verbose:
            for diff in results["differences"]:
                print(f"  {diff}")
        else:
            # Show first 10
            for diff in results["differences"][:10]:
                print(f"  {diff}")
            if len(results["differences"]) > 10:
                print(f"  ... and {len(results['differences']) - 10} more (use -v to see all)")
    else:
        print()
        print("SUCCESS: Outputs are identical (within tolerance)")

    # Exit code
    if results["balls_identical"] and results["contacts_identical"]:
        sys.exit(0)
    elif not results["differences"]:
        # Byte-different but structurally identical within tolerance
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
