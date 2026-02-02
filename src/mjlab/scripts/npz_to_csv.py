"""Script to convert npz motion files to CSV format.

This script converts npz motion files to CSV format, with each array saved as a separate CSV file.
For multi-dimensional arrays (e.g., body_pos_w), columns are flattened with descriptive names.

Usage:
    # Convert a single npz file
    python -m mjlab.scripts.npz_to_csv motion_file/pm_fall4:v0/motion.npz

    # Convert all npz files in a directory
    python -m mjlab.scripts.npz_to_csv --input-dir motion_file/pm_fall4:v0

    # Specify output directory
    python -m mjlab.scripts.npz_to_csv --input-dir motion_file/pm_fall4:v0 --output-dir output_csv

    # Use custom column order (single CSV file)
    python -m mjlab.scripts.npz_to_csv --input-dir motion_file/pm_fall4:v0 --custom-order
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np


# Define joint names in order
JOINT_NAMES = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_WAIST_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
    "J23_HEAD_YAW",
]

# Define body names in order (from user's CSV column order)
BODY_NAMES_ORDER = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L", "LINK_HIP_ROLL_L", "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L", "LINK_ANKLE_PITCH_L", "LINK_ANKLE_ROLL_L",
    "LINK_FOOT_L",
    "LINK_HIP_PITCH_R", "LINK_HIP_ROLL_R", "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R", "LINK_ANKLE_PITCH_R", "LINK_ANKLE_ROLL_R",
    "LINK_FOOT_R",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L", "LINK_SHOULDER_ROLL_L", "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L", "LINK_ELBOW_YAW_L", "LINK_ELBOW_END_L",
    "LINK_SHOULDER_PITCH_R", "LINK_SHOULDER_ROLL_R", "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R", "LINK_ELBOW_END_R",
    "LINK_HEAD_YAW",
]


def convert_npz_to_csv_custom_order(
    npz_file: str,
    output_file: str | None = None,
) -> None:
    """Convert npz file to CSV with custom column order.
    
    Args:
        npz_file: Path to input npz file
        output_file: Path to output CSV file. If None, uses same name as npz with .csv extension.
    """
    npz_path = Path(npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    
    if output_file is None:
        output_file = npz_path.parent / f"{npz_path.stem}.csv"
    else:
        output_file = Path(output_file)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Converting {npz_file} to CSV with custom column order...")
    print(f"[INFO] Output file: {output_file}")
    
    # Load npz file
    data = np.load(npz_file)
    
    # Get data arrays
    joint_pos = data['joint_pos']  # (num_frames, num_joints)
    joint_vel = data['joint_vel']  # (num_frames, num_joints)
    body_pos_w = data['body_pos_w']  # (num_frames, num_bodies, 3)
    body_quat_w = data['body_quat_w']  # (num_frames, num_bodies, 4) - wxyz format
    body_lin_vel_w = data['body_lin_vel_w']  # (num_frames, num_bodies, 3)
    body_ang_vel_w = data['body_ang_vel_w']  # (num_frames, num_bodies, 3)
    
    num_frames = joint_pos.shape[0]
    num_bodies = body_pos_w.shape[1]
    
    print(f"[INFO] Frames: {num_frames}, Bodies: {num_bodies}, Joints: {joint_pos.shape[1]}")
    
    # Find body indices (need to match body names in npz file)
    # We need to know the body order in the npz file - assume it matches BODY_NAMES_ORDER
    # If not, we'll need to map them
    body_indices = {}
    for i, body_name in enumerate(BODY_NAMES_ORDER):
        if i < num_bodies:
            body_indices[body_name] = i
    
    # Find LINK_BASE index (should be 0)
    base_idx = body_indices.get("LINK_BASE", 0)
    
    # Build column data
    columns = []
    column_data = []
    
    # 1. base_x, base_y, base_z
    columns.extend(["base_x", "base_y", "base_z"])
    column_data.extend([
        body_pos_w[:, base_idx, 0],
        body_pos_w[:, base_idx, 1],
        body_pos_w[:, base_idx, 2],
    ])
    
    # 2. base_qx, base_qy, base_qz, base_qw (convert from wxyz to xyzw)
    base_quat = body_quat_w[:, base_idx, :]  # (num_frames, 4) wxyz
    columns.extend(["base_qx", "base_qy", "base_qz", "base_qw"])
    column_data.extend([
        base_quat[:, 1],  # x
        base_quat[:, 2],  # y
        base_quat[:, 3],  # z
        base_quat[:, 0],  # w
    ])
    
    # 3. Joint positions (24 joints)
    for joint_name in JOINT_NAMES:
        columns.append(joint_name)
    # joint_pos is already in the correct order
    for i in range(joint_pos.shape[1]):
        column_data.append(joint_pos[:, i])
    
    # 4. base_vx, base_vy, base_vz
    columns.extend(["base_vx", "base_vy", "base_vz"])
    column_data.extend([
        body_lin_vel_w[:, base_idx, 0],
        body_lin_vel_w[:, base_idx, 1],
        body_lin_vel_w[:, base_idx, 2],
    ])
    
    # 5. base_wx, base_wy, base_wz
    columns.extend(["base_wx", "base_wy", "base_wz"])
    column_data.extend([
        body_ang_vel_w[:, base_idx, 0],
        body_ang_vel_w[:, base_idx, 1],
        body_ang_vel_w[:, base_idx, 2],
    ])
    
    # 6. Joint velocities (24 joints, with d prefix)
    for joint_name in JOINT_NAMES:
        columns.append(f"d{joint_name}")
    for i in range(joint_vel.shape[1]):
        column_data.append(joint_vel[:, i])
    
    # 7. left_foot_contact, right_foot_contact (not in npz, set to 0)
    columns.extend(["left_foot_contact", "right_foot_contact"])
    column_data.extend([
        np.zeros(num_frames),
        np.zeros(num_frames),
    ])
    
    # 8. base_quat_x, base_quat_y, base_quat_z, base_quat_w (repeat, in xyzw format)
    columns.extend(["base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"])
    column_data.extend([
        base_quat[:, 1],  # x
        base_quat[:, 2],  # y
        base_quat[:, 3],  # z
        base_quat[:, 0],  # w
    ])
    
    # 9. All body quaternions (in xyzw format, following BODY_NAMES_ORDER)
    for body_name in BODY_NAMES_ORDER:
        if body_name in body_indices:
            body_idx = body_indices[body_name]
            body_quat = body_quat_w[:, body_idx, :]  # (num_frames, 4) wxyz
            columns.extend([
                f"{body_name}_qx",
                f"{body_name}_qy",
                f"{body_name}_qz",
                f"{body_name}_qw",
            ])
            column_data.extend([
                body_quat[:, 1],  # x
                body_quat[:, 2],  # y
                body_quat[:, 3],  # z
                body_quat[:, 0],  # w
            ])
    
    # Stack all columns
    data_matrix = np.column_stack(column_data)
    
    # Write CSV with fixed-point notation (no scientific notation)
    print(f"[INFO] Writing CSV with {len(columns)} columns and {num_frames} rows...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in data_matrix:
            # Format each value with fixed-point notation, 16 decimal places
            formatted_row = [f"{val:.16f}" for val in row]
            writer.writerow(formatted_row)
    
    print(f"[INFO] ✓ Conversion complete: {output_file}")


def convert_npz_to_csv(
    npz_file: str,
    output_dir: str | None = None,
    flatten_3d: bool = True,
) -> None:
    """Convert a single npz file to CSV format.

    Args:
        npz_file: Path to input npz file
        output_dir: Output directory for CSV files. If None, uses same directory as npz file.
        flatten_3d: If True, flatten 3D arrays (e.g., body_pos_w) into columns with descriptive names.
                    If False, save each slice as a separate CSV file.
    """
    npz_path = Path(npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    # Determine output directory
    if output_dir is None:
        output_dir = npz_path.parent / f"{npz_path.stem}_csv"
    else:
        output_dir = Path(output_dir) / npz_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Converting {npz_file} to CSV...")
    print(f"[INFO] Output directory: {output_dir}")

    # Load npz file
    data = np.load(npz_file)
    keys = list(data.keys())

    print(f"[INFO] Found {len(keys)} arrays in npz file: {keys}")

    # Convert each array to CSV
    for key in keys:
        array = data[key]
        array_shape = array.shape

        # Handle scalar values
        if array_shape == ():
            # Scalar value (e.g., fps)
            csv_path = output_dir / f"{key}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([key])
                writer.writerow([array.item()])
            print(f"  ✓ {key}: scalar -> {csv_path}")

        # Handle 1D arrays
        elif len(array_shape) == 1:
            csv_path = output_dir / f"{key}.csv"
            np.savetxt(csv_path, array, delimiter=',', header=key, comments='', fmt='%.16f')
            print(f"  ✓ {key}: shape {array_shape} -> {csv_path}")

        # Handle 2D arrays
        elif len(array_shape) == 2:
            # Create column names: col_0, col_1, ...
            columns = [f"{key}_{i}" for i in range(array_shape[1])]
            csv_path = output_dir / f"{key}.csv"
            header = ','.join(columns)
            np.savetxt(csv_path, array, delimiter=',', header=header, comments='', fmt='%.16f')
            print(f"  ✓ {key}: shape {array_shape} -> {csv_path} ({array_shape[1]} columns)")

        # Handle 3D arrays
        elif len(array_shape) == 3:
            if flatten_3d:
                # Flatten last two dimensions into columns
                # e.g., body_pos_w (num_frames, num_bodies, 3) -> columns: body_0_x, body_0_y, body_0_z, body_1_x, ...
                num_frames, num_bodies, dim = array_shape
                columns = []
                data_flat = []
                for body_idx in range(num_bodies):
                    for dim_idx in range(dim):
                        col_name = f"{key}_body{body_idx}_{['x', 'y', 'z', 'w'][dim_idx]}"
                        columns.append(col_name)
                        data_flat.append(array[:, body_idx, dim_idx])
                
                # Stack columns
                data_stacked = np.column_stack(data_flat)
                csv_path = output_dir / f"{key}.csv"
                header = ','.join(columns)
                np.savetxt(csv_path, data_stacked, delimiter=',', header=header, comments='', fmt='%.16f')
                print(f"  ✓ {key}: shape {array_shape} -> {csv_path} ({len(columns)} columns, flattened)")
            else:
                # Save each body as a separate CSV file
                for body_idx in range(array_shape[1]):
                    body_data = array[:, body_idx, :]  # (num_frames, dim)
                    columns = [f"{key}_body{body_idx}_{['x', 'y', 'z', 'w'][i]}" 
                              for i in range(array_shape[2])]
                    csv_path = output_dir / f"{key}_body{body_idx}.csv"
                    header = ','.join(columns)
                    np.savetxt(csv_path, body_data, delimiter=',', header=header, comments='', fmt='%.16f')
                print(f"  ✓ {key}: shape {array_shape} -> {array_shape[1]} CSV files (one per body)")

        else:
            print(f"  ⚠ {key}: shape {array_shape} -> Skipped (unsupported dimension)")

    # Also create a summary file with metadata
    summary_path = output_dir / "_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"NPZ File: {npz_path}\n")
        f.write(f"Number of keys: {len(keys)}\n")
        f.write(f"Keys: {', '.join(keys)}\n\n")
        for key in keys:
            array = data[key]
            array_shape = array.shape
            f.write(f"{key}:\n")
            f.write(f"  Shape: {array_shape}\n")
            f.write(f"  Dtype: {array.dtype}\n")
            if array_shape != ():
                f.write(f"  Min: {np.min(array)}\n")
                f.write(f"  Max: {np.max(array)}\n")
            f.write("\n")
    print(f"  ✓ Summary saved to: {summary_path}")

    print(f"[INFO] ✓ Conversion complete: {output_dir}")


def convert_all_npz_in_dir(
    input_dir: str,
    output_dir: str | None = None,
    flatten_3d: bool = True,
    custom_order: bool = False,
) -> None:
    """Convert all npz files in a directory to CSV format.

    Args:
        input_dir: Directory containing npz files
        output_dir: Output directory for CSV files. If None, creates CSV files next to npz files.
        flatten_3d: If True, flatten 3D arrays into columns with descriptive names.
        custom_order: If True, use custom column order (single CSV per npz file).
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # Find all npz files
    npz_files = list(input_path.glob("*.npz"))
    if len(npz_files) == 0:
        print(f"[WARN] No npz files found in {input_dir}")
        return

    print(f"[INFO] Found {len(npz_files)} npz files in {input_dir}")

    # Convert each file
    success_count = 0
    failed_files = []

    for npz_file in npz_files:
        try:
            if custom_order:
                output_file = None
                if output_dir:
                    output_file = Path(output_dir) / f"{npz_file.stem}.csv"
                convert_npz_to_csv_custom_order(
                    str(npz_file),
                    output_file=str(output_file) if output_file else None,
                )
            else:
                convert_npz_to_csv(
                    str(npz_file),
                    output_dir=output_dir,
                    flatten_3d=flatten_3d,
                )
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to convert {npz_file}: {e}")
            failed_files.append((npz_file, str(e)))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"[INFO] Conversion summary:")
    print(f"  ✓ Successfully converted: {success_count}/{len(npz_files)}")
    if failed_files:
        print(f"  ✗ Failed: {len(failed_files)}")
        for npz_file, error in failed_files:
            print(f"    - {npz_file}: {error}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert npz motion files to CSV format"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input npz file path (if converting single file)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing npz files (if converting multiple files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (default: next to npz files)",
    )
    parser.add_argument(
        "--no-flatten-3d",
        action="store_true",
        help="Don't flatten 3D arrays (save each body as separate CSV file)",
    )
    parser.add_argument(
        "--custom-order",
        action="store_true",
        help="Use custom column order (single CSV file per npz, matching specified format)",
    )

    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.error("Either provide an input file or --input-dir")

    if args.input is not None and args.input_dir is not None:
        parser.error("Provide either input file or --input-dir, not both")

    flatten_3d = not args.no_flatten_3d

    if args.input_dir is not None:
        # Convert all files in directory
        convert_all_npz_in_dir(
            args.input_dir,
            output_dir=args.output_dir,
            flatten_3d=flatten_3d,
            custom_order=args.custom_order,
        )
    else:
        # Convert single file
        if args.custom_order:
            output_file = None
            if args.output_dir:
                input_path = Path(args.input)
                output_file = Path(args.output_dir) / f"{input_path.stem}.csv"
            convert_npz_to_csv_custom_order(
                args.input,
                output_file=str(output_file) if output_file else None,
            )
        else:
            convert_npz_to_csv(
                args.input,
                output_dir=args.output_dir,
                flatten_3d=flatten_3d,
            )


if __name__ == "__main__":
    main()
