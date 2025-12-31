"""Batch convert PyTorch (.pt) or ONNX files to MNN format.

This script uses functions from onnx_to_mnn.py to convert all .pt or .onnx files in a directory to MNN format.

Usage:
    # Convert all .pt files in a directory (will look for corresponding .onnx files first)
    python -m mjlab.scripts.pt_to_mnn_batch --input-dir motion_file/pm_fall4:v0/pt

    # Convert all .onnx files in a directory
    python -m mjlab.scripts.pt_to_mnn_batch --input-dir motion_file/pm_fall4:v0 --file-type onnx

    # Specify output directory
    python -m mjlab.scripts.pt_to_mnn_batch --input-dir motion_file/pm_fall4:v0/pt --output-dir output_mnn

    # Keep intermediate ONNX files
    python -m mjlab.scripts.pt_to_mnn_batch --input-dir motion_file/pm_fall4:v0/pt --keep-onnx
"""

import argparse
import os
from pathlib import Path

from mjlab.scripts.onnx_to_mnn import convert_to_mnn


def convert_all_pt_in_dir(
    input_dir: str,
    output_dir: str | None = None,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    keep_onnx: bool = False,
    verbose: bool = True,
    file_type: str = "pt",
) -> None:
    """Convert all .pt or .onnx files in a directory to MNN format.

    Args:
        input_dir: Directory containing .pt or .onnx files
        output_dir: Output directory for MNN files. If None, saves next to input files.
        input_shape: Input tensor shape for PyTorch models (only used for .pt files).
        keep_onnx: Whether to keep intermediate ONNX files (only for .pt files).
        verbose: Whether to print verbose output.
        file_type: File type to convert ("pt" or "onnx").
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # Find all files based on file_type
    if file_type == "onnx":
        files = sorted(list(input_path.glob("*.onnx")))
        file_ext = ".onnx"
    else:
        files = sorted(list(input_path.glob("*.pt")))
        file_ext = ".pt"
    
    if len(files) == 0:
        print(f"[WARN] No {file_ext} files found in {input_dir}")
        return

    print(f"[INFO] Found {len(files)} {file_ext} files in {input_dir}\n")

    # Convert each file
    success_count = 0
    failed_files = []

    for input_file in files:
        try:
            print(f"\n{'=' * 60}")
            print(f"[INFO] Converting: {input_file.name}")
            print(f"{'=' * 60}")

            # For .pt files, check if corresponding .onnx file exists in parent directory
            if file_type == "pt":
                # Check parent directory for corresponding .onnx file
                parent_dir = input_file.parent.parent
                onnx_file = parent_dir / f"{input_file.stem}.onnx"
                if onnx_file.exists():
                    print(f"[INFO] Found corresponding ONNX file: {onnx_file.name}")
                    print(f"[INFO] Converting ONNX directly (skipping PT -> ONNX conversion)")
                    # Use ONNX file instead
                    input_file_to_convert = onnx_file
                else:
                    input_file_to_convert = input_file
            else:
                input_file_to_convert = input_file

            # Determine output file path
            if output_dir is None:
                output_file = input_file.parent / f"{input_file.stem}.mnn"
            else:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_file = output_dir_path / f"{input_file.stem}.mnn"

            # Convert using function from onnx_to_mnn.py
            if file_type == "pt" and input_file_to_convert.suffix == ".onnx":
                # Direct ONNX conversion using MNNConvert tool
                import subprocess
                mnn_convert_path = "/usr/local/bin/MNNConvert"
                if not os.path.exists(mnn_convert_path):
                    # Try to find in PATH
                    import shutil
                    mnn_convert_path = shutil.which("MNNConvert")
                    if mnn_convert_path is None:
                        raise FileNotFoundError("MNNConvert not found. Please install MNN or set PATH.")
                
                command = [
                    mnn_convert_path,
                    "-f", "ONNX",
                    "--modelFile", str(input_file_to_convert),
                    "--MNNModel", str(output_file),
                    "--bizCode", "MNN",
                ]
                
                if verbose:
                    print(f"\nRunning command: {' '.join(command)}\n")
                
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                
                if verbose and result.stdout:
                    print(result.stdout)
                
                onnx_size = os.path.getsize(input_file_to_convert) / (1024 * 1024)  # MB
                mnn_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"\nFile size: ONNX {onnx_size:.2f} MB -> MNN {mnn_size:.2f} MB")
            else:
                convert_to_mnn(
                    input_file=str(input_file_to_convert),
                    output_file=str(output_file),
                    input_shape=input_shape,
                    keep_onnx=keep_onnx,
                    verbose=verbose,
                )

            success_count += 1
            print(f"\n✓ Successfully converted: {input_file.name} -> {output_file.name}")

        except Exception as e:
            print(f"\n❌ Failed to convert {input_file.name}: {e}")
            failed_files.append((input_file, str(e)))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"[INFO] Conversion summary:")
    print(f"  ✓ Successfully converted: {success_count}/{len(files)}")
    if failed_files:
        print(f"  ✗ Failed: {len(failed_files)}")
        for input_file, error in failed_files:
            print(f"    - {input_file.name}: {error}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert PyTorch (.pt) files to MNN format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing .pt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for MNN files (default: next to .pt files)",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=[1, 3, 224, 224],
        help="Input tensor shape for PyTorch models (default: [1, 3, 224, 224])",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep intermediate ONNX files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output",
    )
    parser.add_argument(
        "--file-type",
        type=str,
        choices=["pt", "onnx"],
        default="pt",
        help="File type to convert (default: pt)",
    )

    args = parser.parse_args()

    convert_all_pt_in_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_shape=tuple(args.input_shape),
        keep_onnx=args.keep_onnx,
        verbose=args.verbose,
        file_type=args.file_type,
    )


if __name__ == "__main__":
    main()
