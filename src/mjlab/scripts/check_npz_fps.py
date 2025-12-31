"""Script to check FPS of npz motion files.

Usage:
    # Check single file
    python -m mjlab.scripts.check_npz_fps motion_file/pm_fall4:v0/motion.npz

    # Check all files in directory
    python -m mjlab.scripts.check_npz_fps --input-dir motion_file/pm_fall4:v0
"""

import argparse
from pathlib import Path

import numpy as np


def check_npz_fps(npz_file: str) -> None:
    """Check FPS of a single npz file.
    
    Args:
        npz_file: Path to npz file
    """
    npz_path = Path(npz_file)
    if not npz_path.exists():
        print(f"❌ File not found: {npz_file}")
        return
    
    try:
        data = np.load(npz_file)
        if 'fps' in data:
            fps_value = data['fps']
            # Handle both scalar and array cases
            if fps_value.ndim == 0:
                fps = float(fps_value.item())
            else:
                fps = float(fps_value[0])
            print(f"✓ {npz_path.name}: {fps} Hz")
        else:
            print(f"⚠ {npz_path.name}: FPS not found in file")
    except Exception as e:
        print(f"❌ Error reading {npz_path.name}: {e}")


def check_all_npz_in_dir(input_dir: str) -> None:
    """Check FPS of all npz files in a directory.
    
    Args:
        input_dir: Directory containing npz files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory not found: {input_dir}")
        return
    
    npz_files = sorted(list(input_path.glob("*.npz")))
    if len(npz_files) == 0:
        print(f"⚠ No npz files found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(npz_files)} npz files in {input_dir}\n")
    
    for npz_file in npz_files:
        check_npz_fps(str(npz_file))


def main():
    parser = argparse.ArgumentParser(
        description="Check FPS of npz motion files"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input npz file path (if checking single file)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing npz files (if checking multiple files)",
    )
    
    args = parser.parse_args()
    
    if args.input is None and args.input_dir is None:
        parser.error("Either provide an input file or --input-dir")
    
    if args.input is not None and args.input_dir is not None:
        parser.error("Provide either input file or --input-dir, not both")
    
    if args.input_dir is not None:
        check_all_npz_in_dir(args.input_dir)
    else:
        check_npz_fps(args.input)


if __name__ == "__main__":
    main()
