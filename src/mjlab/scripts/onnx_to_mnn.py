"""Convert PyTorch or ONNX model to MNN format.

This script converts a PyTorch (.pt) or ONNX model file to MNN format.
For PyTorch models, it first converts to ONNX, then to MNN.
MNN is a lightweight deep learning inference framework optimized for mobile devices.

Usage:
    # From PyTorch model
    python onnx_to_mnn.py --input_file path/to/model.pt --output_file path/to/model.mnn

    # From ONNX model
    python onnx_to_mnn.py --input_file path/to/model.onnx --output_file path/to/model.mnn

    # Or let it auto-generate output filename
    python onnx_to_mnn.py --input_file path/to/model.pt

Requirements:
    - MNN Python package: pip install MNN
    - For PyTorch models: pip install torch onnx
"""

import os
import subprocess
import sys
import tempfile

import torch
import tyro


def convert_pt_to_onnx(
  pt_file: str,
  onnx_file: str,
  input_shape: tuple[int, ...] = (1, 3, 224, 224),
  verbose: bool = True,
) -> None:
  """Convert a PyTorch model to ONNX format.

  Args:
    pt_file: Path to the input PyTorch model file (.pt).
    onnx_file: Path to the output ONNX model file.
    input_shape: Input tensor shape for the model.
    verbose: Whether to print verbose output.
  """
  print("Converting PyTorch model to ONNX format...")
  print(f"  Input:  {pt_file}")
  print(f"  Output: {onnx_file}")

  try:
    # Try to load as TorchScript first
    try:
      model = torch.jit.load(pt_file)
      print("✓ Loaded as TorchScript model")
    except Exception as e1:
      if verbose:
        print(f"  Not a TorchScript model: {e1}")

      # Try loading as state dict
      try:
        checkpoint = torch.load(pt_file, map_location="cpu")

        # Check if it's a state dict or full checkpoint
        if isinstance(checkpoint, dict):
          if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("✓ Found 'model_state_dict' in checkpoint")
          elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("✓ Found 'model' in checkpoint")
          elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("✓ Found 'state_dict' in checkpoint")
          else:
            # Assume the whole dict is the state dict
            state_dict = checkpoint
            print("✓ Loaded checkpoint as state dict")
        else:
          raise ValueError("Unable to find model weights in checkpoint")

        print(
          "\n❌ Error: PyTorch state dict detected, but no model architecture provided."
        )
        print("   For state dict models, you need to:")
        print("   1. Load your model architecture in Python")
        print("   2. Load the state dict: model.load_state_dict(state_dict)")
        print(
          "   3. Export to TorchScript: torch.jit.script(model) or torch.jit.trace(model, dummy_input)"
        )
        print("   4. Save as TorchScript: torch.jit.save(scripted_model, 'model.pt')")
        print("   5. Then use this tool to convert the TorchScript model")
        print("\nAlternatively, if you have an ONNX file, you can directly convert it:")
        print("   python onnx_to_mnn.py --input_file model.onnx")
        sys.exit(1)

      except Exception as e2:
        print("\n❌ Failed to load PyTorch model:")
        print(f"   TorchScript error: {e1}")
        print(f"   State dict error: {e2}")
        print("\nSupported formats:")
        print("   - TorchScript models (torch.jit.save)")
        print("   - ONNX models (.onnx)")
        sys.exit(1)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    torch.onnx.export(
      model,
      dummy_input,
      onnx_file,
      export_params=True,
      opset_version=11,
      do_constant_folding=True,
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("✓ Successfully converted PyTorch model to ONNX!")

  except Exception as e:
    if "Failed to load PyTorch model" not in str(e):
      print(f"\n❌ Failed to convert PyTorch to ONNX: {e}")
      sys.exit(1)
    else:
      raise


def convert_onnx_to_mnn_internal(
  onnx_file: str,
  mnn_file: str,
  verbose: bool = True,
) -> None:
  """Internal function to convert ONNX to MNN.

  Args:
    onnx_file: Path to the input ONNX model file.
    mnn_file: Path to the output MNN model file.
    verbose: Whether to print verbose output from the converter.
  """
  # Build MNN converter command
  command = [
    "python",
    "-m",
    "MNN.tools.mnnconvert",
    "-f",
    "ONNX",
    "--modelFile",
    onnx_file,
    "--MNNModel",
    mnn_file,
    "--bizCode",
    "MNN",
  ]

  if verbose:
    print(f"\nRunning command: {' '.join(command)}\n")

  try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    print("✓ Successfully converted ONNX model to MNN!")
    print(f"✓ MNN model saved to: {mnn_file}")

    if verbose and result.stdout:
      print("\nConverter output:")
      print(result.stdout)

    # Print file size info
    onnx_size = os.path.getsize(onnx_file) / (1024 * 1024)  # MB
    mnn_size = os.path.getsize(mnn_file) / (1024 * 1024)  # MB
    print("\nFile size comparison:")
    print(f"  ONNX: {onnx_size:.2f} MB")
    print(f"  MNN:  {mnn_size:.2f} MB")

  except FileNotFoundError:
    print("\n❌ Error: 'python' command not found.")
    print("   Please ensure Python is in your system's PATH.")
    sys.exit(1)

  except subprocess.CalledProcessError as e:
    print("\n❌ Failed to convert ONNX to MNN.")
    print("   Please ensure MNN package is installed: pip install MNN")
    print("\nError details:")
    print(f"Return code: {e.returncode}")
    if e.stderr:
      print(f"Error message:\n{e.stderr}")
    if e.stdout:
      print(f"Output:\n{e.stdout}")
    sys.exit(1)

  except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    sys.exit(1)


def convert_to_mnn(
  input_file: str,
  output_file: str | None = None,
  input_shape: tuple[int, ...] = (1, 3, 224, 224),
  keep_onnx: bool = False,
  verbose: bool = True,
) -> None:
  """Convert PyTorch (.pt) or ONNX model to MNN format.

  Args:
    input_file: Path to the input model file (.pt or .onnx).
    output_file: Path to the output MNN model file. If None, uses same name as input with .mnn extension.
    input_shape: Input tensor shape for PyTorch models (ignored for ONNX).
    keep_onnx: If converting from PyTorch, whether to keep the intermediate ONNX file.
    verbose: Whether to print verbose output from the converter.
  """
  # Validate input file exists
  if not os.path.exists(input_file):
    print(f"Error: Input file does not exist: {input_file}")
    sys.exit(1)

  # Auto-generate output filename if not provided
  if output_file is None:
    output_file = os.path.splitext(input_file)[0] + ".mnn"

  # Determine input file type
  file_ext = os.path.splitext(input_file)[1].lower()

  if file_ext == ".pt":
    # PyTorch model: convert to ONNX first, then to MNN
    print("Detected PyTorch model (.pt)")

    # Create temporary or permanent ONNX file
    if keep_onnx:
      onnx_file = os.path.splitext(input_file)[0] + ".onnx"
    else:
      # Use temporary file
      temp_dir = tempfile.gettempdir()
      onnx_file = os.path.join(temp_dir, f"temp_{os.path.basename(input_file)}.onnx")

    try:
      # Step 1: PT -> ONNX
      convert_pt_to_onnx(input_file, onnx_file, input_shape, verbose)

      # Step 2: ONNX -> MNN
      convert_onnx_to_mnn_internal(onnx_file, output_file, verbose)

      # Cleanup temporary ONNX file if needed
      if not keep_onnx and os.path.exists(onnx_file):
        os.remove(onnx_file)
        print("✓ Cleaned up temporary ONNX file")
      elif keep_onnx:
        print(f"✓ Intermediate ONNX saved to: {onnx_file}")

      # Print final result
      pt_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
      mnn_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
      print(f"\n{'=' * 50}")
      print("✓ Conversion complete!")
      print(f"  PyTorch: {pt_size:.2f} MB -> MNN: {mnn_size:.2f} MB")
      print(f"  Final output: {output_file}")
      print(f"{'=' * 50}")

    except Exception as e:
      # Cleanup on error
      if not keep_onnx and os.path.exists(onnx_file):
        os.remove(onnx_file)
      raise e

  elif file_ext == ".onnx":
    # ONNX model: convert directly to MNN
    print("Detected ONNX model (.onnx)")
    convert_onnx_to_mnn_internal(input_file, output_file, verbose)

    print(f"\n{'=' * 50}")
    print("✓ Conversion complete!")
    print(f"  Final output: {output_file}")
    print(f"{'=' * 50}")

  else:
    print(f"Error: Unsupported file format: {file_ext}")
    print("Supported formats: .pt (PyTorch), .onnx (ONNX)")
    sys.exit(1)


if __name__ == "__main__":
  tyro.cli(convert_to_mnn)
