#!/usr/bin/env python3
"""查看 ONNX 模型的输入输出 tensor 信息。

用法:
    python inspect_onnx.py <onnx_file_path>
"""

import sys
from pathlib import Path

try:
  import onnx
except ImportError:
  print("错误: 需要安装 onnx 库")
  print("请运行: pip install onnx 或 conda install onnx")
  sys.exit(1)


def inspect_onnx(onnx_path: str | Path) -> None:
  """查看 ONNX 模型的输入输出信息。

  Args:
      onnx_path: ONNX 模型文件路径
  """
  onnx_path = Path(onnx_path)

  if not onnx_path.exists():
    print(f"错误: 文件不存在: {onnx_path}")
    sys.exit(1)

  model = onnx.load(str(onnx_path))

  dtype_map = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
  }

  print("=" * 80)
  print(f"ONNX 模型: {onnx_path}")
  print("=" * 80)

  print("\n输入 Tensor 信息:")
  print("-" * 80)
  for i, input_tensor in enumerate(model.graph.input, 1):
    print(f"\n输入 {i}:")
    print(f"  名称: {input_tensor.name}")
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
      if dim.dim_value > 0:
        shape.append(dim.dim_value)
      else:
        shape.append(dim.dim_param)
    print(f"  形状: {shape}")
    dtype = input_tensor.type.tensor_type.elem_type
    print(f"  数据类型: {dtype_map.get(dtype, f'unknown({dtype})')}")

  print("\n" + "=" * 80)
  print("输出 Tensor 信息:")
  print("-" * 80)
  for i, output_tensor in enumerate(model.graph.output, 1):
    print(f"\n输出 {i}:")
    print(f"  名称: {output_tensor.name}")
    shape = []
    for dim in output_tensor.type.tensor_type.shape.dim:
      if dim.dim_value > 0:
        shape.append(dim.dim_value)
      else:
        shape.append(dim.dim_param)
    print(f"  形状: {shape}")
    dtype = output_tensor.type.tensor_type.elem_type
    print(f"  数据类型: {dtype_map.get(dtype, f'unknown({dtype})')}")

  print("\n" + "=" * 80)
  print("模型元数据:")
  print("-" * 80)
  if model.metadata_props:
    for prop in model.metadata_props:
      print(f"  {prop.key}: {prop.value}")
  else:
    print("  无元数据")

  print("=" * 80)


def main():
  """主函数"""
  if len(sys.argv) < 2:
    print("用法: python inspect_onnx.py <onnx_file_path>")
    sys.exit(1)

  inspect_onnx(sys.argv[1])


if __name__ == "__main__":
  main()
