"""批量或将单个 ONNX 文件转换为 MNN 格式。

Usage:
    # 转换目录下所有 ONNX
    python convert_onnx_to_mnn_batch.py --input_dir motion_file/pm_fall4:v0/onnx

    # 只转换单个文件（输出为同目录同名的 .mnn）
    python convert_onnx_to_mnn_batch.py --input_file motion_file/pm_fall4:v0/onnx/toFront_4_force.onnx
"""

import os
import subprocess
import sys
from pathlib import Path


def convert_onnx_to_mnn(onnx_file: str, mnn_file: str, verbose: bool = True) -> None:
    """将 ONNX 文件转换为 MNN 格式。
    
    Args:
        onnx_file: ONNX 文件路径
        mnn_file: 输出 MNN 文件路径
        verbose: 是否打印详细信息
    """
    # 优先使用 MNNConvert 命令行工具，如果不存在则尝试 Python 模块
    import shutil
    
    mnnconvert_cmd = shutil.which("MNNConvert")
    if mnnconvert_cmd:
        command = [
            mnnconvert_cmd,
            "-f",
            "ONNX",
            "--modelFile",
            onnx_file,
            "--MNNModel",
            mnn_file,
            "--bizCode",
            "MNN",
        ]
    else:
        # 回退到 Python 模块方式
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
        print(f"运行命令: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("✓ 成功转换为 MNN 格式!")
        print(f"✓ MNN 模型保存到: {mnn_file}")
        
        if verbose and result.stdout:
            print("\n转换器输出:")
            print(result.stdout)
        
        # 打印文件大小信息
        onnx_size = os.path.getsize(onnx_file) / (1024 * 1024)  # MB
        mnn_size = os.path.getsize(mnn_file) / (1024 * 1024)  # MB
        print("\n文件大小对比:")
        print(f"  ONNX: {onnx_size:.2f} MB")
        print(f"  MNN:  {mnn_size:.2f} MB")
        
    except FileNotFoundError:
        print("\n❌ 错误: 找不到 'python' 命令。")
        print("   请确保 Python 在系统 PATH 中。")
        raise
    except subprocess.CalledProcessError as e:
        print("\n❌ 转换 ONNX 到 MNN 失败。")
        print("   请确保已安装 MNN 包: pip install MNN")
        print("\n错误详情:")
        print(f"返回码: {e.returncode}")
        if e.stderr:
            print(f"错误信息:\n{e.stderr}")
        if e.stdout:
            print(f"输出:\n{e.stdout}")
        raise
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        raise


def convert_all_onnx_in_dir(input_dir: str, verbose: bool = True) -> None:
    """将指定目录下所有的 ONNX 文件转换为 MNN 格式。
    
    Args:
        input_dir: 包含 ONNX 文件的目录路径
        verbose: 是否打印详细信息
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 错误: 目录不存在: {input_dir}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"❌ 错误: 不是目录: {input_dir}")
        sys.exit(1)
    
    # 查找所有 ONNX 文件
    onnx_files = list(input_path.glob("*.onnx"))
    
    if len(onnx_files) == 0:
        print(f"⚠️  在目录 {input_dir} 中未找到 ONNX 文件")
        return
    
    print(f"找到 {len(onnx_files)} 个 ONNX 文件:")
    for f in onnx_files:
        print(f"  - {f.name}")
    print()
    
    # 转换每个文件
    success_count = 0
    failed_files = []
    
    for onnx_file in onnx_files:
        mnn_file = onnx_file.with_suffix(".mnn")
        
        print(f"{'=' * 60}")
        print(f"正在转换: {onnx_file.name}")
        print(f"{'=' * 60}")
        
        try:
            convert_onnx_to_mnn(
                str(onnx_file),
                str(mnn_file),
                verbose=verbose
            )
            success_count += 1
            print(f"✓ 成功: {onnx_file.name} -> {mnn_file.name}\n")
        except Exception as e:
            print(f"❌ 失败: {onnx_file.name}")
            print(f"   错误: {e}\n")
            failed_files.append(onnx_file.name)
    
    # 打印总结
    print(f"\n{'=' * 60}")
    print("转换完成!")
    print(f"{'=' * 60}")
    print(f"成功: {success_count}/{len(onnx_files)}")
    if failed_files:
        print(f"失败: {len(failed_files)}")
        print("失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="批量或将单个 ONNX 文件转换为 MNN 格式"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="包含 ONNX 文件的目录路径（与 --input_file 二选一）",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="单个 ONNX 文件路径（与 --input_dir 二选一），输出为同目录同名的 .mnn",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="打印详细信息",
    )

    args = parser.parse_args()

    if args.input_file is not None and args.input_dir is not None:
        print("❌ 错误: 只能指定 --input_file 或 --input_dir 其中之一")
        sys.exit(1)
    if args.input_file is None and args.input_dir is None:
        parser.error("请指定 --input_file 或 --input_dir")

    if args.input_file is not None:
        onnx_path = Path(args.input_file)
        if not onnx_path.exists():
            print(f"❌ 错误: 文件不存在: {args.input_file}")
            sys.exit(1)
        if onnx_path.suffix.lower() != ".onnx":
            print(f"❌ 错误: 不是 ONNX 文件: {args.input_file}")
            sys.exit(1)
        mnn_path = onnx_path.with_suffix(".mnn")
        print(f"正在转换单个文件: {onnx_path.name}\n")
        convert_onnx_to_mnn(str(onnx_path), str(mnn_path), verbose=args.verbose)
    else:
        convert_all_onnx_in_dir(args.input_dir, verbose=args.verbose)

