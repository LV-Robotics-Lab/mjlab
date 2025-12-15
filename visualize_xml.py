#!/usr/bin/env python3
"""可视化 MuJoCo XML 模型的脚本。

用法:
    python visualize_xml.py [xml_file_path]
    
如果不提供路径，默认使用 serial_pm_v2.xml
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer


def visualize_xml(xml_path: str | Path) -> None:
    """加载并可视化 MuJoCo XML 模型。
    
    Args:
        xml_path: XML 文件的路径
    """
    xml_path = Path(xml_path)
    
    if not xml_path.exists():
        raise FileNotFoundError(f"XML 文件不存在: {xml_path}")
    
    print(f"正在加载模型: {xml_path}")
    
    try:
        # 加载 MuJoCo 模型
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        print(f"模型加载成功!")
        print(f"  - 自由度数量: {model.nv}")
        print(f"  - 执行器数量: {model.nu}")
        print(f"  - 体数量: {model.nbody}")
        print(f"  - 关节数量: {model.njnt}")
        print(f"  - 几何体数量: {model.ngeom}")
        
        # 创建数据
        data = mujoco.MjData(model)
        
        # 如果有 keyframe，使用它来设置初始状态
        if model.nkey > 0:
            print(f"使用 keyframe 设置初始状态")
            mujoco.mj_resetDataKeyframe(model, data, 0)
        
        # 启动交互式查看器
        print("\n启动 MuJoCo 查看器...")
        print("提示: 按 ESC 或关闭窗口退出")
        
        # 使用 launch_passive 需要手动控制仿真循环
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # 运行仿真循环
                while viewer.is_running():
                    # 前进一步仿真
                    mujoco.mj_step(model, data)
                    
                    # 同步查看器
                    viewer.sync()
        except AttributeError:
            # 如果 launch_passive 不可用，尝试使用 launch
            print("使用 launch 方法启动查看器...")
            mujoco.viewer.launch(model, data)
    
    except Exception as e:
        print(f"错误: {e}")
        raise


def main():
    """主函数"""
    # 默认 XML 文件路径
    default_xml = Path(__file__).parent / "src" / "mjlab" / "asset_zoo" / "robots" / "engineai_pm01" / "xmls" / "serial_pm_v2.xml"
    
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        xml_path = default_xml
    
    visualize_xml(xml_path)


if __name__ == "__main__":
    main()

