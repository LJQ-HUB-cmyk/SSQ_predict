"""
双色球LSTM预测系统主运行脚本
一键运行所有步骤：数据获取 -> 数据预处理 -> 模型训练 -> 预测
"""
import os
import sys
import subprocess


def run_step(step_name, script_name):
    """运行单个步骤"""
    print("\n" + "=" * 60)
    print(f"步骤: {step_name}")
    print("=" * 60)
    
    if not os.path.exists(script_name):
        print(f"错误: 找不到脚本 {script_name}")
        return False
    
    try:
        # 使用subprocess运行脚本
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            cwd=os.path.dirname(os.path.abspath(script_name)) or os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"\n✓ {step_name} 完成")
            return True
        else:
            print(f"\n✗ {step_name} 失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ {step_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("双色球LSTM预测系统 - 完整流程")
    print("=" * 60)
    
    steps = [
        ("1. 获取历史数据", "data_fetcher.py"),
        ("2. 数据预处理", "data_processor.py"),
        ("3. 训练LSTM模型", "lstm_model.py"),
        ("4. 预测下一期号码", "predict.py"),
    ]
    
    # 检查是否跳过已完成的步骤
    skip_existing = True
    
    for step_name, script_name in steps:
        # 检查是否需要跳过
        if skip_existing:
            if step_name == "1. 获取历史数据" and os.path.exists("ssq_history.csv"):
                print(f"\n跳过 {step_name} (文件已存在)")
                continue
            elif step_name == "2. 数据预处理" and os.path.exists("processed_data.npz"):
                print(f"\n跳过 {step_name} (文件已存在)")
                continue
            elif step_name == "3. 训练LSTM模型" and os.path.exists("ssq_lstm_model.h5"):
                print(f"\n跳过 {step_name} (模型已存在)")
                continue
        
        # 运行步骤
        if not run_step(step_name, script_name):
            print(f"\n错误: {step_name} 失败，停止执行")
            return
    
    print("\n" + "=" * 60)
    print("所有步骤完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

