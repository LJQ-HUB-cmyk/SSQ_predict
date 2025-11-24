"""
双色球预测脚本
使用训练好的LSTM模型预测下一期号码
"""
import os
import sys
from lstm_model import SSQLSTMModel


def main():
    """主函数"""
    print("=" * 50)
    print("双色球号码预测系统")
    print("=" * 50)
    
    # 检查模型是否存在
    if not os.path.exists("ssq_lstm_model.h5"):
        print("错误: 模型文件不存在，请先运行 lstm_model.py 训练模型")
        return
    
    # 检查数据文件是否存在
    if not os.path.exists("processed_data.npz"):
        print("错误: 处理后的数据文件不存在，请先运行 data_processor.py")
        return
    
    # 创建模型实例并加载
    model = SSQLSTMModel()
    model.load_model()
    
    # 预测下一期（使用最近40期数据）
    print("\n正在预测下一期号码（使用最近40期数据）...")
    prediction = model.predict_next(use_last_n_periods=40)
    
    # 显示结果
    print("\n" + "=" * 50)
    print("预测结果:")
    print("=" * 50)
    print(f"红球: {prediction['红球']}")
    print(f"蓝球: {prediction['蓝球']}")
    print("=" * 50)
    print("\n提示: 此预测仅供参考，彩票开奖具有随机性，请理性投注！")


if __name__ == "__main__":
    main()

