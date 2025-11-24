"""
根据指定期数预测下一期号码
"""
import os
import sys
from lstm_model import SSQLSTMModel


def main():
    """主函数"""
    print("=" * 50)
    print("根据指定期数预测下一期号码")
    print("=" * 50)
    
    # 检查模型是否存在
    if not os.path.exists("ssq_lstm_model.weights.h5"):
        print("错误: 模型文件不存在，请先运行 lstm_model.py 训练模型")
        return
    
    # 创建模型实例并加载
    model = SSQLSTMModel()
    model.load_model()
    
    # 示例1: 使用最近几期的期号
    print("\n示例1: 使用最近10期的期号进行预测")
    print("-" * 50)
    
    # 先获取最近10期的期号
    import pandas as pd
    df = pd.read_csv("ssq_history.csv", encoding='utf-8-sig')
    df = df.sort_values('期号', ascending=True)
    recent_periods = df['期号'].tail(10).tolist()
    
    print(f"使用的期号: {recent_periods}")
    prediction = model.predict_from_periods(recent_periods)
    
    print("\n" + "=" * 50)
    print("预测结果:")
    print("=" * 50)
    print(f"红球: {prediction['红球']}")
    print(f"蓝球: {prediction['蓝球']}")
    print("=" * 50)
    
    # 示例2: 使用指定的期号
    print("\n\n示例2: 使用指定的期号进行预测")
    print("-" * 50)
    print("你可以修改下面的期号列表来使用不同的历史期数")
    
    # 用户可以修改这里的期号列表
    custom_periods = ['2025135', '2025134', '2025133', '2025132', '2025131', 
                      '2025130', '2025129', '2025128', '2025126', '2025125']
    
    print(f"使用的期号: {custom_periods}")
    prediction2 = model.predict_from_periods(custom_periods)
    
    print("\n" + "=" * 50)
    print("预测结果:")
    print("=" * 50)
    print(f"红球: {prediction2['红球']}")
    print(f"蓝球: {prediction2['蓝球']}")
    print("=" * 50)
    print("\n提示: 此预测仅供参考，彩票开奖具有随机性，请理性投注！")


if __name__ == "__main__":
    main()

