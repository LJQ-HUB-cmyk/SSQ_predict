"""
双色球数据预处理脚本
将原始数据转换为LSTM模型可用的格式
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class SSQDataProcessor:
    def __init__(self, data_file="ssq_history.csv"):
        self.data_file = data_file
        self.scaler_file = "scaler.pkl"
        self.processed_data_file = "processed_data.npz"
        
    def load_data(self):
        """加载原始数据"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件 {self.data_file} 不存在，请先运行 data_fetcher.py")
        
        df = pd.read_csv(self.data_file, encoding='utf-8-sig')
        print(f"加载了 {len(df)} 期数据")
        return df
    
    def prepare_features(self, df):
        """
        准备特征数据
        特征包括：红球号码、蓝球号码、号码和、号码差等统计特征
        """
        # 提取红球和蓝球
        red_cols = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
        blue_col = '蓝球'
        
        # 转换为数值类型
        red_balls = df[red_cols].astype(int).values
        blue_balls = df[blue_col].astype(int).values
        
        # 计算统计特征
        features = []
        for i in range(len(df)):
            red = red_balls[i]
            blue = blue_balls[i]
            
            # 基础特征：红球号码（已排序）
            red_sorted = sorted(red)
            
            # 统计特征
            red_sum = sum(red_sorted)
            red_mean = np.mean(red_sorted)
            red_std = np.std(red_sorted)
            red_max = max(red_sorted)
            red_min = min(red_sorted)
            red_range = red_max - red_min
            
            # 奇偶比例
            red_odd_count = sum(1 for x in red_sorted if x % 2 == 1)
            red_even_count = 6 - red_odd_count
            
            # 大小比例（1-17为小，18-33为大）
            red_small_count = sum(1 for x in red_sorted if x <= 17)
            red_large_count = 6 - red_small_count
            
            # 和值段特征
            sum_zone = red_sum // 50
            
            # 组合所有特征
            feature_vector = [
                *red_sorted,  # 6个红球号码
                blue,  # 1个蓝球号码
                red_sum,
                red_mean,
                red_std,
                red_range,
                red_odd_count,
                red_even_count,
                red_small_count,
                red_large_count,
                sum_zone
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_sequences(self, data, seq_length=10):
        """
        创建时间序列数据
        seq_length: 使用前多少期数据来预测下一期
        """
        X, y = [], []
        
        for i in range(seq_length, len(data)):
            # 输入：前seq_length期的特征
            X.append(data[i-seq_length:i])
            # 输出：第i期的红球和蓝球（前7个是红球和蓝球，后面是统计特征）
            y.append(data[i][:7])  # 只预测号码，不预测统计特征
        
        return np.array(X), np.array(y)
    
    def process_data(self, seq_length=10, train_ratio=0.8):
        """
        处理数据并保存
        """
        print("开始处理数据...")
        
        # 加载数据
        df = self.load_data()
        
        # 按时间顺序排序（最老的在前）
        df = df.sort_values('期号', ascending=True)
        
        # 准备特征
        features = self.prepare_features(df)
        print(f"特征维度: {features.shape}")
        
        # 数据标准化
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        
        # 保存scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler已保存到 {self.scaler_file}")
        
        # 创建序列
        X, y = self.create_sequences(features_scaled, seq_length)
        print(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 划分训练集和测试集
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 保存处理后的数据
        np.savez(
            self.processed_data_file,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            features_scaled=features_scaled,
            seq_length=seq_length
        )
        print(f"处理后的数据已保存到 {self.processed_data_file}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'features_scaled': features_scaled
        }


if __name__ == "__main__":
    processor = SSQDataProcessor()
    data_dict = processor.process_data(seq_length=10, train_ratio=0.8)

