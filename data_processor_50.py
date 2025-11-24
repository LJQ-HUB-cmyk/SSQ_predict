"""
双色球数据预处理脚本（递增序列版本）
使用从第1期到当前期的所有历史数据预测下一期
例如：1-50期预测51期，1-51期预测52期，1-52期预测53期...
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class SSQDataProcessor50:
    def __init__(self, data_file="ssq_history.csv"):
        self.data_file = data_file
        self.scaler_file = "scaler.pkl"
        self.processed_data_file = "processed_data_50.npz"
        self.min_seq_length = 50  # 最小序列长度（从第50期开始预测）
        
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
        red_cols = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
        blue_col = '蓝球'
        
        red_balls = df[red_cols].astype(int).values
        blue_balls = df[blue_col].astype(int).values
        
        features = []
        for i in range(len(df)):
            red = red_balls[i]
            blue = blue_balls[i]
            
            red_sorted = sorted(red)
            red_sum = sum(red_sorted)
            red_mean = np.mean(red_sorted)
            red_std = np.std(red_sorted)
            red_max = max(red_sorted)
            red_min = min(red_sorted)
            red_range = red_max - red_min
            red_odd_count = sum(1 for x in red_sorted if x % 2 == 1)
            red_even_count = 6 - red_odd_count
            red_small_count = sum(1 for x in red_sorted if x <= 17)
            red_large_count = 6 - red_small_count
            sum_zone = red_sum // 50
            
            feature_vector = [
                *red_sorted,
                blue,
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
    
    def create_sequences(self, data, min_seq_length=50):
        """
        创建递增序列数据
        使用从第1期到当前期的所有历史数据预测下一期
        例如：1-50期预测51期，1-51期预测52期，1-52期预测53期...
        """
        X, y = [], []
        seq_lengths = []  # 记录每个样本使用的序列长度
        
        # 从min_seq_length+1期开始（因为需要至少min_seq_length期来预测）
        for i in range(min_seq_length, len(data)):
            # 输入：从第1期到第i期的所有数据
            seq_length = i  # 使用从1到i的所有期数
            X.append(data[0:i])  # 使用从第1期到第i期的所有数据
            # 输出：第i+1期的红球和蓝球
            y.append(data[i][:7])  # 只预测号码，不预测统计特征
            seq_lengths.append(seq_length)
        
        return X, y, seq_lengths
    
    def pad_sequences(self, X, max_length):
        """
        将不同长度的序列填充到相同长度（用于训练）
        使用最早的数据进行填充
        """
        X_padded = []
        for seq in X:
            if len(seq) < max_length:
                # 如果序列长度小于max_length，用最早的数据填充
                padding_needed = max_length - len(seq)
                padding = np.tile(seq[0:1], (padding_needed, 1))
                padded_seq = np.vstack([padding, seq])
            else:
                # 如果序列长度大于max_length，只取最近的max_length期
                padded_seq = seq[-max_length:]
            X_padded.append(padded_seq)
        return np.array(X_padded)
    
    def process_data(self, min_seq_length=50, train_ratio=0.8, max_seq_length=None):
        """
        处理数据并保存
        min_seq_length: 最小序列长度（从第几期开始预测）
        max_seq_length: 最大序列长度（如果设置，会截断或填充到该长度）
        """
        print("开始处理数据（使用递增序列：1-N期预测N+1期）...")
        
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
        
        # 创建递增序列
        X, y, seq_lengths = self.create_sequences(features_scaled, min_seq_length)
        print(f"创建了 {len(X)} 个训练样本")
        print(f"序列长度范围: {min(seq_lengths)} - {max(seq_lengths)} 期")
        
        # 如果设置了max_seq_length，进行填充或截断
        if max_seq_length is not None:
            print(f"将序列填充/截断到 {max_seq_length} 期")
            X = self.pad_sequences(X, max_seq_length)
            actual_seq_length = max_seq_length
        else:
            # 使用最大序列长度
            actual_seq_length = max(seq_lengths)
            print(f"将序列填充到最大长度 {actual_seq_length} 期")
            X = self.pad_sequences(X, actual_seq_length)
        
        y = np.array(y)
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
            seq_length=actual_seq_length,
            min_seq_length=min_seq_length,
            seq_lengths=seq_lengths  # 保存原始序列长度信息
        )
        print(f"处理后的数据已保存到 {self.processed_data_file}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'features_scaled': features_scaled,
            'seq_length': actual_seq_length
        }


if __name__ == "__main__":
    processor = SSQDataProcessor50()
    # 可以设置max_seq_length来限制最大序列长度，或者设为None使用所有数据
    data_dict = processor.process_data(min_seq_length=50, train_ratio=0.8, max_seq_length=None)
