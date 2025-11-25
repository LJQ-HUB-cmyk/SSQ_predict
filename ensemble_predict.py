#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
集成学习预测脚本
训练多个模型，组合预测结果，提高准确率
"""
import numpy as np
import pandas as pd
from lstm_model_50 import SSQLSTMModel50
import os
import pickle


class EnsemblePredictor:
    """
    集成学习预测器
    训练多个模型，使用投票或平均的方式组合预测
    """
    def __init__(self, n_models=5, model_prefix="ensemble_model"):
        self.n_models = n_models
        self.model_prefix = model_prefix
        self.models = []
        self.scaler_file = "scaler.pkl"
        
    def train_ensemble(self, epochs=200, batch_size=24):
        """
        训练集成模型
        训练多个不同初始化的模型
        """
        print(f"开始训练 {self.n_models} 个集成模型...")
        print("=" * 60)
        
        for i in range(self.n_models):
            print(f"\n训练模型 {i+1}/{self.n_models}...")
            model_file = f"{self.model_prefix}_{i}.weights.h5"
            
            # 创建模型（每次初始化不同）
            model = SSQLSTMModel50(
                model_file=model_file,
                use_mean_teacher=True,
                teacher_alpha=0.99,
                use_classification=True
            )
            
            # 训练模型
            model.train(epochs=epochs, batch_size=batch_size)
            
            # 保存模型
            self.models.append(model)
            print(f"模型 {i+1} 训练完成，保存到 {model_file}")
        
        print("\n" + "=" * 60)
        print(f"所有 {self.n_models} 个模型训练完成！")
    
    def predict_ensemble(self, use_probability=True, top_k=10):
        """
        使用集成模型进行预测
        
        Args:
            use_probability: 是否使用概率平均（True）还是投票（False）
            top_k: 返回Top-K个最可能的号码
        """
        if len(self.models) == 0:
            # 如果没有训练，尝试加载已有模型
            self.load_models()
        
        if len(self.models) == 0:
            raise ValueError("没有可用的模型，请先训练模型")
        
        print(f"使用 {len(self.models)} 个模型进行集成预测...")
        
        # 获取最新数据
        if not os.path.exists("ssq_history.csv"):
            raise FileNotFoundError("数据文件不存在")
        
        df = pd.read_csv("ssq_history.csv", encoding='utf-8-sig')
        
        # 使用最后一个模型进行预测（所有模型结构相同）
        model = self.models[0]
        if model.model is None:
            model.load_model()
        
        # 准备输入数据
        X = model._prepare_features_from_csv(df)
        
        # 标准化
        with open(self.scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        
        # 创建序列（使用最后50期）
        seq_length = 50
        if len(X_scaled) < seq_length:
            raise ValueError(f"数据不足，需要至少{seq_length}期数据")
        
        X_seq = X_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        # 获取所有模型的预测
        all_predictions = []
        for i, m in enumerate(self.models):
            if m.model is None:
                m.load_model()
            pred = m.model.predict(X_seq, verbose=0)
            all_predictions.append(pred)
            print(f"模型 {i+1} 预测完成")
        
        # 组合预测结果
        if use_probability:
            # 方法1：概率平均
            print("\n使用概率平均方法...")
            predictions = self._average_probabilities(all_predictions)
        else:
            # 方法2：投票
            print("\n使用投票方法...")
            predictions = self._vote_predictions(all_predictions)
        
        # 解析预测结果
        red_balls = []
        for i in range(6):
            probs = predictions[i][0]
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            red_balls.append({
                'indices': top_indices + 1,  # 转换为1-33
                'probabilities': top_probs
            })
        
        blue_probs = predictions[6][0]
        blue_top_indices = np.argsort(blue_probs)[-top_k:][::-1]
        blue_top_probs = blue_probs[blue_top_indices]
        
        # 输出结果
        print("\n" + "=" * 60)
        print("集成预测结果（Top-10概率最高的号码）:")
        print("=" * 60)
        
        for i in range(6):
            print(f"\n红球{i+1} (Top-10):")
            for j, (idx, prob) in enumerate(zip(red_balls[i]['indices'], red_balls[i]['probabilities'])):
                print(f"  {j+1}. 号码 {idx:2d} - 概率: {prob:.4f} ({prob*100:.2f}%)")
        
        print(f"\n蓝球 (Top-10):")
        for j, (idx, prob) in enumerate(zip(blue_top_indices + 1, blue_top_probs)):
            print(f"  {j+1}. 号码 {idx:2d} - 概率: {prob:.4f} ({prob*100:.2f}%)")
        
        # 生成最终预测（选择每个位置概率最高的号码）
        final_red = [red_balls[i]['indices'][0] for i in range(6)]
        final_blue = blue_top_indices[0] + 1
        
        print("\n" + "=" * 60)
        print("最终预测号码（每个位置概率最高）:")
        print("=" * 60)
        print(f"红球: {sorted(final_red)}")
        print(f"蓝球: {final_blue}")
        print("=" * 60)
        
        return {
            'red_balls': final_red,
            'blue_ball': final_blue,
            'red_probabilities': red_balls,
            'blue_probabilities': {
                'indices': blue_top_indices + 1,
                'probabilities': blue_top_probs
            }
        }
    
    def _average_probabilities(self, all_predictions):
        """
        平均所有模型的概率预测
        """
        n_models = len(all_predictions)
        n_outputs = len(all_predictions[0])
        
        averaged = []
        for i in range(n_outputs):
            # 平均所有模型在第i个输出上的预测
            avg_pred = np.mean([pred[i] for pred in all_predictions], axis=0)
            averaged.append(avg_pred)
        
        return averaged
    
    def _vote_predictions(self, all_predictions):
        """
        投票方式组合预测
        每个模型预测一个号码，然后投票
        """
        n_models = len(all_predictions)
        n_outputs = len(all_predictions[0])
        
        voted = []
        for i in range(n_outputs):
            # 每个模型预测的类别
            predictions = [np.argmax(pred[i], axis=-1) for pred in all_predictions]
            # 投票（选择出现最多的）
            from collections import Counter
            votes = Counter(predictions)
            most_common = votes.most_common(1)[0][0]
            
            # 转换为概率分布（one-hot）
            n_classes = all_predictions[0][i].shape[-1]
            voted_pred = np.zeros((1, n_classes))
            voted_pred[0, most_common] = 1.0
            voted.append(voted_pred)
        
        return voted
    
    def load_models(self):
        """
        加载已训练的模型
        """
        for i in range(self.n_models):
            model_file = f"{self.model_prefix}_{i}.weights.h5"
            if os.path.exists(model_file):
                model = SSQLSTMModel50(
                    model_file=model_file,
                    use_mean_teacher=True,
                    use_classification=True
                )
                try:
                    model.load_model()
                    self.models.append(model)
                    print(f"加载模型 {i+1}: {model_file}")
                except:
                    print(f"无法加载模型 {i+1}: {model_file}")
        
        if len(self.models) > 0:
            print(f"成功加载 {len(self.models)} 个模型")
        else:
            print("没有找到已训练的模型")


def main():
    """
    主函数：训练集成模型并预测
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='集成学习预测')
    parser.add_argument('--train', action='store_true', help='训练集成模型')
    parser.add_argument('--predict', action='store_true', help='使用集成模型预测')
    parser.add_argument('--n_models', type=int, default=5, help='集成模型数量（默认5个）')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数（默认200）')
    parser.add_argument('--batch_size', type=int, default=24, help='批次大小（默认24）')
    
    args = parser.parse_args()
    
    ensemble = EnsemblePredictor(n_models=args.n_models)
    
    if args.train:
        ensemble.train_ensemble(epochs=args.epochs, batch_size=args.batch_size)
    
    if args.predict:
        result = ensemble.predict_ensemble(use_probability=True, top_k=10)
        print("\n提示: 此预测仅供参考，彩票开奖具有随机性，请理性投注！")


if __name__ == '__main__':
    main()

