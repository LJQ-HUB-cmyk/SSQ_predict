"""
双色球LSTM预测模型（使用50期数据）
使用LSTM神经网络预测下一期双色球号码
训练时使用50期历史数据来预测下一期
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras import backend as K
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler



def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss用于处理类别不平衡问题
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return focal_loss_fixed


class MeanTeacherCallback(Callback):
    """
    平均老师（EMA）权重更新回调
    """
    def __init__(self, teacher_model, alpha=0.99):
        super().__init__()
        self.teacher_model = teacher_model
        self.alpha = alpha

    def on_train_begin(self, logs=None):
        # 开始训练前，先对齐老师模型和学生模型的初始权重
        if self.teacher_model is not None:
            self.teacher_model.set_weights(self.model.get_weights())

    def on_train_batch_end(self, batch, logs=None):
        if self.teacher_model is None:
            return
        student_weights = self.model.get_weights()
        teacher_weights = self.teacher_model.get_weights()
        updated_weights = []
        for tw, sw in zip(teacher_weights, student_weights):
            updated_weights.append(self.alpha * tw + (1.0 - self.alpha) * sw)
        self.teacher_model.set_weights(updated_weights)



class SSQLSTMModel50:
    """
    使用50期数据训练的LSTM模型
    """
    def __init__(self, model_file="ssq_lstm_model_50.weights.h5", use_mean_teacher=True, teacher_alpha=0.99, use_classification=True):
        if not model_file.endswith(".weights.h5"):
            base, _ = os.path.splitext(model_file)
            model_file = f"{base}.weights.h5"
        self.model_file = model_file
        self.scaler_file = "scaler.pkl"
        self.processed_data_file = "processed_data_50.npz"  # 使用50期的数据文件
        self.data_file = "ssq_history.csv"
        self.model = None
        self.teacher_model = None
        self.use_mean_teacher = use_mean_teacher
        self.teacher_alpha = teacher_alpha
        self.use_classification = use_classification
        self.seq_length = 50  # 使用50期数据
        
    def load_data(self):
        """加载处理后的数据（50期）"""
        if not os.path.exists(self.processed_data_file):
            raise FileNotFoundError(f"处理后的数据文件不存在，请先运行 data_processor.py 处理数据（seq_length=50）")
        
        data = np.load(self.processed_data_file, allow_pickle=True)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        seq_length = int(data['seq_length'])
        
        if seq_length != 50:
            print(f"警告: 数据文件中的seq_length是{seq_length}，不是50。请重新处理数据。")
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 如果使用分类模式，需要转换标签格式
        if self.use_classification:
            # 先反标准化y_train和y_test
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            def inverse_transform_y(y_scaled):
                """批量反标准化y数据"""
                n_samples = len(y_scaled)
                dummy_features = np.zeros((n_samples, scaler.n_features_in_))
                dummy_features[:, :7] = y_scaled
                y_original = scaler.inverse_transform(dummy_features)[:, :7]
                return y_original
            
            y_train_original = inverse_transform_y(y_train)
            y_test_original = inverse_transform_y(y_test)
            
            # 将红球从连续值转换为类别索引（1-33 -> 0-32）
            y_train_classification = {}
            y_test_classification = {}
            
            for i in range(6):
                red_train = np.clip(np.round(y_train_original[:, i]), 1, 33).astype(int)
                red_test = np.clip(np.round(y_test_original[:, i]), 1, 33).astype(int)
                
                if np.any(red_train < 1) or np.any(red_train > 33):
                    red_train = np.clip(red_train, 1, 33)
                if np.any(red_test < 1) or np.any(red_test > 33):
                    red_test = np.clip(red_test, 1, 33)
                
                y_train_classification[f'red_ball_{i}'] = (red_train - 1).astype(int)
                y_test_classification[f'red_ball_{i}'] = (red_test - 1).astype(int)
            
            # 蓝球：将值转换为类别索引（1-16 -> 0-15）
            blue_train = np.clip(np.round(y_train_original[:, 6]), 1, 16).astype(int)
            blue_test = np.clip(np.round(y_test_original[:, 6]), 1, 16).astype(int)
            
            if np.any(blue_train < 1) or np.any(blue_train > 16):
                blue_train = np.clip(blue_train, 1, 16)
            if np.any(blue_test < 1) or np.any(blue_test > 16):
                blue_test = np.clip(blue_test, 1, 16)
            
            y_train_classification['blue_ball'] = (blue_train - 1).astype(int)
            y_test_classification['blue_ball'] = (blue_test - 1).astype(int)
            
            return X_train, X_test, y_train_classification, y_test_classification, seq_length
        else:
            return X_train, X_test, y_train, y_test, seq_length
    
    def build_model(self, input_shape, use_classification=True):
        """
        构建LSTM模型（使用50期数据）
        """
        if use_classification:
            # 改进版本：添加注意力机制和更好的架构
            inputs = Input(shape=input_shape)
            
            # 第一层LSTM：提取基础特征
            lstm1 = LSTM(256, return_sequences=True, name='lstm1')(inputs)
            lstm1_norm = BatchNormalization(name='lstm1_norm')(lstm1)
            lstm1_drop = Dropout(0.3, name='lstm1_drop')(lstm1_norm)
            
            # 第二层LSTM：提取更深层特征
            lstm2 = LSTM(256, return_sequences=True, name='lstm2')(lstm1_drop)
            lstm2_norm = BatchNormalization(name='lstm2_norm')(lstm2)
            lstm2_drop = Dropout(0.3, name='lstm2_drop')(lstm2_norm)
            
            # 残差连接（如果维度匹配）
            if lstm1_drop.shape[-1] == lstm2_drop.shape[-1]:
                lstm2_drop = Add(name='lstm_residual')([lstm1_drop, lstm2_drop])
            
            # 第三层LSTM：进一步提取特征
            lstm3 = LSTM(128, return_sequences=True, name='lstm3')(lstm2_drop)
            lstm3_norm = BatchNormalization(name='lstm3_norm')(lstm3)
            lstm3_drop = Dropout(0.3, name='lstm3_drop')(lstm3_norm)
            
            # 添加注意力机制：让模型关注重要的时间步
            # 使用自注意力机制（query, key, value都使用lstm3_drop）
            attention = MultiHeadAttention(num_heads=4, key_dim=64, name='attention')(lstm3_drop, lstm3_drop, lstm3_drop)
            attention_norm = LayerNormalization(name='attention_norm')(attention)
            attention_drop = Dropout(0.2, name='attention_drop')(attention_norm)
            
            # 最后一层LSTM：汇总信息
            x = LSTM(128, return_sequences=False, name='lstm_final')(attention_drop)
            x = BatchNormalization(name='final_norm')(x)
            x = Dropout(0.3, name='final_drop')(x)
            
            # 共享的全连接层（增加容量，使用更好的激活函数）
            shared = Dense(512, activation='relu', name='shared_dense1')(x)
            shared = BatchNormalization(name='shared_norm1')(shared)
            shared = Dropout(0.3, name='shared_drop1')(shared)
            
            shared = Dense(256, activation='relu', name='shared_dense2')(shared)
            shared = BatchNormalization(name='shared_norm2')(shared)
            shared = Dropout(0.3, name='shared_drop2')(shared)
            
            shared = Dense(128, activation='relu', name='shared_dense3')(shared)
            shared = BatchNormalization(name='shared_norm3')(shared)
            shared = Dropout(0.2, name='shared_drop3')(shared)
            
            # 红球输出：6个位置，每个位置33个类别（1-33）
            # 使用更深的网络和更好的正则化
            red_outputs = []
            for i in range(6):
                red_dense = Dense(256, activation='relu', name=f'red_dense1_{i}')(shared)
                red_dense = BatchNormalization(name=f'red_norm1_{i}')(red_dense)
                red_dense = Dropout(0.3, name=f'red_drop1_{i}')(red_dense)
                
                red_dense2 = Dense(128, activation='relu', name=f'red_dense2_{i}')(red_dense)
                red_dense2 = BatchNormalization(name=f'red_norm2_{i}')(red_dense2)
                red_dense2 = Dropout(0.3, name=f'red_drop2_{i}')(red_dense2)
                
                red_dense3 = Dense(64, activation='relu', name=f'red_dense3_{i}')(red_dense2)
                red_dense3 = Dropout(0.2, name=f'red_drop3_{i}')(red_dense3)
                
                red_output = Dense(33, activation='softmax', name=f'red_ball_{i}')(red_dense3)
                red_outputs.append(red_output)
            
            # 蓝球输出：分类（16个类别，1-16）
            blue_dense = Dense(128, activation='relu', name='blue_dense1')(shared)
            blue_dense = BatchNormalization(name='blue_norm1')(blue_dense)
            blue_dense = Dropout(0.3, name='blue_drop1')(blue_dense)
            
            blue_dense2 = Dense(64, activation='relu', name='blue_dense2')(blue_dense)
            blue_dense2 = BatchNormalization(name='blue_norm2')(blue_dense2)
            blue_dense2 = Dropout(0.3, name='blue_drop2')(blue_dense2)
            
            blue_dense3 = Dense(32, activation='relu', name='blue_dense3')(blue_dense2)
            blue_dense3 = Dropout(0.2, name='blue_drop3')(blue_dense3)
            
            blue_output = Dense(16, activation='softmax', name='blue_ball')(blue_dense3)
            
            # 组合所有输出
            outputs = red_outputs + [blue_output]
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # 定义损失函数：使用Focal Loss处理类别不平衡，提高难样本的学习
            # 对于彩票预测这种类别不平衡问题，Focal Loss更有效
            losses = {}
            loss_weights = {}
            
            # 使用标准交叉熵（Focal Loss在某些情况下可能不稳定，先用标准损失）
            # 如果需要可以切换为focal_loss(gamma=2.0, alpha=0.25)
            for i in range(6):
                losses[f'red_ball_{i}'] = 'sparse_categorical_crossentropy'
                loss_weights[f'red_ball_{i}'] = 1.0
            losses['blue_ball'] = 'sparse_categorical_crossentropy'
            loss_weights['blue_ball'] = 1.2  # 稍微提高蓝球权重
            
            # 编译模型（使用自适应学习率）
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=0.001,  # 初始学习率稍高
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                ),
                loss=losses,
                loss_weights=loss_weights,
                metrics={
                    **{f'red_ball_{i}': 'sparse_categorical_accuracy' for i in range(6)},
                    'blue_ball': 'sparse_categorical_accuracy'
                }
            )
        else:
            # 回归版本
            model = Sequential([
                LSTM(256, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(128, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(64, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(7, activation='linear')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """
        训练模型（使用50期数据）
        """
        print("开始训练LSTM模型（使用50期数据）...")
        
        # 加载数据
        X_train, X_test, y_train, y_test, seq_length = self.load_data()
        
        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, use_classification=self.use_classification)
        self.teacher_model = None
        
        print("模型结构:")
        self.model.summary()
        
        # 回调函数（优化训练策略）
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=40,  # 增加patience，给模型更多训练机会
                restore_best_weights=True,
                verbose=1,
                min_delta=0.0001  # 设置最小改进阈值
            ),
            ModelCheckpoint(
                self.model_file,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # 适中的学习率衰减
                patience=15,  # 增加patience，避免过早降低学习率
                min_lr=1e-7,  # 更小的最小学习率
                verbose=1,
                mode='min',
                cooldown=5  # 学习率降低后的冷却期
            )
        ]

        # 平均老师策略
        if self.use_mean_teacher:
            self.teacher_model = self.build_model(input_shape, use_classification=self.use_classification)
            teacher_callback = MeanTeacherCallback(self.teacher_model, alpha=self.teacher_alpha)
            callbacks.append(teacher_callback)
            print(f"已启用平均老师策略 (alpha={self.teacher_alpha})")

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        print("\n评估模型...")
        student_train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        student_test_loss = self.model.evaluate(X_test, y_test, verbose=0)

        if self.use_classification:
            print(f"学生模型训练集总损失: {student_train_loss[0]:.6f}")
            print(f"学生模型测试集总损失: {student_test_loss[0]:.6f}")
            # evaluate返回顺序：loss(0), red_ball_0_loss(1), red_ball_1_loss(2), ..., blue_ball_loss(7),
            #                   red_ball_0_acc(8), red_ball_1_acc(9), ..., blue_ball_acc(14)
            # 准确率的索引是：8, 9, 10, 11, 12, 13 (红球0-5), 14 (蓝球)
            
            # 计算Top-K准确率（更合理的评估指标）
            def calculate_topk_accuracy(y_true_dict, y_pred_list, k=5):
                """计算Top-K准确率"""
                topk_acc = {}
                for i in range(6):
                    true_labels = y_true_dict[f'red_ball_{i}']
                    pred_probs = y_pred_list[i]
                    topk_pred = np.argsort(pred_probs, axis=1)[:, -k:]
                    correct = np.sum([true_labels[j] in topk_pred[j] for j in range(len(true_labels))])
                    topk_acc[f'red_ball_{i}'] = correct / len(true_labels) * 100
                
                # 蓝球
                true_labels = y_true_dict['blue_ball']
                pred_probs = y_pred_list[6]
                topk_pred = np.argsort(pred_probs, axis=1)[:, -k:]
                correct = np.sum([true_labels[j] in topk_pred[j] for j in range(len(true_labels))])
                topk_acc['blue_ball'] = correct / len(true_labels) * 100
                return topk_acc
            
            # 计算Top-5准确率
            train_pred = self.model.predict(X_train, verbose=0)
            test_pred = self.model.predict(X_test, verbose=0)
            train_top5 = calculate_topk_accuracy(y_train, train_pred, k=5)
            test_top5 = calculate_topk_accuracy(y_test, test_pred, k=5)
            
            for i in range(6):
                acc_idx = 8 + i  # 红球i的准确率索引：8, 9, 10, 11, 12, 13
                if len(student_train_loss) > acc_idx:
                    train_acc = student_train_loss[acc_idx] * 100  # 转换为百分比
                    test_acc = student_test_loss[acc_idx] * 100
                    train_top5_acc = train_top5[f'red_ball_{i}']
                    test_top5_acc = test_top5[f'red_ball_{i}']
                    print(f"  红球{i+1} - Top-1准确率: 训练{train_acc:.2f}%, 测试{test_acc:.2f}% | Top-5准确率: 训练{train_top5_acc:.2f}%, 测试{test_top5_acc:.2f}% (随机猜测: 3.03%)")
            # 蓝球准确率索引：14
            blue_acc_idx = 14
            if len(student_train_loss) > blue_acc_idx:
                train_acc = student_train_loss[blue_acc_idx] * 100
                test_acc = student_test_loss[blue_acc_idx] * 100
                train_top5_acc = train_top5['blue_ball']
                test_top5_acc = test_top5['blue_ball']
                print(f"  蓝球 - Top-1准确率: 训练{train_acc:.2f}%, 测试{test_acc:.2f}% | Top-5准确率: 训练{train_top5_acc:.2f}%, 测试{test_top5_acc:.2f}% (随机猜测: 6.25%)")
        else:
            print(f"学生模型训练集损失: {student_train_loss[0]:.6f}, MAE: {student_train_loss[1]:.6f}")
            print(f"学生模型测试集损失: {student_test_loss[0]:.6f}, MAE: {student_test_loss[1]:.6f}")

        if self.teacher_model is not None:
            teacher_train_loss = self.teacher_model.evaluate(X_train, y_train, verbose=0)
            teacher_test_loss = self.teacher_model.evaluate(X_test, y_test, verbose=0)

            if self.use_classification:
                print(f"教师模型训练集总损失: {teacher_train_loss[0]:.6f}")
                print(f"教师模型测试集总损失: {teacher_test_loss[0]:.6f}")
            else:
                print(f"教师模型训练集损失: {teacher_train_loss[0]:.6f}, MAE: {teacher_train_loss[1]:.6f}")
                print(f"教师模型测试集损失: {teacher_test_loss[0]:.6f}, MAE: {teacher_test_loss[1]:.6f}")

            self.model = self.teacher_model

        # 保存训练历史
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("training_history_50.csv", index=False)
        print("训练历史已保存到 training_history_50.csv")

        # 保存最终模型权重
        self.model.save_weights(self.model_file)
        status = '启用' if self.teacher_model is not None else '未启用'
        print(f"最终模型权重已保存到 {self.model_file} (平均老师策略: {status})")

        return history
    
    def load_model(self):
        """加载已训练的模型"""
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"模型文件 {self.model_file} 不存在，请先训练模型")
        
        # 加载数据以获取input_shape
        X_train, _, _, _, _ = self.load_data()
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 构建模型结构
        self.model = self.build_model(input_shape, use_classification=self.use_classification)
        
        # 加载权重
        self.model.load_weights(self.model_file)
        print(f"模型已从 {self.model_file} 加载（使用50期数据）")
        
        return self.model
    
    def _prepare_features_from_csv(self, df):
        """从CSV数据准备特征"""
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
    
    def predict_next(self, use_probability_sampling=False, random_seed=None, use_latest_data=True):
        """
        预测下一期号码（使用从第1期到当前期的所有历史数据）
        """
        if self.model is None:
            self.load_model()
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 加载数据
        if use_latest_data and os.path.exists(self.data_file):
            print("从CSV文件读取最新数据...")
            df = pd.read_csv(self.data_file, encoding='utf-8-sig')
            df = df.sort_values('期号', ascending=True)
            
            print(f"数据总期数: {len(df)}")
            if len(df) > 0:
                print(f"最新期号: {df.iloc[-1]['期号']}")
                latest_red = [df.iloc[-1][f'红球{i+1}'] for i in range(6)]
                latest_blue = df.iloc[-1]['蓝球']
                print(f"最新开奖: 红球{latest_red}, 蓝球{latest_blue}")
            
            features = self._prepare_features_from_csv(df)
            
            if not os.path.exists(self.scaler_file):
                raise FileNotFoundError(f"Scaler文件 {self.scaler_file} 不存在")
            
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            features_scaled = scaler.transform(features)
            
            # 使用从第1期到当前期的所有数据
            actual_seq_length = len(features_scaled)
            
            # 从processed_data.npz获取模型训练时的最大序列长度
            if os.path.exists(self.processed_data_file):
                data = np.load(self.processed_data_file, allow_pickle=True)
                model_seq_length = int(data['seq_length'])
            else:
                model_seq_length = actual_seq_length
            
            # 如果实际数据长度小于模型需要的长度，用最早的数据填充
            if actual_seq_length < model_seq_length:
                padding_needed = model_seq_length - actual_seq_length
                padding = np.tile(features_scaled[0:1], (padding_needed, 1))
                features_scaled = np.vstack([padding, features_scaled])
                print(f"警告: 数据不足，使用 {actual_seq_length} 期数据 + {padding_needed} 期填充")
            elif actual_seq_length > model_seq_length:
                # 如果实际数据长度大于模型需要的长度，只使用最近的model_seq_length期
                features_scaled = features_scaled[-model_seq_length:]
                print(f"注意: 数据有 {actual_seq_length} 期，模型使用最近 {model_seq_length} 期")
            else:
                print(f"使用所有 {actual_seq_length} 期数据进行预测")
            
            # 显示用于预测的数据（显示最近20期，如果数据少于20期则显示全部）
            display_count = min(20, len(df))
            print(f"\n用于预测的数据（显示最近 {display_count} 期）:")
            for i in range(max(0, len(df) - display_count), len(df)):
                red = [df.iloc[i][f'红球{j+1}'] for j in range(6)]
                blue = df.iloc[i]['蓝球']
                period = df.iloc[i]['期号'] if '期号' in df.columns else f"第{i+1}期"
                print(f"  {period}: 红球{red}, 蓝球{blue}")
            
            sequence = features_scaled.reshape(1, len(features_scaled), -1)
        else:
            if not os.path.exists(self.processed_data_file):
                raise FileNotFoundError(f"处理后的数据文件不存在")
            data = np.load(self.processed_data_file, allow_pickle=True)
            features_scaled = data['features_scaled']
            seq_length = int(data['seq_length'])
            print(f"使用processed_data.npz中的旧数据，数据量: {len(features_scaled)}")
            sequence = features_scaled[-seq_length:].reshape(1, seq_length, -1)
        
        # 预测
        predictions = self.model.predict(sequence, verbose=0)
        
        # 处理预测结果（与原来的predict_next相同的逻辑）
        if self.use_classification:
            print("\n预测概率信息:")
            for i in range(6):
                prob_dist = predictions[i][0]
                top5_indices = np.argsort(prob_dist)[-5:][::-1]
                top5_probs = prob_dist[top5_indices]
                print(f"  红球位置{i+1} 前5候选: ", end="")
                for idx, prob in zip(top5_indices, top5_probs):
                    print(f"{idx+1}({prob:.3f}) ", end="")
                print()
            
            red_balls = []
            for i in range(6):
                prob_dist = predictions[i][0]
                if use_probability_sampling:
                    available_probs = prob_dist.copy()
                    available_indices = np.arange(1, 34)
                    for selected in red_balls:
                        if selected in available_indices:
                            idx = selected - 1
                            available_probs[idx] = 0.0
                    if available_probs.sum() > 0:
                        available_probs = available_probs / available_probs.sum()
                        ball = np.random.choice(available_indices, p=available_probs)
                    else:
                        ball = np.argmax(prob_dist) + 1
                else:
                    prob_dist_copy = prob_dist.copy()
                    for selected in red_balls:
                        if 1 <= selected <= 33:
                            prob_dist_copy[selected - 1] = -np.inf
                    ball = np.argmax(prob_dist_copy) + 1
                red_balls.append(int(ball))
            
            red_balls = sorted(list(set(red_balls)))
            
            while len(red_balls) < 6:
                all_probs = np.mean([predictions[i][0] for i in range(6)], axis=0)
                for selected in red_balls:
                    if 1 <= selected <= 33:
                        all_probs[selected - 1] = 0.0
                if all_probs.sum() > 0:
                    all_probs = all_probs / all_probs.sum()
                    if use_probability_sampling:
                        new_ball = np.random.choice(np.arange(1, 34), p=all_probs)
                    else:
                        new_ball = np.argmax(all_probs) + 1
                else:
                    available = [x for x in range(1, 34) if x not in red_balls]
                    if available:
                        new_ball = available[0]
                    else:
                        new_ball = 1
                
                if new_ball not in red_balls:
                    red_balls.append(int(new_ball))
            
            red_balls = sorted(red_balls[:6])
            
            # 处理蓝球
            blue_prob_dist = predictions[6][0]
            if use_probability_sampling:
                blue_ball = np.random.choice(np.arange(1, 17), p=blue_prob_dist)
            else:
                blue_ball = np.argmax(blue_prob_dist) + 1
            
            print(f"\n蓝球预测信息:")
            top5_indices = np.argsort(blue_prob_dist)[-5:][::-1]
            top5_probs = blue_prob_dist[top5_indices]
            print(f"  前5候选: ", end="")
            for idx, prob in zip(top5_indices, top5_probs):
                print(f"{idx+1}({prob:.3f}) ", end="")
            print()
            print(f"  最终预测: {blue_ball} (概率: {blue_prob_dist[blue_ball-1]:.3f})")
        else:
            # 回归模式
            prediction_scaled = predictions[0] if isinstance(predictions, list) else predictions
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            dummy_features = np.zeros((1, scaler.n_features_in_))
            dummy_features[0, :7] = prediction_scaled[0]
            prediction_original = scaler.inverse_transform(dummy_features)[0]
            red_balls = np.clip(np.round(prediction_original[:6]), 1, 33).astype(int)
            red_balls = sorted(np.unique(red_balls))
            while len(red_balls) < 6:
                available = [x for x in range(1, 34) if x not in red_balls]
                if available:
                    red_balls.append(available[0])
                else:
                    break
            red_balls = sorted(red_balls[:6])
            blue_min = scaler.data_min_[6]
            blue_max = scaler.data_max_[6]
            blue_ball_scaled = prediction_scaled[0][6]
            blue_ball_original = blue_ball_scaled * (blue_max - blue_min) + blue_min
            blue_ball = int(np.round(np.clip(blue_ball_original, 1, 16)))
        
        return {
            '红球': red_balls,
            '蓝球': blue_ball
        }


if __name__ == "__main__":
    model = SSQLSTMModel50()
    
    # 训练模型
    print("开始训练（使用50期数据）...")
    model.train(epochs=200, batch_size=32)
    
    # 预测下一期
    print("\n预测下一期号码:")
    prediction = model.predict_next()
    print(f"红球: {prediction['红球']}")
    print(f"蓝球: {prediction['蓝球']}")

