"""
双色球LSTM预测模型
使用LSTM神经网络预测下一期双色球号码
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler



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



class SSQLSTMModel:
    def __init__(self, model_file="ssq_lstm_model.weights.h5", use_mean_teacher=True, teacher_alpha=0.99, use_classification=True):
        if not model_file.endswith(".weights.h5"):
            base, _ = os.path.splitext(model_file)
            model_file = f"{base}.weights.h5"
        self.model_file = model_file
        self.scaler_file = "scaler.pkl"
        self.processed_data_file = "processed_data.npz"
        self.data_file = "ssq_history.csv"  # 原始数据文件
        self.model = None
        self.teacher_model = None
        self.use_mean_teacher = use_mean_teacher
        self.teacher_alpha = teacher_alpha
        self.use_classification = use_classification  # 是否使用分类模式
        
    def load_data(self):
        """加载处理后的数据"""
        if not os.path.exists(self.processed_data_file):
            raise FileNotFoundError(f"处理后的数据文件不存在，请先运行 data_processor.py")
        
        data = np.load(self.processed_data_file, allow_pickle=True)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        seq_length = int(data['seq_length'])
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 如果使用分类模式，需要转换标签格式
        if self.use_classification:
            # 先反标准化y_train和y_test，因为它们是标准化后的数据（0-1范围）
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # 反标准化红球和蓝球（向量化操作，提高效率）
            def inverse_transform_y(y_scaled):
                """批量反标准化y数据"""
                # 创建完整特征矩阵用于反标准化
                n_samples = len(y_scaled)
                dummy_features = np.zeros((n_samples, scaler.n_features_in_))
                dummy_features[:, :7] = y_scaled  # 前7个是号码
                # 批量反标准化
                y_original = scaler.inverse_transform(dummy_features)[:, :7]
                return y_original
            
            y_train_original = inverse_transform_y(y_train)
            y_test_original = inverse_transform_y(y_test)
            
            # 将红球从连续值转换为类别索引（1-33 -> 0-32）
            y_train_classification = {}
            y_test_classification = {}
            
            for i in range(6):
                # 红球：将值转换为类别索引（值1对应索引0，值33对应索引32）
                # 确保值在有效范围内并四舍五入
                red_train = np.clip(np.round(y_train_original[:, i]), 1, 33).astype(int)
                red_test = np.clip(np.round(y_test_original[:, i]), 1, 33).astype(int)
                
                # 验证没有无效值
                if np.any(red_train < 1) or np.any(red_train > 33):
                    print(f"警告: 训练集红球{i+1}有无效值，已自动修正")
                    red_train = np.clip(red_train, 1, 33)
                if np.any(red_test < 1) or np.any(red_test > 33):
                    print(f"警告: 测试集红球{i+1}有无效值，已自动修正")
                    red_test = np.clip(red_test, 1, 33)
                
                y_train_classification[f'red_ball_{i}'] = (red_train - 1).astype(int)
                y_test_classification[f'red_ball_{i}'] = (red_test - 1).astype(int)
            
            # 蓝球：将值转换为类别索引（1-16 -> 0-15）
            blue_train = np.clip(np.round(y_train_original[:, 6]), 1, 16).astype(int)
            blue_test = np.clip(np.round(y_test_original[:, 6]), 1, 16).astype(int)
            
            # 验证没有无效值
            if np.any(blue_train < 1) or np.any(blue_train > 16):
                print(f"警告: 训练集蓝球有无效值，已自动修正")
                blue_train = np.clip(blue_train, 1, 16)
            if np.any(blue_test < 1) or np.any(blue_test > 16):
                print(f"警告: 测试集蓝球有无效值，已自动修正")
                blue_test = np.clip(blue_test, 1, 16)
            
            y_train_classification['blue_ball'] = (blue_train - 1).astype(int)  # 值1对应索引0，值16对应索引15
            y_test_classification['blue_ball'] = (blue_test - 1).astype(int)
            
            return X_train, X_test, y_train_classification, y_test_classification, seq_length
        else:
            return X_train, X_test, y_train, y_test, seq_length
    
    def build_model(self, input_shape, use_classification=True):
        """
        构建LSTM模型
        use_classification: 如果True，红球使用多分类，蓝球使用回归；如果False，使用原来的回归方式
        """
        if use_classification:
            # 改进版本：红球使用多分类，蓝球使用回归
            inputs = Input(shape=input_shape)
            
            # 共享的LSTM层
            x = LSTM(128, return_sequences=True)(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = LSTM(64, return_sequences=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = LSTM(32, return_sequences=False)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # 共享的全连接层
            shared = Dense(64, activation='relu')(x)
            shared = Dropout(0.2)(shared)
            shared = Dense(32, activation='relu')(shared)
            shared = Dropout(0.2)(shared)
            
            # 红球输出：6个位置，每个位置33个类别（号码范围：1-33）
            # 注意：33个类别对应索引0-32，值1-33
            red_outputs = []
            for i in range(6):
                red_dense = Dense(64, activation='relu', name=f'red_dense_{i}')(shared)
                red_dense = Dropout(0.2)(red_dense)
                red_output = Dense(33, activation='softmax', name=f'red_ball_{i}')(red_dense)  # 33个类别：1-33
                red_outputs.append(red_output)
            
            # 蓝球输出：分类（16个类别，1-16）
            blue_dense = Dense(32, activation='relu', name='blue_dense')(shared)  # 注意：这里的32是隐藏层单元数，不是蓝球范围
            blue_dense = Dropout(0.2)(blue_dense)
            blue_output = Dense(16, activation='softmax', name='blue_ball')(blue_dense)  # 16个类别：1-16
            
            # 组合所有输出
            outputs = red_outputs + [blue_output]
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # 定义损失函数：红球和蓝球都用分类损失
            losses = {}
            loss_weights = {}
            for i in range(6):
                losses[f'red_ball_{i}'] = 'sparse_categorical_crossentropy'
                loss_weights[f'red_ball_{i}'] = 1.0
            losses['blue_ball'] = 'sparse_categorical_crossentropy'  # 蓝球也使用分类损失
            loss_weights['blue_ball'] = 0.8  # 蓝球权重稍低
            
            # 编译模型
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=losses,
                loss_weights=loss_weights,
                metrics={
                    **{f'red_ball_{i}': 'sparse_categorical_accuracy' for i in range(6)},
                    'blue_ball': 'sparse_categorical_accuracy'  # 蓝球也使用准确率
                }
            )
        else:
            # 原来的回归版本（向后兼容）
            model = Sequential([
                # 第一层LSTM
                LSTM(128, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.3),
                
                # 第二层LSTM
                LSTM(64, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                
                # 第三层LSTM
                LSTM(32, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),
                
                # 全连接层
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                
                # 输出层：7个数字（6个红球 + 1个蓝球）
                Dense(7, activation='linear')
            ])
            
            # 编译模型
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """
        训练模型
        """
        print("开始训练LSTM模型...")
        
        # 加载数据
        X_train, X_test, y_train, y_test, seq_length = self.load_data()
        
        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, use_classification=self.use_classification)
        self.teacher_model = None
        
        print("模型结构:")
        self.model.summary()
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_file,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
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
            # 分类模式的评估输出
            print(f"学生模型训练集总损失: {student_train_loss[0]:.6f}")
            print(f"学生模型测试集总损失: {student_test_loss[0]:.6f}")
            # 显示红球准确率
            for i in range(6):
                train_acc_idx = 1 + i if i < 6 else None
                test_acc_idx = 1 + i if i < 6 else None
                if train_acc_idx and test_acc_idx and len(student_train_loss) > train_acc_idx:
                    print(f"  红球{i+1}训练准确率: {student_train_loss[train_acc_idx]:.4f}, 测试准确率: {student_test_loss[test_acc_idx]:.4f}")
            if len(student_train_loss) > 7:
                print(f"  蓝球训练准确率: {student_train_loss[7]:.4f}, 测试准确率: {student_test_loss[7]:.4f}")
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

            # 使用教师模型作为最终模型
            self.model = self.teacher_model

        # 保存训练历史
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("training_history.csv", index=False)
        print("训练历史已保存到 training_history.csv")

        # 保存最终模型权重（若使用平均老师则保存教师模型）
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
        print(f"模型已从 {self.model_file} 加载")
        
        return self.model
    
    def _prepare_features_from_csv(self, df):
        """
        从CSV数据准备特征（与data_processor.py中的逻辑一致）
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
    
    def predict_from_periods(self, period_numbers, use_probability_sampling=False, random_seed=None):
        """
        根据指定的期号列表预测下一期号码
        
        参数:
            period_numbers: 期号列表（字符串或数字），例如 ['2025135', '2025134', ...] 或 [2025135, 2025134, ...]
            use_probability_sampling: 如果True，使用概率采样（结果会随机）；如果False，使用argmax（确定性结果）
            random_seed: 随机种子，如果设置则每次预测结果相同（仅在use_probability_sampling=True时有效）
        
        返回:
            预测结果字典，包含红球和蓝球
        """
        if self.model is None:
            self.load_model()
        
        # 设置随机种子（如果提供）
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 加载CSV数据
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件 {self.data_file} 不存在，请先运行 data_fetcher.py")
        
        df = pd.read_csv(self.data_file, encoding='utf-8-sig')
        
        # 将期号转换为字符串格式以便匹配
        df['期号'] = df['期号'].astype(str)
        period_numbers = [str(p) for p in period_numbers]
        
        # 筛选指定的期号
        selected_df = df[df['期号'].isin(period_numbers)].copy()
        
        if len(selected_df) == 0:
            raise ValueError(f"未找到指定的期号: {period_numbers}")
        
        if len(selected_df) < len(period_numbers):
            found_periods = selected_df['期号'].tolist()
            missing_periods = [p for p in period_numbers if p not in found_periods]
            print(f"警告: 部分期号未找到: {missing_periods}")
        
        # 按期号排序（按时间顺序，最老的在前）
        selected_df = selected_df.sort_values('期号', ascending=True)
        
        print(f"使用 {len(selected_df)} 期数据进行预测:")
        for idx, row in selected_df.iterrows():
            red = [row[f'红球{i+1}'] for i in range(6)]
            blue = row['蓝球']
            print(f"  {row['期号']}: 红球{red}, 蓝球{blue}")
        
        # 准备特征
        features = self._prepare_features_from_csv(selected_df)
        
        # 加载scaler并标准化
        if not os.path.exists(self.scaler_file):
            raise FileNotFoundError(f"Scaler文件 {self.scaler_file} 不存在，请先运行 data_processor.py")
        
        with open(self.scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        
        features_scaled = scaler.transform(features)
        
        # 从processed_data.npz获取seq_length（模型训练时使用的序列长度）
        if os.path.exists(self.processed_data_file):
            data = np.load(self.processed_data_file, allow_pickle=True)
            model_seq_length = int(data['seq_length'])
        else:
            model_seq_length = 10  # 默认值
        
        # 使用指定的期数，但模型输入长度必须匹配训练时的长度
        actual_periods = len(features_scaled)
        seq_length = model_seq_length
        
        if actual_periods < seq_length:
            # 如果提供的期数少于模型需要的期数，用最早的数据填充
            padding_needed = seq_length - actual_periods
            padding = np.tile(features_scaled[0:1], (padding_needed, 1))
            features_scaled = np.vstack([padding, features_scaled])
            print(f"\n注意: 提供的期数({actual_periods})少于模型需要的期数({seq_length})，已用最早的数据填充")
        elif actual_periods > seq_length:
            # 如果提供的期数多于模型需要的期数，只使用最近的seq_length期
            features_scaled = features_scaled[-seq_length:]
            print(f"\n注意: 提供的期数({actual_periods})多于模型需要的期数({seq_length})，只使用最近{seq_length}期")
        
        # 准备输入序列
        sequence = features_scaled.reshape(1, seq_length, -1)
        print(f"使用 {seq_length} 期数据进行预测")
        
        # 预测
        predictions = self.model.predict(sequence, verbose=0)
        
        # 显示预测概率信息（仅分类模式）
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
        
        # 处理预测结果（与predict_next相同的逻辑）
        if self.use_classification:
            # 分类模式：红球是概率分布，蓝球也是概率分布
            red_balls = []
            
            # 处理6个红球位置
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
            
            # 如果少于6个，从概率分布中补充
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
            
            # 处理蓝球（分类值，16个类别）
            blue_prob_dist = predictions[6][0]
            
            if use_probability_sampling:
                blue_ball = np.random.choice(np.arange(1, 17), p=blue_prob_dist)
            else:
                blue_ball = np.argmax(blue_prob_dist) + 1
            
            # 显示蓝球预测信息
            print(f"\n蓝球预测信息:")
            top5_indices = np.argsort(blue_prob_dist)[-5:][::-1]
            top5_probs = blue_prob_dist[top5_indices]
            print(f"  前5候选: ", end="")
            for idx, prob in zip(top5_indices, top5_probs):
                print(f"{idx+1}({prob:.3f}) ", end="")
            print()
            print(f"  最终预测: {blue_ball} (概率: {blue_prob_dist[blue_ball-1]:.3f})")
            
        else:
            # 回归模式（保持向后兼容）
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
    
    def predict_next(self, use_last_n_periods=10, use_probability_sampling=False, random_seed=None, use_latest_data=True):
        """
        预测下一期号码
        use_probability_sampling: 如果True，使用概率采样（结果会随机）；如果False，使用argmax（确定性结果）
        random_seed: 随机种子，如果设置则每次预测结果相同（仅在use_probability_sampling=True时有效）
        use_latest_data: 如果True，从CSV文件读取最新数据；如果False，使用processed_data.npz中的旧数据
        """
        if self.model is None:
            self.load_model()
        
        # 设置随机种子（如果提供）
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 加载数据
        if use_latest_data and os.path.exists(self.data_file):
            # 从CSV文件读取最新数据
            print("从CSV文件读取最新数据...")
            df = pd.read_csv(self.data_file, encoding='utf-8-sig')
            df = df.sort_values('期号', ascending=True)  # 按时间顺序排序
            
            # 显示使用的数据信息
            print(f"数据总期数: {len(df)}")
            if len(df) > 0:
                print(f"最新期号: {df.iloc[-1]['期号']}")
                latest_red = [df.iloc[-1][f'红球{i+1}'] for i in range(6)]
                latest_blue = df.iloc[-1]['蓝球']
                print(f"最新开奖: 红球{latest_red}, 蓝球{latest_blue}")
            
            # 准备特征
            features = self._prepare_features_from_csv(df)
            
            # 加载scaler并标准化
            if not os.path.exists(self.scaler_file):
                raise FileNotFoundError(f"Scaler文件 {self.scaler_file} 不存在，请先运行 data_processor.py")
            
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            features_scaled = scaler.transform(features)
            
            # 从processed_data.npz获取seq_length（模型训练时使用的序列长度）
            if os.path.exists(self.processed_data_file):
                data = np.load(self.processed_data_file, allow_pickle=True)
                model_seq_length = int(data['seq_length'])  # 模型训练时的序列长度
            else:
                model_seq_length = 10  # 默认值
            
            # 显示用于预测的最后几期数据（反标准化后）
            # 显示用户请求的期数，但模型输入使用训练时的长度
            display_periods = min(use_last_n_periods, len(features_scaled))
            seq_length = model_seq_length  # 模型输入长度（必须匹配训练时）
            
            if use_last_n_periods > model_seq_length:
                print(f"\n注意: 模型训练时使用 {model_seq_length} 期数据")
                print(f"      将显示最近 {display_periods} 期数据，但模型输入使用最近 {seq_length} 期")
                print(f"      如需使用 {use_last_n_periods} 期进行预测，请重新训练模型（修改 data_processor.py 中的 seq_length 参数）")
            
            print(f"\n用于预测的最后 {display_periods} 期数据:")
            for i in range(max(0, len(features_scaled) - display_periods), len(features_scaled)):
                dummy = np.zeros((1, scaler.n_features_in_))
                dummy[0, :7] = features_scaled[i, :7]
                original = scaler.inverse_transform(dummy)[0]
                red = [int(round(original[j])) for j in range(6)]
                blue = int(round(original[6]))
                if i < len(df):
                    period = df.iloc[i]['期号'] if '期号' in df.columns else f"第{i+1}期"
                    print(f"  {period}: 红球{red}, 蓝球{blue}")
        else:
            # 使用processed_data.npz中的旧数据
            if not os.path.exists(self.processed_data_file):
                raise FileNotFoundError(f"处理后的数据文件不存在，请先运行 data_processor.py")
            data = np.load(self.processed_data_file, allow_pickle=True)
            features_scaled = data['features_scaled']
            seq_length = int(data['seq_length'])
            print(f"使用processed_data.npz中的旧数据，数据量: {len(features_scaled)}")
        
        # 使用最近的数据
        # 确保不超过可用数据量
        actual_seq_length = min(seq_length, len(features_scaled))
        sequence = features_scaled[-actual_seq_length:].reshape(1, actual_seq_length, -1)
        
        # 如果实际使用的期数少于请求的期数，需要填充
        if actual_seq_length < seq_length:
            # 用最早的数据填充到seq_length长度
            padding_needed = seq_length - actual_seq_length
            padding = np.tile(features_scaled[0:1], (1, padding_needed, 1))
            sequence = np.concatenate([padding, sequence], axis=1)
            print(f"警告: 数据不足，使用 {actual_seq_length} 期数据 + {padding_needed} 期填充")
        else:
            print(f"使用最近 {seq_length} 期数据进行预测（总数据量: {len(features_scaled)}）")
        
        # 预测
        predictions = self.model.predict(sequence, verbose=0)
        
        # 显示预测概率信息（仅分类模式）
        if self.use_classification:
            print("\n预测概率信息:")
            for i in range(6):
                prob_dist = predictions[i][0]
                top5_indices = np.argsort(prob_dist)[-5:][::-1]  # 前5个最可能的号码
                top5_probs = prob_dist[top5_indices]
                print(f"  红球位置{i+1} 前5候选: ", end="")
                for idx, prob in zip(top5_indices, top5_probs):
                    print(f"{idx+1}({prob:.3f}) ", end="")
                print()
        
        if self.use_classification:
            # 分类模式：红球是概率分布，蓝球是回归值
            red_balls = []
            
            # 处理6个红球位置
            for i in range(6):
                prob_dist = predictions[i][0]  # 33个类别的概率分布
                
                if use_probability_sampling:
                    # 使用概率采样，但排除已选择的号码
                    # 创建候选池（排除已选号码）
                    available_probs = prob_dist.copy()
                    available_indices = np.arange(1, 34)  # 1-33
                    
                    # 排除已选择的号码
                    for selected in red_balls:
                        if selected in available_indices:
                            idx = selected - 1
                            available_probs[idx] = 0.0
                    
                    # 归一化概率
                    if available_probs.sum() > 0:
                        available_probs = available_probs / available_probs.sum()
                        # 采样
                        ball = np.random.choice(available_indices, p=available_probs)
                    else:
                        # 如果所有概率都为0，使用argmax
                        ball = np.argmax(prob_dist) + 1
                else:
                    # 使用argmax（确定性），但排除已选择的号码
                    prob_dist_copy = prob_dist.copy()
                    for selected in red_balls:
                        if 1 <= selected <= 33:
                            prob_dist_copy[selected - 1] = -np.inf  # 标记为已选，使用-inf确保不会被选中
                    ball = np.argmax(prob_dist_copy) + 1
                
                red_balls.append(int(ball))
            
            # 确保红球不重复且排序
            red_balls = sorted(list(set(red_balls)))
            
            # 如果少于6个，从概率分布中补充
            while len(red_balls) < 6:
                # 合并所有位置的概率分布（取平均）
                all_probs = np.mean([predictions[i][0] for i in range(6)], axis=0)
                # 排除已选号码
                for selected in red_balls:
                    if 1 <= selected <= 33:
                        all_probs[selected - 1] = 0.0
                # 归一化
                if all_probs.sum() > 0:
                    all_probs = all_probs / all_probs.sum()
                    if use_probability_sampling:
                        new_ball = np.random.choice(np.arange(1, 34), p=all_probs)
                    else:
                        # 确定性选择：选择概率最高的
                        new_ball = np.argmax(all_probs) + 1
                else:
                    # 如果所有概率都为0，按顺序选择未选中的号码
                    available = [x for x in range(1, 34) if x not in red_balls]
                    if available:
                        new_ball = available[0]  # 确定性选择第一个
                    else:
                        new_ball = 1  # 兜底
                
                if new_ball not in red_balls:
                    red_balls.append(int(new_ball))
            
            red_balls = sorted(red_balls[:6])
            
            # 处理蓝球（分类值，16个类别）
            # predictions[6]是蓝球输出，形状应该是(1, 16) - 16个类别的概率分布
            blue_prob_dist = predictions[6][0]  # 16个类别的概率分布（索引0-15对应号码1-16）
            
            if use_probability_sampling:
                # 使用概率采样
                blue_ball = np.random.choice(np.arange(1, 17), p=blue_prob_dist)  # 1-16
            else:
                # 使用argmax（确定性）
                blue_ball = np.argmax(blue_prob_dist) + 1  # 索引0对应号码1，索引15对应号码16
            
            # 显示蓝球预测信息
            print(f"\n蓝球预测信息:")
            top5_indices = np.argsort(blue_prob_dist)[-5:][::-1]  # 前5个最可能的号码
            top5_probs = blue_prob_dist[top5_indices]
            print(f"  前5候选: ", end="")
            for idx, prob in zip(top5_indices, top5_probs):
                print(f"{idx+1}({prob:.3f}) ", end="")
            print()
            print(f"  最终预测: {blue_ball} (概率: {blue_prob_dist[blue_ball-1]:.3f})")
            
        else:
            # 原来的回归模式
            prediction_scaled = predictions[0] if isinstance(predictions, list) else predictions
            
            # 反标准化（只反标准化号码部分）
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # 创建完整特征向量用于反标准化
            dummy_features = np.zeros((1, scaler.n_features_in_))
            dummy_features[0, :7] = prediction_scaled[0]  # 前7个是号码
            
            # 反标准化
            prediction_original = scaler.inverse_transform(dummy_features)[0]
            
            # 提取红球和蓝球
            red_balls = prediction_original[:6]
            blue_ball_original = prediction_original[6]
            
            # 处理红球：确保在1-33范围内（保持浮点数处理）
            red_balls = np.clip(red_balls, 1, 33)
            
            # 最后预测时四舍五入转换为整数
            red_balls = np.round(red_balls).astype(int)
            # 去重
            red_balls = np.unique(red_balls)
            
            # 如果少于6个，按顺序补充未选中的号码（确定性）
            while len(red_balls) < 6:
                available = [x for x in range(1, 34) if x not in red_balls]
                if available:
                    new_ball = available[0]  # 确定性选择第一个未选中的
                    red_balls = np.append(red_balls, new_ball)
                else:
                    break
            
            red_balls = np.sort(red_balls[:6])
            
            # 处理蓝球：使用scaler的min_和scale_直接反标准化（更准确）
            blue_min = scaler.data_min_[6]
            blue_max = scaler.data_max_[6]
            # 如果反标准化后的值不合理，使用直接计算
            if blue_ball_original > 20 or blue_ball_original < 0:
                blue_ball_scaled = prediction_scaled[0][6]
                blue_ball_original = blue_ball_scaled * (blue_max - blue_min) + blue_min
            
            blue_ball = int(np.round(np.clip(blue_ball_original, 1, 16)))
        
        result = {
            '红球': red_balls.tolist() if isinstance(red_balls, np.ndarray) else red_balls,
            '蓝球': blue_ball
        }
        
        # 如果是分类模式，添加概率信息
        if self.use_classification:
            result['预测概率'] = {}
            for i in range(6):
                prob_dist = predictions[i][0]
                selected_idx = red_balls[i] - 1 if i < len(red_balls) else None
                if selected_idx is not None and 0 <= selected_idx < 33:
                    result['预测概率'][f'红球{i+1}'] = {
                        '选中号码': red_balls[i],
                        '选中概率': float(prob_dist[selected_idx]),
                        '最高概率号码': int(np.argmax(prob_dist) + 1),
                        '最高概率': float(np.max(prob_dist))
                    }
        
        return result


if __name__ == "__main__":
    model = SSQLSTMModel()
    
    # 训练模型
    print("开始训练...")
    model.train(epochs=200, batch_size=32)
    
    # 预测下一期
    print("\n预测下一期号码:")
    prediction = model.predict_next()
    print(f"红球: {prediction['红球']}")
    print(f"蓝球: {prediction['蓝球']}")

