# 双色球LSTM预测系统

使用LSTM神经网络预测双色球下一期号码的完整系统。

## 功能特性

- 自动获取历年双色球历史数据
- 数据预处理和特征工程
- LSTM神经网络模型训练
- 下一期号码预测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 方式一：一键运行（推荐）

```bash
python main.py
```

这将自动执行所有步骤：数据获取 -> 数据预处理 -> 模型训练 -> 预测

### 方式二：分步运行

#### 1. 获取历史数据

```bash
python data_fetcher.py
```

这将从 78500.cn 网站获取双色球历史数据，并保存到 `ssq_history.csv` 文件中。如果网络获取失败，会自动生成示例数据用于测试。

**数据源**: https://m.78500.cn/kaijiang/ssq/

#### 2. 数据预处理

```bash
python data_processor.py
```

这将：
- 加载原始数据
- 提取特征（红球、蓝球、统计特征等）
- 创建时间序列数据
- 数据标准化
- 保存处理后的数据到 `processed_data.npz`

#### 3. 训练LSTM模型

```bash
python lstm_model.py
```

这将：
- 构建LSTM神经网络模型
- 训练模型（默认100个epoch）
- 保存最佳模型到 `ssq_lstm_model.h5`
- 保存训练历史到 `training_history.csv`

#### 4. 预测下一期号码

```bash
python predict.py
```

这将使用训练好的模型预测下一期的红球和蓝球号码。

## 项目结构

```
双色球/
├── main.py              # 主运行脚本（一键运行所有步骤）
├── data_fetcher.py      # 数据获取脚本
├── data_processor.py    # 数据预处理脚本
├── lstm_model.py        # LSTM模型训练脚本
├── predict.py           # 预测脚本
├── requirements.txt     # 依赖包列表
├── README.md            # 项目说明文档
├── ssq_history.csv      # 历史数据（运行后生成）
├── processed_data.npz   # 处理后的数据（运行后生成）
├── scaler.pkl          # 标准化器（运行后生成）
└── ssq_lstm_model.h5   # 训练好的模型（运行后生成）
```

## 模型说明

### 模型架构
- 3层LSTM网络（128 -> 64 -> 32 单元）
- BatchNormalization 和 Dropout 防止过拟合
- 全连接层输出7个值（6个红球 + 1个蓝球）

### 特征工程
- 红球号码（6个）
- 蓝球号码（1个）
- 统计特征：和值、均值、标准差、范围等
- 奇偶比例、大小比例等

### 训练参数
- 优化器：Adam
- 学习率：0.001（自适应调整）
- 损失函数：MSE
- 早停机制：验证集损失15个epoch不下降则停止

## 注意事项

1. **数据获取**：如果网络获取失败，脚本会自动生成示例数据用于测试
2. **预测准确性**：彩票开奖具有随机性，此模型仅用于学习和研究，不保证预测准确性
3. **理性投注**：请理性对待彩票投注，不要过度依赖预测结果

## 改进建议

1. 尝试更多特征工程方法
2. 调整模型架构（增加层数、调整单元数）
3. 使用集成学习方法
4. 尝试其他深度学习模型（GRU、Transformer等）
5. 添加交叉验证提高模型稳定性

