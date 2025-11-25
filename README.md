# 双色球LSTM预测系统

基于LSTM神经网络的双色球号码预测系统，支持自动数据获取、模型训练和号码预测。

## 📋 功能特性

- 🔄 **自动数据获取**：支持从多个数据源获取双色球历史开奖数据
- 📊 **数据预处理**：自动特征工程和数据标准化
- 🤖 **LSTM模型训练**：支持CPU和GPU训练，支持不同规模的模型
- 🔮 **号码预测**：基于训练好的模型预测下一期号码
- 🔄 **数据更新**：支持自动获取最新一期数据并更新到历史文件
- 💾 **模型管理**：自动保存最佳模型和训练历史

## 🚀 快速开始

### 环境要求

- Python 3.7+
- TensorFlow 2.10.0+
- 其他依赖见 `requirements.txt`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 一键运行（推荐）

```bash
python main.py
```

这将自动执行所有步骤：数据获取 → 数据预处理 → 模型训练 → 预测

## 📖 详细使用说明

### 1. 数据获取

#### 方式一：从500.com获取（推荐）

```bash
# 获取所有历史数据
python data_fetcher_500_com.py

# 仅获取最新一期并添加到文件顶部
python update_latest_500_com.py
```

**特点：**
- 自动获取所有历史期号
- 支持重试机制，提高成功率
- 自动验证数据有效性
- 最新数据脚本会自动检查期号是否已存在，避免重复添加

#### 方式二：从其他数据源获取

```bash
python data_fetcher.py
```

数据将保存到 `ssq_history.csv` 文件中。

### 2. 数据预处理

#### 标准版本

```bash
python data_processor.py
```

#### 50期版本（用于快速测试）

```bash
python data_processor_50.py
```

**处理内容：**
- 加载原始CSV数据
- 提取特征（红球、蓝球、统计特征等）
- 创建时间序列数据
- 数据标准化
- 保存处理后的数据到 `processed_data.npz`

### 3. 模型训练

#### 标准版本

```bash
python lstm_model.py
```

#### 50期快速训练版本

```bash
python train_50.py
# 或
python lstm_model_50.py
```

**训练参数：**
- 默认100个epoch
- 早停机制（验证集损失15个epoch不下降则停止）
- 自动保存最佳模型
- 保存训练历史到CSV文件

**模型架构：**
- 3层LSTM网络（128 → 64 → 32 单元）
- BatchNormalization 和 Dropout 防止过拟合
- 全连接层输出7个值（6个红球 + 1个蓝球）

### 4. 号码预测

#### 预测下一期

```bash
python predict.py
```

#### 从指定期数开始预测

```bash
python predict_from_periods.py
```

### 5. 更新最新数据

获取最新一期开奖数据并添加到历史文件顶部：

```bash
python update_latest_500_com.py
```

**功能特点：**
- 自动检查期号是否已存在
- 如果已存在，跳过添加
- 如果不存在，下载并插入到文件第一行（表头之后）

## 📁 项目结构

```
双色球/
├── main.py                    # 主运行脚本（一键运行所有步骤）
│
├── 数据获取
│   ├── data_fetcher.py        # 从78500.cn获取数据
│   ├── data_fetcher_500_com.py # 从500.com获取所有历史数据
│   ├── update_latest.py        # 更新最新一期数据（旧版）
│   └── update_latest_500_com.py # 更新最新一期数据（500.com，推荐）
│
├── 数据处理
│   ├── data_processor.py      # 标准数据预处理
│   └── data_processor_50.py   # 50期快速测试版本
│
├── 模型训练
│   ├── lstm_model.py          # 标准LSTM模型训练
│   ├── lstm_model_50.py        # 50期快速训练版本
│   └── train_50.py            # 50期训练脚本
│
├── 预测
│   ├── predict.py             # 预测下一期号码
│   └── predict_from_periods.py # 从指定期数预测
│
├── 配置文件
│   ├── requirements.txt        # Python依赖包列表
│   ├── .gitignore             # Git忽略文件配置
│   └── activate_tf_env.sh     # TensorFlow环境激活脚本
│
├── 文档
│   ├── README.md              # 项目说明文档（本文件）
│   ├── README_GPU.md          # GPU使用说明
│   ├── 使用说明.md            # 中文使用说明
│   └── 快速启用GPU.md         # GPU快速启用指南
│
├── 数据文件（运行后生成）
│   ├── ssq_history.csv        # 历史开奖数据
│   ├── processed_data.npz     # 预处理后的数据
│   ├── processed_data_50.npz  # 50期预处理数据
│   ├── scaler.pkl             # 数据标准化器
│   ├── training_history.csv   # 训练历史记录
│   └── training_history_50.csv # 50期训练历史
│
└── 模型文件（训练后生成）
    ├── ssq_lstm_model.h5      # 标准模型文件
    ├── ssq_lstm_model.weights.h5 # 模型权重
    └── ssq_lstm_model_50.weights.h5 # 50期模型权重
```

## 🔧 模型说明

### 模型架构

- **输入层**：时间序列特征（红球、蓝球、统计特征）
- **LSTM层1**：128单元，返回序列
- **LSTM层2**：64单元，返回序列
- **LSTM层3**：32单元
- **BatchNormalization**：归一化
- **Dropout**：0.3，防止过拟合
- **全连接层**：输出7个值（6个红球 + 1个蓝球）

### 特征工程

- **基础特征**：红球号码（6个）、蓝球号码（1个）
- **统计特征**：和值、均值、标准差、范围、最大值、最小值
- **比例特征**：奇偶比例、大小比例（以17为界）

### 训练参数

- **优化器**：Adam
- **学习率**：0.001（带自适应调整）
- **损失函数**：MSE（均方误差）
- **批次大小**：32
- **验证集比例**：20%
- **早停机制**：验证集损失15个epoch不下降则停止

## 💡 使用建议

### 日常使用流程

1. **首次使用**：运行 `python main.py` 完成初始化和训练
2. **定期更新数据**：运行 `python update_latest_500_com.py` 更新最新开奖数据
3. **重新训练**（可选）：数据更新后可以重新训练模型以提高准确性
4. **预测**：运行 `python predict.py` 获取预测结果

### 快速测试

如果想快速测试系统功能，可以使用50期版本：

```bash
# 使用50期数据快速训练和测试
python data_processor_50.py
python train_50.py
python predict.py
```

## ⚠️ 重要提示

1. **数据获取**：网络获取失败时，部分脚本会自动生成示例数据用于测试
2. **预测准确性**：彩票开奖具有随机性，此模型仅用于学习和研究，**不保证预测准确性**
3. **理性投注**：请理性对待彩票投注，不要过度依赖预测结果
4. **数据更新**：建议定期运行 `update_latest_500_com.py` 更新最新数据
5. **模型训练**：数据更新后建议重新训练模型，以获得更好的预测效果

## 🔄 数据更新流程

推荐的数据更新流程：

```bash
# 1. 获取最新一期数据（自动检查是否已存在）
python update_latest_500_com.py

# 2. 重新预处理数据（如果数据有更新）
python data_processor.py

# 3. 重新训练模型（可选，建议定期执行）
python lstm_model.py

# 4. 进行预测
python predict.py
```

## 🛠️ GPU支持

如果系统有NVIDIA GPU，可以启用GPU加速训练：

1. 安装CUDA和cuDNN（参考 `README_GPU.md`）
2. 运行 `source activate_tf_env.sh` 激活GPU环境
3. 正常训练即可，TensorFlow会自动使用GPU

详细说明请参考：
- `README_GPU.md` - GPU使用说明
- `快速启用GPU.md` - GPU快速启用指南

## 📊 数据格式

CSV文件格式（`ssq_history.csv`）：

```csv
期号,开奖日期,红球1,红球2,红球3,红球4,红球5,红球6,蓝球
2025135,2025-11-23,1,2,5,9,25,32,10
2025134,2025-11-20,3,5,9,13,26,29,12
...
```

## 🔍 故障排除

### 数据获取失败

- 检查网络连接
- 尝试使用不同的数据源（`data_fetcher.py` 或 `data_fetcher_500_com.py`）
- 检查目标网站是否可访问

### 训练失败

- 检查是否有足够的数据（建议至少100期）
- 检查依赖包是否正确安装
- 查看错误日志定位问题

### 预测结果异常

- 确认模型文件是否存在
- 检查数据预处理是否正确完成
- 尝试重新训练模型

## 📝 改进建议

1. **特征工程**：尝试更多统计特征和组合特征
2. **模型架构**：调整LSTM层数和单元数，尝试GRU、Transformer等
3. **集成学习**：使用多个模型进行集成预测
4. **交叉验证**：添加K折交叉验证提高模型稳定性
5. **超参数优化**：使用网格搜索或贝叶斯优化寻找最佳超参数

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**免责声明**：本项目仅用于技术学习和研究，不构成任何投注建议。彩票具有随机性，请理性对待。
