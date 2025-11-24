# GPU支持配置说明

## 当前状态
- ✅ CUDA 11.4 已安装
- ✅ TensorFlow 2.10.0 已安装（兼容CUDA 11.4）
- ✅ Python 3.8 环境已配置
- ❌ cuDNN 未安装（需要安装才能使用GPU）

## 使用环境

### 方法1: 使用激活脚本（推荐）
```bash
cd /home/user/kkde_SSQ/双色球
source activate_tf_env.sh
python train_50.py
```

### 方法2: 手动激活
```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate tf_env
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python train_50.py
```

## 安装cuDNN以启用GPU支持

TensorFlow 2.10.0 需要 cuDNN 8.x。有两种安装方法：

### 方法1: 使用安装脚本（推荐）

1. **下载cuDNN**
   - 访问: https://developer.nvidia.com/cudnn
   - 需要注册NVIDIA账号（免费）
   - 下载对应CUDA 11.4的cuDNN版本（推荐8.6.0或8.7.0）
   - 文件名类似: `cudnn-11.4-linux-x64-v8.9.7.29.tgz`

2. **运行安装脚本**
   ```bash
   cd /home/user/kkde_SSQ/双色球
   ./install_cudnn.sh ~/Downloads/cudnn-11.4-linux-x64-v8.x.x.x.tgz
   ```

3. **验证安装**
   ```bash
   source activate_tf_env.sh
   python -c "import tensorflow as tf; print('GPU设备:', tf.config.list_physical_devices('GPU'))"
   ```

### 方法2: 手动安装

1. **下载cuDNN**（同上）

2. **手动安装**
   ```bash
   # 解压下载的文件
   tar -xzvf cudnn-11.4-linux-x64-v8.x.x.x.tgz
   
   # 复制文件到CUDA目录
   sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.4/include
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.4/lib64
   sudo chmod a+r /usr/local/cuda-11.4/include/cudnn*.h
   sudo chmod a+r /usr/local/cuda-11.4/lib64/libcudnn*
   
   # 创建符号链接
   cd /usr/local/cuda-11.4/lib64
   sudo ln -sf libcudnn.so.8.x.x libcudnn.so.8
   ```

3. **验证安装**（同上）

## 当前使用CPU训练

即使没有cuDNN，TensorFlow也可以正常工作，只是会使用CPU进行训练。训练速度会较慢，但功能完全正常。

## 依赖包

所有依赖已安装在 `tf_env` 环境中：
- tensorflow==2.10.0
- pandas
- scikit-learn
- numpy
- 其他依赖包

