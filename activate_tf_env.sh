#!/bin/bash
# 激活TensorFlow环境并配置CUDA路径

# 激活conda环境
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate tf_env

# 设置CUDA环境变量
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=/usr/local/cuda-11.4/bin:$PATH

# 设置protobuf使用纯Python实现（避免库版本冲突）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "=========================================="
echo "TensorFlow环境已激活"
echo "Python版本: $(python --version)"
echo "TensorFlow版本: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo '未安装')"
echo "=========================================="
echo ""
echo "注意: 要使用GPU，需要安装cuDNN 8.x"
echo "下载地址: https://developer.nvidia.com/cudnn"
echo "安装后，cuDNN库应位于 /usr/local/cuda-11.4/lib64/"
echo ""

