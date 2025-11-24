#!/bin/bash
# cuDNN安装脚本
# 注意：需要先下载cuDNN文件

echo "=========================================="
echo "cuDNN安装脚本"
echo "=========================================="
echo ""

# 检查CUDA路径
CUDA_PATH="/usr/local/cuda-11.4"
if [ ! -d "$CUDA_PATH" ]; then
    echo "错误: 未找到CUDA路径 $CUDA_PATH"
    exit 1
fi

echo "CUDA路径: $CUDA_PATH"
echo ""

# 检查是否提供了cuDNN压缩包路径
if [ -z "$1" ]; then
    echo "使用方法:"
    echo "  $0 <cudnn_tar_gz_path>"
    echo ""
    echo "例如:"
    echo "  $0 ~/Downloads/cudnn-11.4-linux-x64-v8.9.7.29.tgz"
    echo ""
    echo "下载地址: https://developer.nvidia.com/cudnn"
    echo "需要注册NVIDIA账号并下载对应CUDA 11.4的cuDNN 8.x版本"
    exit 1
fi

CUDNN_FILE="$1"

if [ ! -f "$CUDNN_FILE" ]; then
    echo "错误: 文件不存在: $CUDNN_FILE"
    exit 1
fi

echo "开始安装cuDNN..."
echo ""

# 创建临时目录
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# 解压cuDNN
echo "解压cuDNN文件..."
tar -xzf "$CUDNN_FILE"

# 复制文件
echo "复制cuDNN文件到CUDA目录..."
sudo cp cuda/include/cudnn*.h "$CUDA_PATH/include/" 2>/dev/null || sudo cp include/cudnn*.h "$CUDA_PATH/include/"
sudo cp cuda/lib64/libcudnn* "$CUDA_PATH/lib64/" 2>/dev/null || sudo cp lib64/libcudnn* "$CUDA_PATH/lib64/"

# 设置权限
echo "设置文件权限..."
sudo chmod a+r "$CUDA_PATH/include/cudnn*.h"
sudo chmod a+r "$CUDA_PATH/lib64/libcudnn*"

# 创建符号链接
echo "创建符号链接..."
cd "$CUDA_PATH/lib64"
CUDNN_SO=$(ls libcudnn.so.*.*.* 2>/dev/null | head -1)
if [ -n "$CUDNN_SO" ]; then
    CUDNN_VERSION=$(echo "$CUDNN_SO" | sed 's/libcudnn\.so\.//')
    MAJOR_MINOR=$(echo "$CUDNN_VERSION" | cut -d. -f1-2)
    sudo ln -sf "libcudnn.so.$CUDNN_VERSION" libcudnn.so.$MAJOR_MINOR
    sudo ln -sf "libcudnn.so.$MAJOR_MINOR" libcudnn.so.8
    echo "已创建符号链接: libcudnn.so.8 -> libcudnn.so.$MAJOR_MINOR -> libcudnn.so.$CUDNN_VERSION"
fi

# 清理
cd -
rm -rf "$TMP_DIR"

echo ""
echo "=========================================="
echo "cuDNN安装完成！"
echo "=========================================="
echo ""
echo "验证安装:"
echo "  ls -la $CUDA_PATH/lib64/libcudnn*"
echo ""
echo "测试TensorFlow GPU检测:"
echo "  source activate_tf_env.sh"
echo "  python -c \"import tensorflow as tf; print('GPU设备:', tf.config.list_physical_devices('GPU'))\""
echo ""

