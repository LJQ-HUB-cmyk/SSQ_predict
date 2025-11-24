# 快速启用GPU支持

## 问题
当前TensorFlow无法检测到GPU，因为缺少cuDNN库。

## 解决方案

### 步骤1: 下载cuDNN

1. 访问: https://developer.nvidia.com/cudnn
2. 注册/登录NVIDIA账号（免费）
3. 选择 "Download cuDNN"
4. 选择版本：
   - **CUDA Version**: 11.4
   - **cuDNN Version**: 8.6.0 或 8.7.0 或 8.9.x
5. 下载 Linux x86_64 版本（.tgz文件）

### 步骤2: 安装cuDNN

```bash
cd /home/user/kkde_SSQ/双色球

# 假设下载的文件在 ~/Downloads/ 目录
./install_cudnn.sh ~/Downloads/cudnn-11.4-linux-x64-v8.x.x.x.tgz
```

### 步骤3: 验证GPU

```bash
source activate_tf_env.sh
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPU设备:', gpus); print('GPU数量:', len(gpus))"
```

如果看到GPU设备列表，说明安装成功！

### 步骤4: 运行训练（使用GPU）

```bash
source activate_tf_env.sh
python train_50.py
```

训练时会自动使用GPU加速。

## 常见问题

**Q: 下载cuDNN需要付费吗？**
A: 不需要，注册NVIDIA账号即可免费下载。

**Q: 必须安装cuDNN吗？**
A: 是的，TensorFlow需要cuDNN才能使用GPU。如果不安装，只能使用CPU训练（速度较慢）。

**Q: 安装后还是检测不到GPU？**
A: 检查以下几点：
1. cuDNN版本是否正确（需要8.x）
2. 文件是否复制到正确位置
3. 符号链接是否正确创建
4. 运行 `ldconfig` 更新库缓存

```bash
sudo ldconfig
```

