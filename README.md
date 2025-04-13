# 从0开始搭建的 CIFAR-10 三层神经网络图像分类器

本项目实现了一个基于卷积神经网络（CNN）的 CIFAR-10 图像分类模型，使用 CuPy 进行 GPU 加速。项目包括模型定义、数据加载、训练、测试以及超参数搜索功能，模块化设计清晰，支持用户自定义数据路径、输出路径和模型保存路径。

## 项目结构

项目包含以下文件：

- `model.py`: 定义 `ConvNet` 模型类及相关的卷积、池化、批归一化等操作。
- `train.py`: 实现 CIFAR-10 数据加载、模型训练和验证逻辑。
- `test.py`: 实现测试集评估功能。
- `hyperparameter_search.py`: 实现超参数搜索，包括学习率、正则化强度、卷积核数量和 dropout 率。
- `demo.sh`: Bash 脚本，用于简化训练和测试流程，用户只需指定数据路径、输出路径和模型保存路径即可运行整个流程。

## 环境要求

- **操作系统**: Linux（推荐 Ubuntu）或其他支持 Bash 和 Python 的系统。
- **Python 版本**: Python 3.8+
- **依赖库**:
  - `cupy-cudaXX`（根据你的 CUDA 版本安装，例如 `cupy-cuda11x`）
  - `pandas`
  - `matplotlib`
  - `tqdm`
- **硬件**: 支持 CUDA 的 NVIDIA GPU。
- **数据集**: CIFAR-10 数据集（需下载并解压到指定目录）。

安装依赖的示例命令：
```bash
pip install cupy-cuda12x pandas matplotlib tqdm
```

## 数据准备

1. 下载 CIFAR-10 数据集（Python 版本）：
   - 官方网站：[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
   - 或使用脚本自动下载（需安装 `wget` 和 `tar`）：
     ```bash
     mkdir -p ./data
     cd ./data
     wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
     tar -xzvf cifar-10-python.tar.gz
     cd ../..
     ```
   - 解压后的目录结构应包含 `cifar-10-batches-py/`，内有 `data_batch_1` 到 `data_batch_5` 和 `test_batch`。

2. 确保数据集路径正确，例如 `/path/to/cifar-10-batches-py`。

## 使用说明

### 1. 训练模型

本项目通过 `demo.sh` 脚本简化了训练流程。用户只需指定以下三个路径：
- `data_dir`: CIFAR-10 数据集所在的目录（包含 `data_batch_1` 等文件）。
- `output_dir`: 保存训练过程中的图表（损失曲线）和结果文件（超参数表格）的目录。
- `model_dir`: 保存模型权重（`.npz` 文件）的目录。

#### 步骤

1. **克隆代码仓库**：
   ```bash
   git clone https://github.com/RunRiotComeOn/first_cv_assignment20250413.git
   cd your-repo
   ```

2. **确保文件权限**：
   为 `demo.sh` 添加执行权限：
   ```bash
   chmod +x demo.sh
   ```

3. **运行训练**：
   执行以下命令，替换路径为实际路径：
   ```bash
   ./demo.sh /path/to/cifar-10/cifar-10-batches-py /path/to/output /path/to/models
   ```

   - 示例：
     ```bash
     ./demo.sh ./data/cifar-10/cifar-10-batches-py ./output ./models
     ```
   - 脚本会自动创建 `output_dir` 和 `model_dir`（如果不存在），并运行超参数搜索。
   - 超参数搜索包括：
     - 学习率：`[0.1, 0.05, 0.01]`
     - 正则化强度：`[0.01, 0.005, 0.001]`
     - 卷积核数量：`[(64, 32), (32, 32), (32, 16)]`
     - Dropout 率：`[0, 0.3, 0.5, 0.7]`

4. **训练输出**：
   - **控制台**：显示每个 epoch 的训练损失、训练准确率、验证损失和验证准确率，以及超参数组合的结果。
   - **输出目录**（`output_dir`）：
     - `loss_plot_*.png`：每个超参数组合的训练和验证损失曲线。
     - `hyperparameter_results.csv`：超参数搜索结果表格。
     - `dropout_results.csv`：Dropout 率搜索结果表格。
   - **模型目录**（`model_dir`）：
     - `model_*.npz`：每个超参数组合的最佳模型权重。
     - `model_dropout_*.npz`：每个 Dropout 率的最佳模型权重。

### 2. 测试模型

测试过程已集成在 `hyperparameter_search.py` 中，会自动使用最佳超参数和 Dropout 率加载模型权重，并在测试集上评估准确率。无需单独运行测试脚本。

如果需要手动测试某一模型权重，可以修改 `hyperparameter_search.py` 或编写单独的测试脚本，加载指定 `.npz` 文件并调用 `test.py` 中的 `test` 函数。例如：

```python
import cupy as cp
from model import ConvNet
from test import test
from train import load_cifar10_data

# 加载数据
data_dir = "/path/to/cifar-10/cifar-10-batches-py"
_, _, _, _, test_data, test_labels = load_cifar10_data(data_dir)

# 初始化模型
model = ConvNet(num_filters1=32, num_filters2=32, dropout_rate=0.5)

# 加载权重
weights = cp.load("/path/to/models/model_dropout_0.5.npz")
model.w1, model.b1 = weights['w1'], weights['b1']
model.w2, model.b2 = weights['w2'], weights['b2']
model.w3, model.b3 = weights['w3'], weights['b3']
model.gamma1, model.beta1 = weights['gamma1'], weights['beta1']
model.gamma2, model.beta2 = weights['gamma2'], weights['beta2']
model.bn_cache1['running_mean'] = weights['running_mean1']
model.bn_cache1['running_var'] = weights['running_var1']
model.bn_cache2['running_mean'] = weights['running_mean2']
model.bn_cache2['running_var'] = weights['running_var2']

# 测试
test_acc = test(model, test_data, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")
```

### 3. 模型权重下载

训练好的模型权重已上传至百度云盘，供参考和直接使用：

- **百度云盘链接**:
  [https://pan.baidu.com/s/1p48X5sZQNKtiQRHpkxOAgQ?pwd=best](https://pan.baidu.com/s/1p48X5sZQNKtiQRHpkxOAgQ?pwd=best)

**说明**：
- 由于模型权重文件较大，建议下载后放置在 `model_dir` 中（例如 `./models`），并确保路径与 `hyperparameter_search.py` 或自定义测试脚本一致。
- 最佳模型权重为 `model_dropout_0.5.npz`（假设 Dropout 率 0.5 表现最佳，具体请参考训练输出）。

### 4. 示例输出 `example.ipynb`

示例输出有以下功能：

- **数据加载**：从CIFAR-10数据集中加载图像和标签。
- **模型加载**：加载预训练的卷积神经网络模型权重。
- **单图像分析**：
   - 预测图像类别
   - 可视化特征图
   - 显示类别概率分布
- **特征可视化**：可视化卷积层和池化层的特征图。
- **类别概率分布**：显示模型对每个类别的预测概率。

代码结构如下介绍：
1. **`load_cifar10_batch(file)`**  
   - 功能：从CIFAR-10数据文件中加载图像和标签。
   - 输入：数据文件路径。
   - 输出：图像数据（归一化到[0, 1]）和标签。

2. **`visualize_features(features, layer_name)`**  
   - 功能：可视化指定层的所有特征图。
   - 输入：
     - `features`：特征图（CuPy数组，形状为`(n, c, h, w)`）。
     - `layer_name`：层名称（如`'conv1'`）。
   - 输出：特征图的可视化图像。

3. **`analyze_single_image(model, image, true_label, model_file)`**  
   - 功能：对单个图像进行分析，包括预测类别、显示概率分布和可视化特征图。
   - 输入：
     - `model`：卷积神经网络模型。
     - `image`：输入图像。
     - `true_label`：图像的真实标签。
     - `model_file`：模型权重文件路径。
   - 输出：
     - 输入图像
     - 预测类别
     - 特征图可视化
     - 类别概率分布

4. **`main()`**  
   - 功能：主函数，加载数据和模型，调用分析函数。



## 常见问题

1. **Q: 数据集路径错误怎么办？**
   - A: 确保 `data_dir` 指向正确的 CIFAR-10 目录，包含 `data_batch_1` 等文件。检查路径是否拼写正确。

2. **Q: CuPy 报错“CUDA 环境不兼容”？**
   - A: 确认安装的 CuPy 版本与你的 CUDA 版本匹配。例如，CUDA 11.2 则使用 `pip install cupy-cuda112`。

3. **Q: 模型权重文件无法加载？**
   - A: 确保 `.npz` 文件路径正确，且文件未损坏。尝试重新下载或检查文件完整性。

4. **Q: 如何修改超参数范围？**
   - A: 编辑 `hyperparameter_search.py` 中的 `lrs`, `regs`, `filter_sizes`, `dropout_rates` 列表，添加或修改值。

## 联系方式

如有问题，请通过 GitHub Issues 提交，或联系 [23307110412@m.fudan.edu.cn]


