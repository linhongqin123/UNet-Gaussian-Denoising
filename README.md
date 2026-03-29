
# ResUNet-Denoise-RGB: 基于 U-Net 的高斯噪声去除

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

本仓库包含一个基于 **U-Net** 架构的卷积神经网络实现，用于图像去噪任务。该模型专门针对 RGB 彩色图像中的加性高斯白噪声（AWGN），并采用**全局残差学习**来加速收敛并提高细节保留能力。

本项目旨在分析卷积神经网络（CNN）在底层视觉任务中的**泛化能力**。模型在固定的噪声强度（$\sigma=25$）下进行严格训练，并在未见过的噪声分布上进行泛化测试。

---

##  核心特性
- **网络架构**: 带有跳跃连接（Skip Connections）的 U-Net，有效保留高频纹理和边缘信息。
- **残差学习**: 网络负责预测噪声残差，而非直接输出干净图像（$去噪图像 = 噪声图像 - 预测的噪声$）。
- **RGB 支持**: 完全支持 3 通道彩色图像的高保真去噪。
- **泛化测试**: 包含完整的自动化测试脚本，用于评估模型在不同噪声强度 $\sigma \in \{15, 25, 35, 50\}$ 下的表现。

---

##  环境依赖与安装

代码使用 Python 编写，核心深度学习框架为 PyTorch。请使用 `pip` 安装所需的依赖库：

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-image pillow
````

*(注意：如果您使用的是 RTX 50 系列等最新架构的顶级显卡，请确保安装 PyTorch 的 Nightly 预览版，或者直接在 Kaggle 等云端环境中运行本代码)。*

-----

##  数据集准备

模型使用高分辨率的 **DIV2K 数据集** 进行训练和评估。

1.  下载 DIV2K 数据集（包含 Train 和 Validation 的高分辨率 HR 图像）。
2.  将数据集解压到本项目的根目录中。
3.  您的项目文件夹结构应如下所示：

<!-- end list -->

```text
ResUNet-Denoise-RGB/
├── DIV2K_train_HR/         # 训练集图片 (800张 png)
├── DIV2K_valid_HR/         # 验证集/测试集图片 (100张 png)
├── dataset.py              # 数据加载与高斯噪声生成模块
├── model.py                # U-Net 网络架构定义
├── main.py                 # 模型训练主脚本
├── test.py                 # 泛化测试与可视化脚本
└── README.md
```

-----

##  使用说明

### 1\. 模型训练

要在 DIV2K 数据集上从头开始训练模型（固定添加 $\sigma=25$ 的噪声）：

```bash
python main.py
```

  * 脚本在训练时会动态地将高分辨率图像随机裁剪为 $128 \times 128$ 的图像块 (patches) 以节省显存。
  * 训练过程中，表现最好的模型权重将自动保存至 `checkpoints/best_unet_denoise.pth`。

### 2\. 测试与可视化评估

要评估训练好的模型在不同噪声强度（$\sigma=15, 25, 35, 50$）下的泛化能力，并自动生成视觉对比图：

```bash
python test.py
```

  * 核心评估指标（PSNR 和 SSIM）将直接打印在终端中。
  * 直观的“原图-加噪图-去噪图”对比图将自动生成并保存在 `results/` 文件夹中。

-----

##  实验结果

下表展示了我们的模型（**仅**在 $\sigma=25$ 的条件下训练）在 DIV2K 验证集上面对不同噪声水平时的泛化性能测试结果：

| 测试噪声强度 ($\sigma$) | 平均 PSNR (dB) | 平均 SSIM |
| :---: | :---: | :---: |
| **15** | 30.21 | 0.8114 |
| **25 (匹配的训练强度)** | 28.70 | 0.7708 |
| **35** | 26.30 | 0.6174 |
| **50** | 21.08 | 0.3690 |

**结果分析：**
模型在其匹配的训练噪声水平 ($\sigma=25$) 以及更低的噪声水平下表现非常稳定且优异。然而，这也印证了单一固定域训练的 CNN 的局限性：当面对极端的分布外强噪声（如 $\sigma=50$）时，模型的去噪性能会出现断崖式下降。这突显了在未来的盲去噪（Blind Denoising）任务中，引入噪声水平图（Noise Level Map）等条件输入特征的必要性。

-----

##  开源协议

本项目开源，遵循 [MIT License](https://www.google.com/search?q=LICENSE) 协议。

```
```
