import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm2d -> ReLU) * 2
    这是 U-Net 中最基础的卷积块，每次特征图分辨率不变或减半时，都会经过两次这样的卷积。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二层卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetDenoise(nn.Module):
    """
    用于图像去噪的 U-Net 模型。
    采用了全局残差学习：网络预测的是噪声，最终输出 = 输入 - 噪声。
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # ==================== 编码器 (Encoder) ====================
        # 提取特征，空间尺寸逐渐缩小，通道数逐渐翻倍
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # ==================== 瓶颈层 (Bottleneck) ====================
        self.bottleneck = DoubleConv(512, 1024)

        # ==================== 解码器 (Decoder) ====================
        # 恢复图像尺寸，通道数逐渐减半，并与编码器特征拼接 (Skip Connections)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512) # 512(来自下采样) + 512(来自上采样) = 1024
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        # ==================== 输出层 ====================
        # 将通道数降维到目标通道数 (RGB为3，灰度图为1)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 保存输入图像，用于最后的残差相减
        input_img = x
        
        # --- 编码过程 ---
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))

        # --- 瓶颈层 ---
        b = self.bottleneck(self.pool4(x4))

        # --- 解码过程 (包含跳跃连接 torch.cat) ---
        d1 = self.up1(b)
        d1 = torch.cat([x4, d1], dim=1) # 跳跃连接：将 x4 拼接到 d1
        d1 = self.up_conv1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x3, d2], dim=1) # 跳跃连接
        d2 = self.up_conv2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([x2, d3], dim=1) # 跳跃连接
        d3 = self.up_conv3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([x1, d4], dim=1) # 跳跃连接
        d4 = self.up_conv4(d4)

        # --- 预测噪声 ---
        predicted_noise = self.out_conv(d4)

        # --- 全局残差学习 ---
        # 去噪后的图像 = 带噪输入 - 网络预测出的噪声
        denoised_img = input_img - predicted_noise
        
        return denoised_img

# ==== 测试网络是否能跑通 (仅作验证，正式运行时可删除) ====
if __name__ == "__main__":
    # 模拟一个 Batch Size 为 4，3通道，128x128 尺寸的输入张量
    dummy_input = torch.randn(4, 3, 128, 128) 
    model = UNetDenoise(in_channels=3, out_channels=3)
    
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") 
    # 如果输出依然是 [4, 3, 128, 128]，说明网络结构没有问题！