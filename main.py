import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import os

# 导入我们之前写好的模块
from dataset import DIV2KDataset, add_gaussian_noise
from model import UNetDenoise

# ==================== 1. 计算 PSNR 的辅助函数 ====================
def calculate_psnr(img1, img2):
    """
    计算两张图像的峰值信噪比 (PSNR)
    输入参数为值域在 [0, 1] 之间的 PyTorch 张量
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0 # 如果完全一样，返回一个很大的值
    PIXEL_MAX = 1.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

# ==================== 2. 主训练流程 ====================
def main():
    # --- 检查 GPU 是否可用 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的设备: {device}")

    # --- 超参数设置 ---
    batch_size = 16          # 如果显存不够（如报 OOM 错误），请改成 8 或 4
    learning_rate = 1e-4     # 初始学习率
    num_epochs = 20          # 训练轮数（时间紧的话跑 10-20 轮就有初步效果）
    train_sigma = 25.0       # 作业要求：训练时噪声强度固定为 25
    patch_size = 128         # 训练时的裁剪尺寸
    
    # --- 数据集与 DataLoader ---
    print("正在加载数据集...")
    # 请确保路径与你实际解压的文件夹名称一致！
    train_dataset = DIV2KDataset(root_dir='./DIV2K_train_HR', patch_size=patch_size, is_train=True)
    val_dataset = DIV2KDataset(root_dir='./DIV2K_valid_HR', patch_size=patch_size, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 初始化模型、损失函数与优化器 ---
    model = UNetDenoise(in_channels=3, out_channels=3).to(device)
    criterion = nn.L1Loss() # 图像恢复任务中，L1 Loss (MAE) 通常比 MSE 效果更好，边缘更清晰
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 创建保存权重的文件夹 ---
    os.makedirs("checkpoints", exist_ok=True)
    best_psnr = 0.0

    # ==================== 3. 开始 Epoch 循环 ====================
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train() # 设置为训练模式
        epoch_loss = 0.0
        
        for batch_idx, clean_images in enumerate(train_loader):
            clean_images = clean_images.to(device)
            
            # 动态添加高斯噪声 (sigma = 25)
            noisy_images, _ = add_gaussian_noise(clean_images, sigma=train_sigma)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播 (网络输出的是去噪后的图像)
            denoised_images = model(noisy_images)
            
            # 计算损失 (去噪后的图像 vs 干净的原图)
            loss = criterion(denoised_images, clean_images)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 每 10 个 batch 打印一次进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # ==================== 4. 验证阶段 ====================
        model.eval() # 设置为评估模式 (关闭 Dropout 和 BatchNorm 的更新)
        val_psnr = 0.0
        
        with torch.no_grad(): # 验证时不计算梯度，节省显存并加速
            for clean_images in val_loader:
                clean_images = clean_images.to(device)
                
                # 验证时也使用 sigma = 25 测试基础性能
                noisy_images, _ = add_gaussian_noise(clean_images, sigma=train_sigma)
                
                # 模型推理
                denoised_images = model(noisy_images)
                
                # 计算这个 batch 的平均 PSNR
                for i in range(clean_images.size(0)):
                    val_psnr += calculate_psnr(denoised_images[i], clean_images[i])
                    
        # 计算整个验证集的平均 PSNR
        avg_val_psnr = val_psnr / len(val_dataset)
        print(f"===> Epoch [{epoch+1}/{num_epochs}] 结束 | 训练集平均 Loss: {epoch_loss/len(train_loader):.4f} | 验证集平均 PSNR: {avg_val_psnr:.2f} dB")

        # 保存表现最好的模型权重
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), "checkpoints/best_unet_denoise.pth")
            print(f"[*] 发现更高的 PSNR: {best_psnr:.2f} dB, 模型权重已保存！\n")

if __name__ == "__main__":
    main()