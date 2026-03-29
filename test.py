import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

from dataset import DIV2KDataset, add_gaussian_noise
from model import UNetDenoise

def tensor_to_numpy(tensor):
    """将 PyTorch Tensor 转换为可显示的 numpy 数组"""
    img = tensor.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) # 转换维度为 H, W, C
    img = np.clip(img, 0.0, 1.0)
    return img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的测试设备: {device}")

    # ==================== 1. 加载你刚才跑出来的模型 ====================
    model_path = "checkpoints/best_unet_denoise.pth"
    model = UNetDenoise(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 设置为评估模式
    print("成功加载训练好的权重！")

    # ==================== 2. 准备测试数据 ====================
    # 测试集依然从 DIV2K_valid_HR 里取
    test_dataset = DIV2KDataset(root_dir='./DIV2K_valid_HR', patch_size=256, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 作业要求的测试噪声强度
    test_sigmas = [15.0, 25.0, 35.0, 50.0]
    results = {sigma: {'psnr': 0.0, 'ssim': 0.0} for sigma in test_sigmas}
    os.makedirs("results", exist_ok=True)

    # ==================== 3. 开始测试 ====================
    num_test_images = 10 # 测试前10张图片计算平均值，足以写进报告
    print(f"\n开始进行不同噪声强度下的泛化测试...")

    with torch.no_grad():
        for sigma in test_sigmas:
            epoch_psnr, epoch_ssim = 0.0, 0.0
            
            for i, clean_image in enumerate(test_loader):
                if i >= num_test_images: break
                    
                clean_image = clean_image.to(device)
                # 动态添加不同的测试噪声
                noisy_image, _ = add_gaussian_noise(clean_image, sigma=sigma)
                # 模型去噪
                denoised_image = model(noisy_image)

                clean_np = tensor_to_numpy(clean_image[0])
                noisy_np = tensor_to_numpy(noisy_image[0])
                denoised_np = tensor_to_numpy(denoised_image[0])

                # 计算指标
                cur_psnr = calculate_psnr(clean_np, denoised_np, data_range=1.0)
                cur_ssim = calculate_ssim(clean_np, denoised_np, data_range=1.0, channel_axis=2)
                
                epoch_psnr += cur_psnr
                epoch_ssim += cur_ssim

                # ======= 画图：保存第一张图作为作业的视觉对比图 =======
                if i == 0:
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.title("Original Image")
                    plt.imshow(clean_np)
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.title(f"Noisy Image (Sigma={int(sigma)})")
                    plt.imshow(noisy_np)
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.title(f"Denoised (PSNR: {cur_psnr:.2f}dB)")
                    plt.imshow(denoised_np)
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f"results/comparison_sigma_{int(sigma)}.png", bbox_inches='tight', dpi=150)
                    plt.close()

            results[sigma]['psnr'] = epoch_psnr / num_test_images
            results[sigma]['ssim'] = epoch_ssim / num_test_images

    # ==================== 4. 打印作业要求的表格 ====================
    print("\n" + "="*50)
    print(" 作业提交表格：U-Net 在不同测试噪声下的表现")
    print(" (注意：模型固定在 Sigma=25 下训练)")
    print("="*50)
    print(f"{'测试噪声 (Sigma)':<15} | {'平均 PSNR (dB)':<15} | {'平均 SSIM':<15}")
    print("-" * 50)
    for sigma in test_sigmas:
        print(f"{sigma:<18.0f} | {results[sigma]['psnr']:<17.2f} | {results[sigma]['ssim']:<15.4f}")
    print("="*50)
    print("\n 可视化对比图已生成，请在项目文件夹的 `results/` 目录下查看！")

if __name__ == "__main__":
    main()