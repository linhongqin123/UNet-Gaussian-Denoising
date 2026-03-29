import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DIV2KDataset(Dataset):
    """
    DIV2K 数据集加载类。
    用于读取高分辨率图像，并随机裁剪成适合网络输入的 Patch。
    """
    def __init__(self, root_dir, patch_size=128, is_train=True):
        """
        参数:
            root_dir (str): DIV2K 高清图像所在目录的路径。
            patch_size (int): 裁剪的图像块大小，默认为 128。
            is_train (bool): 是否为训练模式，训练模式下会加入数据增强。
        """
        self.root_dir = root_dir
        # 获取目录下所有的 png 文件
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
        # 定义图像预处理流水线
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(patch_size),         # 随机裁剪到指定大小
                transforms.RandomHorizontalFlip(),         # 随机水平翻转 (数据增强)
                transforms.RandomVerticalFlip(),           # 随机垂直翻转 (数据增强)
                transforms.ToTensor()                      # 转换为张量，并归一化到 [0, 1] 范围
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(patch_size),         # 验证集也裁剪，保证输入尺寸一致
                transforms.ToTensor()
            ])

    def __len__(self):
        # 返回数据集大小
        return len(self.image_files)

    def __getitem__(self, idx):
        # 根据索引读取图像
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB') # 确保转换为 RGB 三通道
        
        # 应用预处理，得到干净的图像 (Ground Truth)
        clean_image = self.transform(image)
        return clean_image


def add_gaussian_noise(clean_image, sigma):
    """
    向干净的图像张量添加指定强度的高斯噪声。
    
    参数:
        clean_image (Tensor): 形状为 (C, H, W) 或 (B, C, H, W) 的图像张量，值域为 [0, 1]。
        sigma (float): 噪声的标准差 (例如 15, 25, 35, 50)。
        
    返回:
        noisy_image (Tensor): 加噪后的图像张量，值域限制在 [0, 1]。
    """
    # 生成与 clean_image 形状相同的高斯噪声
    noise = torch.randn_like(clean_image) * (sigma / 255.0)
    
    # 图像加噪
    noisy_image = clean_image + noise
    
    # 【关键步骤】加噪后部分像素值可能会超出 [0, 1] 的范围，必须截断
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    
    return noisy_image, noise # 返回噪声本身，方便后续计算 Residual Loss 