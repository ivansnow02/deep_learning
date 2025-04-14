#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子生成对抗网络(Quantum GAN)训练脚本，带有TensorBoard监控
"""

import math
import os
import random
from datetime import datetime

# 设置matplotlib使用Agg后端，避免Tkinter线程错误
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torchvision
import torchvision.transforms as transforms

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保可重复性
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 创建TensorBoard日志目录
log_dir = os.path.join(
    "runs", f"quantum_gan_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
writer = SummaryWriter(log_dir)

# 导入手写数字数据集
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    print("正在安装ucimlrepo...")
    import subprocess

    subprocess.check_call(["pip", "install", "ucimlrepo"])
    from ucimlrepo import fetch_ucirepo

# 检查PennyLane和GPU支持
try:
    import pennylane as qml
except ImportError:
    print("正在安装pennylane和pennylane-lightning-gpu...")
    import subprocess

    subprocess.check_call(["pip", "install", "pennylane", "pennylane-lightning-gpu"])
    import pennylane as qml

# 获取数据集
print("加载数据集...")
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
X = optical_recognition_of_handwritten_digits.data.features
y = optical_recognition_of_handwritten_digits.data.targets
print(f"数据集形状: {X.shape}, 标签形状: {y.shape}")

if hasattr(X, "values"):
    X = X.values
if hasattr(y, "values"):
    y = y.values


class DigitsDataset(Dataset):
    """PyTorch数据集，用于手写数字识别数据集"""

    def __init__(self, X, y, label=0, transform=None):
        """
        Args:
            X (array-like): 特征数据.
            y (array-like): 目标标签.
            label (int): 要筛选的标签（只保留该数字的图像）.
            transform (callable, optional): 可选的变换应用于样本.
        """
        self.transform = transform

        # 如果是pandas DataFrame，转换为numpy数组
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # 确保y是一维数组
        if len(y.shape) > 1:
            y = y.flatten()

        # 根据标签过滤样本
        mask = y == label

        # 仅沿第一维度（样本）应用掩码
        self.images = X[mask]
        self.labels = np.full(len(self.images), label)  # 所有标签相同(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx] / 16.0  # 归一化
        image = np.array(image, dtype=np.float32).reshape(8, 8)

        if self.transform:
            image = self.transform(image)

        # 返回图像和标签（总是'label'）
        return image, int(self.labels[idx])


# 定义判别器网络
class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(64, 64),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 16),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# 定义量子电路
def setup_quantum_generator(n_qubits=5, n_a_qubits=1, q_depth=6, n_generators=4):
    """设置量子生成器及相关量子电路"""
    # 量子模拟器
    dev = qml.device("lightning.qubit", wires=n_qubits)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    @qml.qnode(dev, diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        weights = weights.reshape(q_depth, n_qubits)

        # 初始化latent向量
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)

        # 重复层
        for i in range(q_depth):
            # 参数化层
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)

            # 控制Z门
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(n_qubits)))

    # 用于非线性变换
    def partial_measure(noise, weights):
        # 非线性变换
        probs = quantum_circuit(noise, weights)
        probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
        probsgiven0 /= torch.sum(probs)

        # 后处理
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven

    return partial_measure, device


# 定义量子生成器
class PatchQuantumGenerator(nn.Module):
    """分块量子生成器"""

    def __init__(
        self,
        n_qubits,
        n_a_qubits,
        q_depth,
        n_generators,
        device,
        partial_measure,
        q_delta=1,
    ):
        """
        Args:
            n_qubits (int): 量子位总数.
            n_a_qubits (int): 辅助量子位数.
            q_depth (int): 参数化量子电路的深度.
            n_generators (int): 分块方法中使用的子生成器数量.
            device (torch.device): 训练设备.
            partial_measure (callable): 量子电路测量函数.
            q_delta (float, optional): 参数初始化的随机分布范围.
        """

        super().__init__()
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.device = device
        self.partial_measure = partial_measure
        self.n_generators = n_generators

        self.q_params = nn.ParameterList([
            nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
            for _ in range(n_generators)
        ])

    def forward(self, x):
        # 每个子生成器输出的大小
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits)

        # 创建张量以"捕获"来自for循环的一批图像。x.size(0)是批量大小
        images = torch.Tensor(x.size(0), 0).to(self.device)

        # 遍历所有子生成器
        for params in self.q_params:
            # 创建张量以"捕获"来自单个子生成器的批量patch
            patches = torch.Tensor(0, patch_size).to(self.device)
            for elem in x:
                q_out = self.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # 每个patch批次与其他patch连接以创建图像批次
            images = torch.cat((images, patches), 1)

        return images


# 保存生成的图像用于评估
def save_generated_images(images, filename):
    """保存生成的图像到文件"""
    fig = plt.figure(figsize=(8, 2))

    for i, image in enumerate(images):
        ax = plt.subplot(1, 8, i + 1)
        plt.axis("off")
        plt.imshow(image.reshape(8, 8).cpu().numpy(), cmap="gray")

    plt.savefig(filename)
    plt.close(fig)
    return fig


def train(config):
    """
    训练量子GAN

    Args:
        config (dict): 配置参数
    """
    # 设置参数
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    digit_label = config["digit_label"]
    n_qubits = config["n_qubits"]
    n_a_qubits = config["n_a_qubits"]
    q_depth = config["q_depth"]
    n_generators = config["n_generators"]
    lrG = config["lrG"]
    lrD = config["lrD"]
    num_iter = config["num_iter"]

    # 设置数据加载器
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = DigitsDataset(X, y, transform=transform, label=digit_label)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # 显示一些样本图像
    plt.figure(figsize=(8, 2))
    for i in range(8):
        image = dataset[i][0].reshape(image_size, image_size)
        plt.subplot(1, 8, i + 1)
        plt.axis("off")
        plt.imshow(image.numpy(), cmap="gray")
    plt.savefig(os.path.join(log_dir, "real_samples.png"))
    plt.close()

    # 设置量子生成器组件
    partial_measure, device = setup_quantum_generator(
        n_qubits, n_a_qubits, q_depth, n_generators
    )

    # 初始化模型
    discriminator = Discriminator().to(device)
    generator = PatchQuantumGenerator(
        n_qubits, n_a_qubits, q_depth, n_generators, device, partial_measure
    ).to(device)

    # 添加模型图到TensorBoard
    dummy_input = torch.rand(batch_size, image_size * image_size).to(device)
    writer.add_graph(discriminator, dummy_input)

    # 创建损失和优化器
    criterion = nn.BCELoss()
    optD = optim.SGD(discriminator.parameters(), lr=lrD)
    optG = optim.SGD(generator.parameters(), lr=lrG)

    real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

    # 固定噪声，用于可视化训练过程中生成的图像
    fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

    # 训练循环
    print("开始训练...")
    counter = 0
    results = []

    while True:
        for i, (data, _) in enumerate(dataloader):
            # 准备数据
            data = data.reshape(-1, image_size * image_size)
            real_data = data.to(device)

            # 随机噪声，均匀分布在[0,pi/2)范围内
            noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2

            # -------------- 训练判别器 --------------
            discriminator.zero_grad()

            # 真实数据的输出
            outD_real = discriminator(real_data).view(-1)
            errD_real = criterion(outD_real, real_labels)

            # 生成假数据
            fake_data = generator(noise)
            outD_fake = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(outD_fake, fake_labels)

            # 反向传播
            errD_real.backward()
            errD_fake.backward()

            errD = errD_real + errD_fake
            optD.step()

            # -------------- 训练生成器 --------------
            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()

            # 记录损失并添加到TensorBoard
            writer.add_scalar("Loss/Discriminator", errD.item(), counter)
            writer.add_scalar("Loss/Generator", errG.item(), counter)

            # 计算判别器对真实和生成数据的平均预测
            writer.add_scalar("Performance/D(x)", outD_real.mean().item(), counter)
            writer.add_scalar("Performance/D(G(z))", outD_fake.mean().item(), counter)

            counter += 1

            # 显示训练进度
            if counter % 10 == 0:
                print(
                    f"迭代次数: {counter}/{num_iter}, "
                    f"判别器损失: {errD.item():0.3f}, 生成器损失: {errG.item():0.3f}, "
                    f"D(x): {outD_real.mean().item():0.3f}, D(G(z)): {outD_fake.mean().item():0.3f}"
                )

                # 生成测试图像
                with torch.no_grad():
                    test_images = generator(fixed_noise)
                    test_images_2d = test_images.view(8, 1, image_size, image_size)

                # 添加生成的图像到TensorBoard
                if counter % 50 == 0:
                    # 保存图像到文件
                    img_path = os.path.join(
                        log_dir, f"generated_images_iter_{counter}.png"
                    )
                    save_generated_images(test_images, img_path)

                    # 添加到TensorBoard
                    img_grid = torchvision.utils.make_grid(
                        test_images_2d, normalize=True
                    )
                    writer.add_image(
                        f"Generated Images/Iteration {counter}", img_grid, counter
                    )

                    # 保存模型检查点
                    checkpoint_path = os.path.join(
                        log_dir, f"checkpoint_iter_{counter}.pt"
                    )
                    torch.save(
                        {
                            "iteration": counter,
                            "generator_state_dict": generator.state_dict(),
                            "discriminator_state_dict": discriminator.state_dict(),
                            "generator_optimizer": optG.state_dict(),
                            "discriminator_optimizer": optD.state_dict(),
                        },
                        checkpoint_path,
                    )

            if counter >= num_iter:
                break

        if counter >= num_iter:
            break

    # 训练完成，保存最终模型
    final_model_path = os.path.join(log_dir, "final_model.pt")
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "config": config,
        },
        final_model_path,
    )

    print(f"训练完成! 最终模型保存在 {final_model_path}")
    print(f"TensorBoard日志保存在 {log_dir}")

    writer.close()


if __name__ == "__main__":
    # 配置参数
    config = {
        "image_size": 8,  # 图像大小（8x8）
        "batch_size": 8,  # 批量大小
        "digit_label": 3,  # 要生成的数字（此处为3）
        "n_qubits": 5,  # 量子位总数
        "n_a_qubits": 1,  # 辅助量子位数
        "q_depth": 4,  # 参数化量子电路的深度
        "n_generators": 4,  # 分块方法中使用的子生成器数量
        "lrG": 0.3,  # 生成器学习率
        "lrD": 0.01,  # 判别器学习率
        "num_iter": 3000,  # 训练迭代次数
    }

    train(config)
