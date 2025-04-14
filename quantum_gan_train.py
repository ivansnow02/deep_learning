#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子Wasserstein生成对抗网络(Quantum WGAN-GP)训练脚本，带有TensorBoard监控
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

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保可重复性
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 确保CUDA上下文正确初始化
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    # 打印CUDA内存信息
    print(
        f"CUDA可用内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
    )
    print(f"已使用内存: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
else:
    device = torch.device("cpu")

# 创建TensorBoard日志目录
log_dir = os.path.join(
    "runs", f"quantum_wgan_gp_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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

        # 数据归一化到[-1, 1]而非[0, 1]，更适合WGAN
        self.images = (self.images / 8.0) - 1.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]  # 已归一化到[-1, 1]
        image = np.array(image, dtype=np.float32).reshape(8, 8)

        if self.transform:
            image = self.transform(image)

        # 返回图像和标签
        return image, int(self.labels[idx])


# 定义 Minibatch Discrimination 层
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, intermediate_features=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features

        # 可学习的变换张量 T
        self.T = nn.Parameter(
            torch.Tensor(in_features, out_features * intermediate_features)
        )
        nn.init.normal_(self.T, 0, 1)  # 使用正态分布初始化

    def forward(self, x):
        # x shape: (batch_size, in_features)
        batch_size = x.size(0)

        # 计算 M = x * T
        # M shape: (batch_size, out_features * intermediate_features)
        M = x.mm(self.T)

        # 将 M reshape 为 (batch_size, out_features, intermediate_features)
        M = M.view(batch_size, self.out_features, self.intermediate_features)

        # 计算 L1 距离
        # M_expanded_1 shape: (batch_size, batch_size, out_features, intermediate_features)
        M_expanded_1 = M.unsqueeze(1).expand(
            batch_size, batch_size, self.out_features, self.intermediate_features
        )
        # M_expanded_2 shape: (batch_size, batch_size, out_features, intermediate_features)
        M_expanded_2 = M.unsqueeze(0).expand(
            batch_size, batch_size, self.out_features, self.intermediate_features
        )

        # L1 距离 shape: (batch_size, batch_size, out_features)
        l1_dist = torch.sum(torch.abs(M_expanded_1 - M_expanded_2), dim=3)

        # 计算相似度 c_b(x_i, x_j) = exp(-l1_dist)
        # similarity shape: (batch_size, batch_size, out_features)
        similarity = torch.exp(-l1_dist)

        # 计算 o(x_i)_b = sum_{j=1..n, j!=i} c_b(x_i, x_j)
        # o_b shape: (batch_size, out_features)
        # 我们需要从总和中减去自身与自身的相似度（即对角线元素，其L1距离为0，指数为1）
        o_b = torch.sum(similarity, dim=1) - torch.exp(
            torch.zeros_like(l1_dist[:, :, 0])
        )  # 减去对角线元素 (exp(0)=1)

        # 将 o_b 与原始输入 x 连接
        # x shape: (batch_size, in_features)
        # o_b shape: (batch_size, out_features)
        # combined shape: (batch_size, in_features + out_features)
        combined = torch.cat([x, o_b], dim=1)

        return combined


# 定义Wasserstein GAN的Critic网络（不是判别器）
class Critic(nn.Module):
    """Wasserstein GAN中的评论家网络，带有Minibatch Discrimination"""

    def __init__(
        self,
        image_size=8,
        dropout_rate=0.3,
        mb_out_features=5,
        mb_intermediate_features=16,
    ):
        super().__init__()

        # 定义 Minibatch Discrimination 层的输出维度
        self.mb_out_features = mb_out_features

        # 定义模型主体部分（不包括最后的线性层和 Minibatch Discrimination）
        self.features = nn.Sequential(
            # 输入到第一个隐藏层
            nn.Linear(image_size * image_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            # 第一个隐藏层到第二个
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            # 第二个隐藏层到第三个
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
        )

        # 定义 Minibatch Discrimination 层
        self.minibatch_discrimination = MinibatchDiscrimination(
            in_features=32,  # 输入来自 features 模块的输出
            out_features=self.mb_out_features,
            intermediate_features=mb_intermediate_features,
        )

        # 定义最后的线性层
        # 输入维度是 features 输出维度 (32) + Minibatch Discrimination 输出维度 (mb_out_features)
        self.final_layer = nn.Linear(32 + self.mb_out_features, 1)

        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用正交初始化，有助于梯度流动
            nn.init.orthogonal_(m.weight.data, 0.8)
            if m.bias is not None:
                m.bias.data.fill_(0)
        # 初始化 Minibatch Discrimination 层的参数 (如果需要特定初始化)
        # elif isinstance(m, MinibatchDiscrimination):
        #     nn.init.normal_(m.T, 0, 0.02) # 示例：不同的初始化

    def forward(self, x):
        # 通过模型主体部分
        features_out = self.features(x)
        # 通过 Minibatch Discrimination 层
        mb_out = self.minibatch_discrimination(features_out)
        # 通过最后的线性层
        output = self.final_layer(mb_out)
        return output


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

        # 初始化latent向量 - 使用RX和RY门创建更丰富的初始态
        for i in range(n_qubits):
            qml.Hadamard(wires=i)  # 创建叠加态
            qml.RY(noise[i], wires=i)
            if i % 2 == 0:  # 在偶数量子比特上添加RX门
                qml.RX(noise[(i + 1) % n_qubits], wires=i)

        # 重复参数化层
        for i in range(q_depth):
            # 参数化旋转层
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)

            # 添加RZ门增加表达能力
            for y in range(n_qubits):
                qml.RZ(weights[i][(y + 1) % n_qubits], wires=y)

            # 纠缠层 - 使用CZ门创建纠缠
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

            # 添加额外的CNOT门增强纠缠
            if i % 2 == 0:
                for y in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[y, (y + 1) % n_qubits])
            else:
                for y in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[y, (y + 1) % n_qubits])

        return qml.probs(wires=list(range(n_qubits)))

    # 用于非线性变换
    def partial_measure(noise, weights):
        # 非线性变换
        probs = quantum_circuit(noise, weights)
        probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
        # 确保概率和为1
        probsgiven0 /= torch.sum(probsgiven0)

        # 后处理 - 映射到[-1,1]范围而不是[0,1]，更适合WGAN
        probsgiven = 2 * probsgiven0 - 1
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

        # 使用改进的初始化 - 使用更小的初始范围
        self.q_params = nn.ParameterList([
            nn.Parameter(
                q_delta * (2 * torch.rand(q_depth * n_qubits) - 1), requires_grad=True
            )
            for _ in range(n_generators)
        ])

        # 添加后处理层，提高生成图像质量
        self.post_process = nn.Sequential(
            nn.Linear(2 ** (n_qubits - n_a_qubits) * n_generators, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.Tanh(),  # 输出范围[-1,1]
        )

    def forward(self, x):
        # 每个子生成器输出的大小
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits)

        # 创建张量以"捕获"来自for循环的一批图像
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

        # 应用后处理层
        processed_images = self.post_process(images)

        # 重新调整大小为8x8图像（假设输出大小为64）
        return processed_images.reshape(x.size(0), 8, 8).reshape(x.size(0), -1)


# 梯度惩罚函数，WGAN-GP的核心
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算梯度惩罚项"""
    # 随机权重项，用于两个样本的线性组合
    alpha = torch.rand((real_samples.size(0), 1), device=device)

    # 获取真实样本和生成样本间的线性插值
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    # 计算critic对插值样本的评分
    d_interpolates = critic(interpolates)

    # 创建全1张量以计算梯度
    fake = torch.ones(real_samples.size(0), 1, device=device)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 计算梯度的范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# 保存生成的图像用于评估
def save_generated_images(images, filename):
    """保存生成的图像到文件"""
    fig = plt.figure(figsize=(8, 2))

    for i, image in enumerate(images):
        ax = plt.subplot(1, 8, i + 1)
        plt.axis("off")
        # 将[-1,1]范围转换回[0,1]以正确显示
        img_display = (image.reshape(8, 8).cpu().numpy() + 1) / 2
        plt.imshow(img_display, cmap="gray", vmin=0, vmax=1)

    plt.savefig(filename)
    plt.close(fig)
    return fig


def train(config):
    """
    训练量子WGAN-GP

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
    lrC = config["lrC"]
    num_iter = config["num_iter"]
    n_critic = config["n_critic"]
    lambda_gp = config["lambda_gp"]

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
        # 转换回[0,1]范围显示
        img_display = (image.numpy() + 1) / 2
        plt.subplot(1, 8, i + 1)
        plt.axis("off")
        plt.imshow(img_display, cmap="gray", vmin=0, vmax=1)
    plt.savefig(os.path.join(log_dir, "real_samples.png"))
    plt.close()

    # 设置量子生成器组件
    partial_measure, device = setup_quantum_generator(
        n_qubits, n_a_qubits, q_depth, n_generators
    )

    # 初始化模型
    critic = Critic(image_size=image_size).to(device)
    generator = PatchQuantumGenerator(
        n_qubits, n_a_qubits, q_depth, n_generators, device, partial_measure
    ).to(device)

    # 添加模型图到TensorBoard
    dummy_input = torch.rand(batch_size, image_size * image_size).to(device)
    writer.add_graph(critic, dummy_input)

    # 创建优化器 - 使用Adam优化器
    optC = optim.Adam(critic.parameters(), lr=lrC, betas=(0.5, 0.9))
    optG = optim.Adam(generator.parameters(), lr=lrG, betas=(0.5, 0.9))

    # 学习率调度器
    schedulerC = optim.lr_scheduler.CosineAnnealingLR(optC, num_iter)
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optG, num_iter)

    # 固定噪声，用于可视化训练过程中生成的图像
    fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

    # 训练循环
    print("开始训练...")
    global_step = 0
    generator_iters = 0  # 跟踪生成器训练迭代次数

    critic_losses = []
    generator_losses = []

    # 归一化为[-1, 1]的数据趋势
    print("将数据归一化至[-1, 1]范围以提高WGAN性能")

    while global_step < num_iter:
        for i, (data, _) in enumerate(dataloader):
            # 准备数据
            data = data.reshape(-1, image_size * image_size)
            real_data = data.to(device)

            # ---------------------
            # 训练Critic (n_critic次)
            # ---------------------
            for _ in range(n_critic):
                # 生成随机噪声作为生成器输入
                noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2

                critic.zero_grad()

                # 生成假数据
                fake_data = generator(noise)

                # 计算Critic对真实数据和生成数据的评分
                critic_real = critic(real_data).mean()
                critic_fake = critic(fake_data.detach()).mean()

                # 计算梯度惩罚
                gradient_penalty = compute_gradient_penalty(
                    critic, real_data, fake_data.detach(), device
                )

                # 损失是真实数据和生成数据的分数之差，并添加梯度惩罚
                # 我们的目标是最大化真实分数和最小化假分数
                critic_loss = -critic_real + critic_fake + lambda_gp * gradient_penalty

                # 反向传播和优化
                critic_loss.backward()
                optC.step()

                critic_losses.append(critic_loss.item())

            # 每n_critic次critic迭代后，训练一次生成器
            if global_step % n_critic == 0:
                # ---------------------
                # 训练生成器
                # ---------------------
                generator.zero_grad()

                # 生成新的假数据
                noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
                fake_data = generator(noise)

                # 计算critic对生成数据的评分
                gen_score = critic(fake_data).mean()

                # 生成器的目标是最大化评分（即让critic将生成数据当作真实数据）
                # 由于我们使用梯度下降，所以要最小化负得分
                generator_loss = -gen_score

                # 反向传播和优化
                generator_loss.backward()
                optG.step()

                generator_losses.append(generator_loss.item())
                generator_iters += 1

                # 调整学习率
                schedulerC.step()
                schedulerG.step()

                # 显示训练进度
                if generator_iters % 10 == 0:
                    print(
                        f"进度: {global_step}/{num_iter}, "
                        f"Critic损失: {critic_loss.item():0.3f}, 生成器损失: {generator_loss.item():0.3f}, "
                        f"C(x): {critic_real.item():0.3f}, C(G(z)): {gen_score.item():0.3f}"
                    )

                    # 记录损失和其他指标到TensorBoard
                    writer.add_scalar("Loss/Critic", critic_loss.item(), global_step)
                    writer.add_scalar(
                        "Loss/Generator", generator_loss.item(), global_step
                    )
                    writer.add_scalar(
                        "Metrics/Critic_real", critic_real.item(), global_step
                    )
                    writer.add_scalar(
                        "Metrics/Critic_fake", gen_score.item(), global_step
                    )
                    writer.add_scalar(
                        "Metrics/Gradient_penalty", gradient_penalty.item(), global_step
                    )
                    writer.add_scalar(
                        "LearningRate/Critic", schedulerC.get_last_lr()[0], global_step
                    )
                    writer.add_scalar(
                        "LearningRate/Generator",
                        schedulerG.get_last_lr()[0],
                        global_step,
                    )

                # 每50次迭代生成测试图像
                if generator_iters % 10 == 0:
                    with torch.no_grad():
                        test_images = generator(fixed_noise)
                        test_images_2d = (
                            test_images.view(8, 1, image_size, image_size) + 1
                        ) / 2  # 转换到[0,1]显示

                    # 保存图像到文件
                    img_path = os.path.join(
                        log_dir, f"generated_images_iter_{global_step}.png"
                    )
                    save_generated_images(test_images, img_path)

                    # 添加到TensorBoard
                    img_grid = torchvision.utils.make_grid(
                        test_images_2d, normalize=True
                    )
                    writer.add_image(
                        f"Generated Images/Step {global_step}", img_grid, global_step
                    )

                # 每100次生成器迭代保存检查点
                if generator_iters % 20 == 0:
                    checkpoint_path = os.path.join(
                        log_dir, f"checkpoint_step_{global_step}.pt"
                    )
                    torch.save(
                        {
                            "global_step": global_step,
                            "generator_state_dict": generator.state_dict(),
                            "critic_state_dict": critic.state_dict(),
                            "generator_optimizer": optG.state_dict(),
                            "critic_optimizer": optC.state_dict(),
                        },
                        checkpoint_path,
                    )

            global_step += 1
            if global_step >= num_iter:
                break

    # 训练完成，保存最终模型
    final_model_path = os.path.join(log_dir, "final_model.pt")
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": config,
        },
        final_model_path,
    )

    # 保存损失曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(critic_losses, label="Critic Loss")
    plt.plot(generator_losses, label="Generator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.savefig(os.path.join(log_dir, "training_losses.png"))
    plt.close()

    print(f"训练完成! 最终模型保存在 {final_model_path}")
    print(f"TensorBoard日志保存在 {log_dir}")

    writer.close()


if __name__ == "__main__":
    # 配置参数
    # config = {
    #     "image_size": 8,                 # 图像大小（8x8）
    #     "batch_size": 4,                 # 增加批量大小以提高训练稳定性
    #     "digit_label": 3,                # 要生成的数字（此处为3）
    #     "n_qubits": 5,                   # 量子位总数
    #     "n_a_qubits": 1,                 # 辅助量子位数
    #     "q_depth": 6,                    # 增加参数化量子电路的深度
    #     "n_generators": 4,               # 分块方法中使用的子生成器数量
    #     "lrG": 5e-5,                     # 生成器学习率降低
    #     "lrC": 1e-4,                     # Critic学习率
    #     "num_iter": 1500,                # 增加训练迭代次数
    #     "n_critic": 5,                   # 每训练一次生成器，训练critic的次数
    #     "lambda_gp": 10,                 # 梯度惩罚系数
    # }
    # 修改配置参数以尝试缓解模式坍塌
    config = {
        "image_size": 8,
        "batch_size": 8,
        "digit_label": 4,
        "n_qubits": 4,  # 减少量子位数
        "n_a_qubits": 1,
        "q_depth": 4,  # 减少电路深度
        "n_generators": 2,  # 减少生成器数量
        "lrG": 1e-4,
        "lrC": 1e-4,
        "num_iter": 1200,
        "n_critic": 3,  # 减少Critic更新频率
        "lambda_gp": 1,  # 调整梯度惩罚系数
    }

    train(config)
