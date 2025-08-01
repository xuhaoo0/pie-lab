import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import time
import random
from pytorch_fid.fid_score import calculate_fid_given_paths
from unet import UNet

# 超参数
EPOCHS = 30
GEN_EPOCHS = 5
FID_SAMPLES = 100  # FID评估的样本数量
IMG_SIZE = 32
CHANNELS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 1000  # 加噪步数
BETA_MIN = 1e-4
BETA_MAX = 0.02
BETAS = torch.linspace(BETA_MIN, BETA_MAX, T).to(DEVICE)  # 所有下标t都是: [0, T-1]
ALPHAS = 1. - BETAS
ALPHAS_BAR = torch.cumprod(ALPHAS, dim=0)  # alpha累积
MODIFIED_ALPHAS_BAR = torch.cat([torch.tensor([1.0], device=DEVICE), ALPHAS_BAR[:-1]])  # 去掉ALPHAS_BAR最后一个，前面补1
SQRT_ALPHAS_BAR = torch.sqrt(ALPHAS_BAR)  # train
SQRT_ONE_MINUS_ALPHAS_BAR = torch.sqrt(1 - ALPHAS_BAR)  # train
SQRT_REVERSE_ALPHAS = torch.sqrt(1. / ALPHAS)  # sample
STD = torch.sqrt(BETAS * (1. - MODIFIED_ALPHAS_BAR) / (1. - ALPHAS_BAR))  # sample，恰好首位为0，(variance方差，std标准差)
# 重要调参
BATCH_SIZE = 128
LR = 1e-4
# 利用相对路径创建数据和结果目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RES_DIR = os.path.join(BASE_DIR, 'res')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)



def add_noise(x, t):
    noise = torch.randn_like(x)  # 每个x_i的噪声都不同
    sqrt_alphas_bar = SQRT_ALPHAS_BAR[t][:, None, None, None]  # 扩展为(BATCH_SIZE, 1, 1, 1)
    sqrt_one_minus_alphas_bar = SQRT_ONE_MINUS_ALPHAS_BAR[t][:, None, None, None]
    return sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar * noise, noise


@torch.no_grad()  # 禁用梯度计算
def sample(model, n=16):
    model.eval()
    x = torch.randn((n, CHANNELS, IMG_SIZE, IMG_SIZE)).to(DEVICE)
    for t in reversed(range(0, T)):  # t: [ T-1...0 ]
        time_tensor = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        predicted_noise = model(x, time_tensor)
        beta = BETAS[t]
        sqrt_reverse_alpha = SQRT_REVERSE_ALPHAS[t]
        sqrt_one_minus_alpha_bar = SQRT_ONE_MINUS_ALPHAS_BAR[t]
        mean = sqrt_reverse_alpha * (x - beta / sqrt_one_minus_alpha_bar * predicted_noise)  # 均值
        noise = torch.randn_like(x)
        x = mean + STD[t] * noise  # 设置: STD[0]=0
    x = torch.clamp(x, -1., 1.)  # 通过截断，限制数据在[-1, 1]范围内
    return x


def plot(train_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(RES_DIR, 'train_loss.png'))
    plt.close()


def print_time(begin, end):
    # 总时间
    total_time = end - begin
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"Training time: {minutes}m {seconds}s")



@torch.no_grad()
def calculate_fid(model, dataset, epoch, n_samples=FID_SAMPLES):
    model.eval()
    # 保存生成图和真实图的目录
    gen_dir = os.path.join(RES_DIR, 'fid_generated')
    real_dir = os.path.join(RES_DIR, 'fid_real')
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    # 清空旧图像
    for f in os.listdir(gen_dir):
        os.remove(os.path.join(gen_dir, f))
    for f in os.listdir(real_dir):
        os.remove(os.path.join(real_dir, f))
    # 生成图像
    gen_images = sample(model, n=n_samples)
    gen_images = (gen_images + 1) / 2  # [0,1]
    for i in range(n_samples):
        save_image(gen_images[i], os.path.join(gen_dir, f"{i}.png"))
    # 保存真实图像
    indices = random.sample(range(len(dataset)), n_samples)
    for idx, i in enumerate(indices):
        img, _ = dataset[i]
        img = (img * 0.5 + 0.5)  # 反归一化 [-1,1] -> [0,1]
        save_image(img, os.path.join(real_dir, f"{idx}.png"))
    # 使用pytorch-fid计算
    fid_value = calculate_fid_given_paths([real_dir, gen_dir], batch_size=10, device=DEVICE, dims=2048)
    print(f"FID score at epoch {epoch+1}: {fid_value:.2f}")


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, count = 0, 0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        t = torch.randint(0, T, (imgs.size(0),), device=DEVICE).long()  # t: [0, T-1]
        x_t, noise = add_noise(imgs, t)  # 一批图像，每个图像的t随机，noise也随机
        predicted_noise = model(x_t, t)
        loss = criterion(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)
    return total_loss / count
    

def main():
    # 数据处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 归一化到[0, 1]，转为tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]，符合正态分布
    ])
    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = UNet().to(DEVICE)
    # model.load_state_dict(torch.load(os.path.join(RES_DIR, 'model.pth')))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()  # 均方损失
    
    train_losses = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {train_loss:.4f}")
        # 每GEN_EPOCHS个epoch生成图像、保存模型
        if (epoch + 1) % GEN_EPOCHS == 0:
            samples = sample(model, n=16)
            samples = (samples + 1) / 2  # 还原为[0, 1]
            save_image(samples, os.path.join(RES_DIR, f"epoch_{epoch+1}.png"), nrow=4)
            torch.save(model.state_dict(), os.path.join(RES_DIR, 'model.pth'))
            calculate_fid(model, train_dataset, epoch)  # FID评估

    end_time = time.time()
    print_time(start_time, end_time)
    plot(train_losses)
     

if __name__ == '__main__':
    main()
