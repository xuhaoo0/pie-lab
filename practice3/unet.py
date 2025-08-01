import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 仿照 UNet 的实现，添加时间嵌入和残差连接

def sinusoidal_embedding(timesteps, dim):
    """正弦余弦时间嵌入，转为dim维向量"""
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        time_emb = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.residual(x)


class DownBlock(nn.Module):
    """下采样块用于缩小特征图尺寸、增加通道数"""
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.block1 = ResidualBlock(in_c, out_c, time_emb_dim)
        self.block2 = ResidualBlock(out_c, out_c, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        x_down = self.pool(x)
        return x_down, x


class UpBlock(nn.Module):
    """上采样块用于增加特征图尺寸、减少通道数"""
    def __init__(self, in_c, skip_c, out_c, time_emb_dim):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.block1 = ResidualBlock(out_c + skip_c, out_c, time_emb_dim)
        self.block2 = ResidualBlock(out_c, out_c, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x



class UNet(nn.Module):
    """输入图像、编码器下采样、中间层处理、解码器上采样、输出噪声预测"""
    def __init__(self, img_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.time_emb_dim = time_emb_dim
        self.input_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.middle1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.middle2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.up1 = UpBlock(in_c=base_channels * 4, skip_c=base_channels * 4, out_c=base_channels * 2, time_emb_dim=time_emb_dim)
        self.up2 = UpBlock(in_c=base_channels * 2, skip_c=base_channels * 2, out_c=base_channels, time_emb_dim=time_emb_dim)
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        x = self.input_conv(x)
        x1, skip1 = self.down1(x, t_emb)
        x2, skip2 = self.down2(x1, t_emb)
        x_mid = self.middle1(x2, t_emb)
        x_mid = self.middle2(x_mid, t_emb)
        x = self.up1(x_mid, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        return self.output_conv(x)

