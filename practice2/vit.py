import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# B：批量大小，C：类别数，H：图片高度，W：图片宽度
# P：patch大小，E：嵌入维度，N：patch个数
# h：注意力头数，hd：每个头的维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 100  # 150
DROP_RATE = 0.2
IMG_CHANNELS = 3  # 图像为3通道
IMG_SIZE = 32  # 图像：32*32*3
PATCH_SIZE = 4  # 切出的patch：P*P*3
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 # 一张图切出的patch的数量
# 重要调参
WEIGHT_DECAY = 1e-3
LR = 1e-3
EMBED_DIM = 192
NUM_HEADS = 3  # 注意力头数，这里要求E是h的倍数
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 每个头的维度
NUM_ENCODERS = 6  # encoder层数
MLP_RATIO = 4  # encoder-mlp中将E投影到x倍
# 采用相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RES_DIR = os.path.join(BASE_DIR, 'res')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # 用卷积层实现：将图像切成patch，每个patch映射为E维向量，相当于全连接
        self.proj = nn.Conv2d(IMG_CHANNELS, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.proj(x)  # [B, 3, 32, 32] -> [B, E, H/P, W/P]，相当于 H/P * W/P 个patch
        x = x.flatten(2).transpose(1, 2)  # 将H/P * W/P展开为一列，并交换1、2维度：[B, H/P * W/P, E]
        return x


class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # 1：利用广播机制，将位置编码添加到整个批量上，这里采用自学习矩阵
        self.position_embed = nn.Parameter(torch.randn(1, NUM_PATCHES+1, EMBED_DIM))

    def forward(self, x):
        return x + self.position_embed  # x是PatchEmbedding并加上cls之后的


class MyMultiheadAttention(nn.Module):  # 相较于笔记的公式，这里很多地方都简化了
    def __init__(self):
        super().__init__()
        self.scale = HEAD_DIM ** -0.5  # 缩放因子，避免点积过大
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3)  # 一次性得到q、k、v
        self.dropout = nn.Dropout(DROP_RATE)
        self.out = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, N, E = x.shape  # 批量大小、patch数量（含cls）、嵌入维度
        # 首先对x线性变换，再拆成q、k、v三部分，每部分都是把x映射到E维
        # 每部分有h个头，每个头hd维，且h*hd = E
        # 调整维度顺序：[3, B, h, N, hd]
        qkv = self.qkv(x).reshape(B, N, 3, NUM_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, h, N, hd]，这里hd就是原文的dk、dv
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_out = attn_probs @ v  # [B, h, N, hd]
        attn_out = attn_out.transpose(1, 2).reshape(B, N, E)  # 拼接各个头：-> [B, N, h, hd] -> [B, N, E]
        return self.out(attn_out)


class OfficialMultiheadAttention(nn.Module):  # 官方的MultiheadAttention
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=DROP_RATE,
            batch_first=True  # 保证输入输出形状为[B, N, E]，与自定义代码兼容
        )
        self.dropout = nn.Dropout(DROP_RATE)

    def forward(self, x):
        attn_out, _ = self.attn(query=x, key=x, value=x)  # 注意力输出, 注意力权重
        attn_out = self.dropout(attn_out)
        return attn_out


class TransformerEncoder(nn.Module):  # 与vit架构图中的Encoder一致
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn = MyMultiheadAttention()  # 自定义的
        # self.attn = OfficialMultiheadAttention()  # 官方的
        self.layer_norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, int(EMBED_DIM * MLP_RATIO)),
            nn.GELU(),  # 激活函数
            nn.Dropout(DROP_RATE),
            nn.Linear(int(EMBED_DIM * MLP_RATIO), EMBED_DIM),
            nn.Dropout(DROP_RATE),
        )

    def forward(self, x):
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.position_embed = PositionEmbedding()
        self.cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM))  # 初始化一个CLS向量，认为它已经patch_embed了
        self.transformer_encoder = nn.Sequential(  # 堆多层encoder
            *[ TransformerEncoder()
            for _ in range(NUM_ENCODERS) ]
        )
        self.layer_norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)
        # self.apply(self._init_weights)  # 调用初始化【效果不好】

    def forward(self, x):
        B = x.size(0)  # 批量大小
        x = self.patch_embed(x)  # [B, 3, 32, 32] -> [B, N, E]
        cls = self.cls.expand(B, -1, -1)  # 把CLS向量扩展为批量个
        x = torch.cat((cls, x), dim=1)  # cls与x拼接：[B, N+1, E]
        x = self.position_embed(x)
        x = self.transformer_encoder(x)
        cls_out = self.layer_norm(x[:, 0])  # 取出CLS：[B, E]，然后将它归一化
        return self.classifier(cls_out)  # 用CLS分类：[B, C]
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 线性层使用Xavier初始化
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # 卷积层使用Kaiming初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            # 层归一化初始化
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def plot(train_losses, test_losses, train_accs, test_accs):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validate Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(RES_DIR, 'vit_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Validate Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(RES_DIR, 'vit_accuracy.png'))
    plt.close()


def print_time(begin, end, best_epoch, best_test_acc):
    # 总时间
    total_time = end - begin
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"Training time: {minutes}m {seconds}s")
    # 最佳acc的时间
    best_time = total_time * best_epoch / EPOCHS
    minutes = int(best_time // 60)
    seconds = int(best_time % 60)
    print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    print(f"Time at best accuracy: {minutes}m {seconds}s")



def main():
    # 检查用什么卡
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. Using CPU.")

    # 数据增强（可以很好限制过拟合，但也会使得收敛变慢）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),     # 水平翻转
        transforms.RandomRotation(15),         # 随机旋转±15°
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # cifar10的均值和标准差
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4940, 0.4850, 0.4504], std=[0.2467, 0.2429, 0.2616])
    ])

    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = ViT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)  # 学习率调度
    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_test_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()  # 更新学习率
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if test_acc > best_test_acc:  # 跟踪最佳测试准确率及其周期
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(RES_DIR, 'vit.pth'))
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{EPOCHS}: Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f} | LR {current_lr:.6f}")

    end_time = time.time()
    print_time(start_time, end_time, best_epoch, best_test_acc)
    plot(train_losses, test_losses, train_accs, test_accs)
   

if __name__ == '__main__':
    main()
