import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit import ViT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RES_DIR = os.path.join(BASE_DIR, 'res')
MODEL_PATH = os.path.join(RES_DIR, 'vit.pth')  # 模型参数路径


def load_data():
    """加载测试数据集"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=2
    )
    return test_loader, test_set.classes  # 返回测试数据加载器和类别名称


def load_model():
    """加载模型并加载预训练参数"""
    model = ViT().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # 切换到评估模式
    print(f"模型加载成功：{MODEL_PATH}")
    return model


def test_model(model, test_loader, classes):
    """测试模型并输出准确率"""
    total = 0
    correct = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    with torch.no_grad():  # 关闭梯度计算，加速推理
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            
            # 总准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 输出总准确率
    print(f"\n总测试准确率：{100 * correct / total:.2f}% ({correct}/{total})")
    
    # 输出每个类别的准确率
    print("\n各类别准确率：")
    for i in range(len(classes)):
        print(f"{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})")


if __name__ == '__main__':
    test_loader, classes = load_data()
    model = load_model()
    test_model(model, test_loader, classes)