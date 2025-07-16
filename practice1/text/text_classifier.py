import os, re, random, tarfile, urllib.request, torch, time
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 超参数
MODEL_NAME = 'bilstm'  # 选择模型，见get_model
NUM_LAYERS = 1     # LSTM层数
MAX_LEN = 300
BATCH_SIZE = 64
EMBED_DIM = 200  # 词嵌入维度
HIDDEN_SIZE = 256  # LSTM隐藏层大小
NUM_CLASSES = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT = 0.5  
MAX_VOCAB_SIZE = 20000  # 只留下最常用的词，其他一律当作unk
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的所在目录
DATA_DIR = os.path.join(BASE_DIR, 'data/aclImdb')  # 数据存放的位置


# 工具
def get_model(vocab_size):
    """根据模型名称获取对应的模型类"""
    if MODEL_NAME == 'bilstm':
        return BiLSTMClassifier(vocab_size)
    elif MODEL_NAME == 'lstm':
        return LSTMClassifier(vocab_size)
    elif MODEL_NAME == 'res_lstm':
        return ResLSTMClassifier(vocab_size)
    elif MODEL_NAME == 'my_lstm':
        return MyLSTMClassifier(vocab_size)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")
    

def download_and_extract():
    """下载并解压IMDB数据集"""
    if not os.path.exists(DATA_DIR):
        os.makedirs('data', exist_ok=True)
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(URL, 'data/aclImdb_v1.tar.gz')
        print("Extracting...")
        with tarfile.open('data/aclImdb_v1.tar.gz', 'r:gz') as tar:
            tar.extractall(path='data')
        print("Done.")


def tokenizer(text):
    """将文本小写，并用正则表达式分词"""
    return re.findall(r'\b\w+\b', text.lower())


def build_vocab(tokenized_texts, max_size=MAX_VOCAB_SIZE, min_freq=2):
    """构建词汇表，保留最频繁的max_size个词"""
    counter = Counter()
    for text in tokenized_texts:
        counter.update(text)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    """将text编码为长度为max_len的ID序列"""
    tokens = tokenizer(text)
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len]]
    ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids[:max_len]


def read_data(split_dir):
    """读取数据集，返回标签、文本对"""
    data = []
    for label in ['pos', 'neg']:
        folder = os.path.join(split_dir, label)
        for fname in os.listdir(folder):
            if fname.endswith('.txt'):
                with open(os.path.join(folder, fname), encoding='utf-8') as f:
                    text = f.read()
                    data.append((label, text))
    random.shuffle(data)
    return data


def plot(train_losses, val_losses, train_accs, val_accs):
    res_dir = os.path.join(BASE_DIR, 'res')
    os.makedirs(res_dir, exist_ok=True)
    # 绘制 loss 曲线
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(f"{res_dir}/{MODEL_NAME}{NUM_LAYERS}_loss.png")
    plt.close()
    # 绘制 accuracy 曲线
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(f"{res_dir}/{MODEL_NAME}{NUM_LAYERS}_accuracy.png")
    plt.close()




# 训练与验证
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


# 数据集、模型类
class IMDBDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = [(encode(text, vocab), 1 if label == 'pos' else 0) for label, text in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)  # 嵌入矩阵
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, batch_first=True,
                            bidirectional=True, dropout=DROPOUT)  # num_layers层双向LSTM，每个词从embed_dim到hidden_size*2
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = torch.mean(out, dim=1)  # 把一个句子中的所有词取平均，得到一个综合性的hidden_size*2
        out = self.dropout(out)
        return self.fc(out)
    

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)  # 嵌入矩阵
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, batch_first=True,
                            bidirectional=False, dropout=DROPOUT)  
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE , NUM_CLASSES)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = torch.mean(out, dim=1)  # 把一个句子中的所有词取平均，得到一个综合性的hidden_size
        out = self.dropout(out)
        return self.fc(out)


class ResLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Identity()  # 如果维度不匹配则通过线性层变换
        if input_size != hidden_size:
            self.res_proj = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        residual = self.res_proj(x)
        out, _ = self.lstm(x)
        out = self.dropout(out + residual)
        return out


class ResLSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.res_lstm_layers = nn.ModuleList()
        input_size = EMBED_DIM
        for _ in range(NUM_LAYERS):  # 每层都是一个残差LSTM
            self.res_lstm_layers.append(ResLSTMBlock(input_size, HIDDEN_SIZE, DROPOUT))
            input_size = HIDDEN_SIZE  # 只有第一层的输入是embed_dim
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        emb = self.embedding(x)  
        out = emb
        for layer in self.res_lstm_layers:
            out = layer(out)  
        out = torch.mean(out, dim=1)  
        return self.fc(out)


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # LSTM四个门的线性变换
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden  # 上一时刻隐藏状态和细胞状态

        gates = self.x2h(x) + self.h2h(h_prev)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, 1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c_current = f_gate * c_prev + i_gate * g_gate
        h_current = o_gate * torch.tanh(c_current)

        return h_current, c_current


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # 多层LSTM
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(MyLSTMCell(layer_input_size, hidden_size))

    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_len, input_size]
        hidden: (h_0, c_0), each is [num_layers, batch_size, hidden_size]
        """
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            (h_0, c_0) = hidden
            h_t = [h_0[i] for i in range(self.num_layers)]
            c_t = [c_0[i] for i in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_new, c_new = self.cells[layer](input_t, (h_t[layer], c_t[layer]))
                h_t[layer], c_t[layer] = h_new, c_new
                input_t = self.dropout(h_new) if layer != self.num_layers - 1 else h_new
            outputs.append(input_t.unsqueeze(1))

        out = torch.cat(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
        h_n = torch.stack(h_t)  # [num_layers, batch_size, hidden_size]
        c_n = torch.stack(c_t)

        return out, (h_n, c_n)


class MyLSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = MyLSTM(EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, dropout=DROPOUT)  # 使用自定义的MyLSTM
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        emb = self.embedding(x)  
        out, _ = self.lstm(emb)  
        out = torch.mean(out, dim=1)  # 取时间维度平均
        out = self.dropout(out)
        return self.fc(out)


def main():
    # 检查用什么卡
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. Using CPU.")

    download_and_extract()  # 会检查数据是否下载

    train_raw = read_data(os.path.join(DATA_DIR, 'train'))
    test_raw = read_data(os.path.join(DATA_DIR, 'test'))
    tokenized_texts = [tokenizer(text) for _, text in train_raw]
    vocab = build_vocab(tokenized_texts)

    train_dataset = IMDBDataset(train_raw, vocab)
    test_dataset = IMDBDataset(test_raw, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    vacab_size = len(vocab)
    model = get_model(vacab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    plot(train_losses, val_losses, train_accs, val_accs)


if __name__ == '__main__':
    main()  # 需要先修改MODEL_NAME、NUM_LAYERS

