import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import random
import numpy as np

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()  # 默认使用42作为种子

device = "cuda" if torch.cuda.is_available() else "cpu"

# 由于我这里不知道为什么下不下来，所以我就手动下载了Cora数据集
dataset = Planetoid(root='Data', name='Cora')
data    = dataset[0].to(device)
print(f'[Data] {data.num_nodes} nodes, {data.num_edges} edges, '
      f'{data.num_features} feats, {dataset.num_classes} classes.')

# GCN网络结构
class GCN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def embed(self, x, edge_index):
        return F.relu(self.conv1(x, edge_index))
    
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[mask].argmax(1)
    acc = (pred == data.y[mask]).float().mean().item()
    return acc, pred.cpu()  

# 可视化
@torch.no_grad()
def plot():
    model.eval()
    emb = model.embed(data.x, data.edge_index).cpu()
    tsne = TSNE(n_components=2)
    emb_2d = tsne.fit_transform(emb)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1],
                    hue=data.y.cpu(), palette='tab10', s=15)
    plt.title('GCN embeddings (t-SNE)')
    plt.legend(title='Class')
    plt.savefig('gcn_tsne.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    set_seed()  # 在主程序开始前再次确保种子设置
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(200):
        loss = train()
        train_acc, _ = test(data.train_mask)
        val_acc, _ = test(data.val_mask)

        train_losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'Graph Convolutional Network\GCN-Cora\GCN_cora.pt')
            best_epoch = epoch
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}  Loss {loss:.4f}  Train {train_acc:.4f}  Val {val_acc:.4f}')

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()

    model.load_state_dict(torch.load('Graph Convolutional Network\GCN-Cora\GCN_cora.pt'))
    test_acc, test_pred = test(data.test_mask)
    print(f'\n[Test] Accuracy = {test_acc:.4f}')
    class_names = [f'Class_{i}' for i in range(dataset.num_classes)]    # 生成类别名称
    print(classification_report(data.y[data.test_mask].cpu(),test_pred,target_names=class_names,digits=4))

    plot()