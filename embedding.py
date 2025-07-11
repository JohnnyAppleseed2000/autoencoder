from model import Autoencoder
import dataset
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# MNIST 데이터셋 준비
train_loader = dataset.train_loader
test_loader = dataset.test_loader

# 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth'))  # 학습된 모델 있으면 불러오기
model.eval()

# 임베딩과 라벨 저장용
embeddings = []
labels = []
with torch.no_grad():
    for image, label in test_loader:
      emb = model.get_embedding(image.to(device))
      embeddings.append(emb.cpu().numpy())
      labels.append(label.cpu().numpy())
embeddings = np.vstack(embeddings)
labels = np.hstack(labels)

# tsne 사용해 2D로 차원 축소
tsne = TSNE(n_components=2)
emb_2d = tsne.fit_transform(embeddings)

#시각화
plt.figure(figsize=(10,8))
scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1],c=labels, cmap='tab10', s=5)
plt.colorbar(scatter, ticks=range(10))
plt.title("MNIST Embedding Visualization by t-SNE")
plt.show()
