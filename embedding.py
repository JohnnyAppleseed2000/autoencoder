import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Autoencoder 정의 (생략 가능, 이미 있으니 그냥 재사용)
class Autoencoder(nn.Module):
    def __init__(self, enc_size=256):
        super(Autoencoder, self).__init__()
        self.enc_size = enc_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 96, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, self.enc_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.enc_size, 96 * 3 * 3),
            nn.Unflatten(1, (96, 3, 3)),
            nn.ConvTranspose2d(96, 48, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),
            nn.Conv2d(12, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # 임베딩 추출 함수 추가
    def get_embedding(self, x):
        return self.encoder(x)

# MNIST 데이터셋 준비
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# 모델 불러오기 혹은 새로 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder(enc_size=256).to(device)
# model.load_state_dict(torch.load('autoencoder.pth'))  # 학습된 모델 있으면 불러오기

model.eval()

# 임베딩과 라벨 저장용
embeddings = []
labels = []

with torch.no_grad():
    for images, lbls in testloader:
        images = images.to(device)
        emb = model.get_embedding(images)  # (batch_size, enc_size)
        embeddings.append(emb.cpu().numpy())
        labels.append(lbls.numpy())

embeddings = np.vstack(embeddings)  # (num_samples, enc_size)
labels = np.hstack(labels)

# t-SNE로 2D 차원 축소
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# 시각화
plt.figure(figsize=(10,8))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=5)
plt.colorbar(scatter, ticks=range(10))
plt.title("MNIST Autoencoder Embedding Visualization (t-SNE)")
plt.show()
