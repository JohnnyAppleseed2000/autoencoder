from model import Autoencoder
import dataset
import matplotlib.pyplot as plt
import torch

# device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 모델이 위치한 device

# 모델 불러오기
model = Autoencoder.to(device)

# 모델을 evaluation 모드로 전환
model.eval()


# 테스트용 데이터 준비
test_loader = dataset.test_loader
with torch.no_grad():
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)

    # 모델에 입력하여 출력 생성
    outputs = model.forward(images)

# CPU로 이동 후 numpy 변환
images = images.cpu().numpy()
outputs = outputs.cpu().numpy()

# 시각화: 원본과 비교
n = 6
plt.figure(figsize=(12, 4))
for i in range(n):
    # 원본 이미지
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 복원 이미지
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(outputs[i][0], cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()