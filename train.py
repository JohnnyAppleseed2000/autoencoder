from model import Autoencoder
from model import EarlyStopping
import dataset

# 모델 초기화
model = Autoencoder().to(device)
train_loader = dataset.train_loader
test_loader = dataset.test_loader

# 3. 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 학습 루프
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # 입력과 출력 비교

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    EarlyStopping(running_loss)
    if EarlyStopping.EarlyStop == True:
        break
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}")

print("훈련 완료!")