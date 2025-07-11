import torch
import torchvision
import torchvision.transforms as transforms


##이미지 텐서로 변환
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root='./data', train='True', transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
