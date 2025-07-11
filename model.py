import torch.nn as nn


# Autoencoder 모델 구현
class Autoencoder(nn.Module):
    def __init__(self, enc_size=256):
        super(Autoencoder, self).__init__()
        self.enc_size = enc_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (24, 14, 14)

            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (48, 7, 7)

            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (96, 3, 3)

            nn.Flatten(),
            nn.Linear(96 * 3 * 3, self.enc_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.enc_size, 96 * 3 * 3),
            nn.Unflatten(1, (96, 3, 3)),
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),  # (48, 6, 6)
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),  # (24, 12, 12)
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=3, stride=2, padding=1, output_padding=1),  # (12, 24, 24)
            nn.ReLU(),
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),  # (12, 28, 28)
            nn.Conv2d(12, 1, kernel_size=3, padding=1),  # (1, 28, 28)
            nn.Sigmoid()
        )

    def get_embedding(self, x):
        encoded = self.encoder(x)
        return encoded

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Early Stopping 기능
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.EarlyStop = False

    def __call__(self, running_loss):
        if self.best_loss is None:
            self.best_loss = running_loss
        elif self.best_loss >= running_loss:
            self.best_loss = running_loss
        elif self.best_loss < running_loss:
            self.best_loss = self.best_loss
            self.counter += 1
            if self.counter == self.patience:
                self.EarlyStop = True
