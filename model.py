import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.enc_size = 512
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
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
