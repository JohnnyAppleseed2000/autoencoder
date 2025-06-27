class Autoencoder(nn.module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        enc_size = 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=28,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Relu(),
            nn.MaxPool2D(2,2),
            # 출력크기 (28,14,14)

            nn.Conv2d(
                in_channels=1,
                out_channels=56,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Relu(),
            nn.MaxPool2D(2, 2)
            # 출력크기 (56,7,7)

        nn.Flatten(),
        nn.Linear(56 * 7 * 7, enc_size)
        )

        self.decoder = nn.Sequential

