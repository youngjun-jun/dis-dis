import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, latent_units=20, d=64, image_size=64, num_channels=3, context_dim=32):
        super().__init__()
        self.image_encoder = ImageEncoder(latent_units=latent_units, d=d, image_size=image_size, num_channels=num_channels)
        self.mlps = nn.ModuleList([MLPModule(input_dim=1, output_dim=context_dim) for _ in range(latent_units)])

    def forward(self, x):
        features = self.image_encoder(x)
        tokens = [mlp(features[:, i:i+1]) for i, mlp in enumerate(self.mlps)]
        return torch.stack(tokens, dim=1)

class ImageEncoder(nn.Module):
    def __init__(self, latent_units, d=64, image_size=64, num_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(d * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d * 4, d * 4, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(d * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d * 4, d * 4, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(d * 4),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(d * 4 * (image_size // 16) * (image_size // 16), 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_units)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MLPModule(nn.Module):
    def __init__(self, input_dim=1, output_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
