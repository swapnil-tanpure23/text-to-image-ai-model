import torch.nn as nn

class ImageGenerator(nn.Module):
    def __init__(self, text_embedding_dim=512):
        super(ImageGenerator, self).__init__()
        self.fc = nn.Linear(text_embedding_dim, 256 * 8 * 8)

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, text_embedding):
        x = self.fc(text_embedding).view(-1, 256, 8, 8)
        return self.generator(x)
