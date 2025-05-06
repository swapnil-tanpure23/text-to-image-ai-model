import torch
from torch.utils.data import DataLoader
from text_encoder import TextEncoder
from model.generator import ImageGenerator
from dataset import TextImageDataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = TextImageDataset("data/captions.csv", "data/images")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load models
text_encoder = TextEncoder().to(device)
generator = ImageGenerator().to(device)

optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    for images, captions in dataloader:
        images = images.to(device)
        text_embeddings = text_encoder(captions).to(device)
        generated_images = generator(text_embeddings)

        loss = F.mse_loss(generated_images, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Save the models after training
torch.save(text_encoder.state_dict(), "text_encoder.pth")
torch.save(generator.state_dict(), "generator.pth")

print("Models saved to 'text_encoder.pth' and 'generator.pth'")
