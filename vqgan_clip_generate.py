import torch
from taming.models.vqgan import VQModel
import clip
from transformers import CLIPTextModel, CLIPTokenizer
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms

# Load VQGAN model configuration and checkpoint
with open('vqgan_config.yaml', 'r') as f:
    vqgan_config = yaml.safe_load(f)

# Initialize VQGAN model with correct parameters
vqgan_params = vqgan_config['model']['params']
vqgan_model = VQModel(**vqgan_params)  # Ensure the parameters match the expected ones for VQModel
vqgan_model.load_state_dict(torch.load('vqgan_checkpoint.ckpt'))
vqgan_model = vqgan_model.to('cuda')

# Load CLIP model
clip_model, clip_preprocess = clip.load('ViT-B/32', device='cuda')

# Tokenizer and Text Encoder from CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')

def generate_image_from_text(prompt):
    # Encode the text prompt using CLIP
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    text_embeddings = text_encoder(**inputs).last_hidden_state.mean(dim=1).to('cuda')

    # Generate noise image
    noise = torch.randn(1, 3, 256, 256).cuda()

    # Define the optimizer for the image generation process
    optimizer = torch.optim.Adam([noise.requires_grad_()], lr=0.05)

    # Preprocess function for converting the generated image for CLIP
    preprocess = clip_preprocess

    for i in range(100):  # 100 iterations to optimize the generated image
        optimizer.zero_grad()

        # Forward pass through VQGAN (decoder part)
        generated_image = vqgan_model.decode(noise)

        # Preprocess the generated image for CLIP
        generated_image_pil = Image.fromarray(generated_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8))
        image_input = preprocess(generated_image_pil).unsqueeze(0).to('cuda')

        # Compute the similarity between the image and the text using CLIP
        image_features = clip_model.encode_image(image_input)
        similarity = torch.cosine_similarity(image_features, text_embeddings)

        # Backpropagate the loss to optimize the image
        loss = -similarity.mean()
        loss.backward()
        optimizer.step()

        # Every 10 steps, print the progress
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}: Loss = {loss.item():.4f}")

    # Final image generation
    generated_image_pil = Image.fromarray(generated_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8))
    generated_image_pil.save("generated_image_vqgan_clip.png")

# Example text prompt
if __name__ == "__main__":
    prompt = "A futuristic city skyline with neon lights and flying cars"
    generate_image_from_text(prompt)
