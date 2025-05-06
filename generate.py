from vqgan_clip_generate import generate_image_from_text

def generate_image_from_prompt(prompt):
    """Generates an image from a text prompt using VQGAN+CLIP."""
    print(f"Generating image for prompt: {prompt}")
    generate_image_from_text(prompt)

if __name__ == "__main__":
    # Example text prompt
    prompt = "A futuristic city skyline with neon lights and flying cars"
    generate_image_from_prompt(prompt)
