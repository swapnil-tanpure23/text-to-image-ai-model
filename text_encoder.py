import torch
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder(torch.nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.text_encoder(**tokens)
        return outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_dim)
