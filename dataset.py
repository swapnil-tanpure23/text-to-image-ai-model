import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.image_dir}/{row['filename']}").convert("RGB")
        return self.transform(image), row['text']
