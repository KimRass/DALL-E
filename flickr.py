# Sources:
    # https://www.kaggle.com/datasets/adityajn105/flickr8k
    # https://www.kaggle.com/datasets/adityajn105/flickr30k
# References:
    # https://github.com/KimRass/CLIP/blob/main/flickr.py

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path
import os
from PIL import Image
import re
from collections import defaultdict
import random

from bpe import tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def encode(text, tokenizer, max_len):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    return encoding["input_ids"]


# def pad(token_ids, max_len, pad_id):
#     token_ids = token_ids[: max_len]
#     token_ids += [pad_id] * (max_len - len(token_ids))
#     return token_ids


# def get_attention_mask(token_ids, pad_id):
#     return (token_ids != pad_id).long()


class FlickrDS(Dataset):
    def __init__(self, data_dir, tokenizer, max_len, img_size):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.images_dir = self.data_dir/"Images"
        self.img_paths = sorted(list(map(str, self.images_dir.glob("**/*.jpg"))))

        self.transformer = T.Compose([
            T.Resize(size=img_size),
            T.CenterCrop(size=img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.captions = defaultdict(list)
        with open(self.data_dir/"captions.txt", mode="r") as f:
            for line in f:
                line = line.strip()
                if ".jpg" in line:
                    split_idx = re.search(pattern=r"(.jpg)", string=line).span()[1]
                    img_path = str(self.images_dir/line[: split_idx])
                    text = line[split_idx + 1:].replace(" .", ".")
                    if img_path in self.img_paths:
                        self.captions[img_path].append(text)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = self.transformer(image)

        texts = self.captions[img_path]
        text = random.choice(texts)
        token_ids = encode(text, tokenizer=self.tokenizer, max_len=self.max_len)
        return image, token_ids


if __name__ == "__main__":
    # MAX_LEN = 256
    MAX_LEN = 64
    IMG_SIZE = 256
    IMG_TOKEN_SIZE = 32
    ds = FlickrDS(
        data_dir="/Users/jongbeomkim/Documents/datasets/flickr30k",
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        img_size=IMG_SIZE,        
    )
    ds[0]
