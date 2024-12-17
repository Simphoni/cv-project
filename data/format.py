# format.py converts the ROCO dataset into huggingface datasets format (image folder with a metadata.jsonl file)

import os
import json
from pathlib import Path
from PIL import Image
from io import BytesIO

import datasets

data_path = Path("/home/ubuntu/lora/data")
train_path = data_path / "train"

def main():
    metadata = train_path / "captions.txt"
    image_dir = train_path
    with open(metadata, "r") as f:
        entries = f.readlines()
    
    valid_data = []
    for entry in entries:
        tmp = list(entry.split('\t'))
        tmp = [item.strip() for item in tmp]
        img = tmp[0]
        caption = tmp[1]
        if os.path.exists(image_dir / f"{img}.jpg"):
            fail = False
            try:
                image = Image.open(image_dir / f"{img}.jpg")
            except:
                fail = True
            if fail:
                print(f"image {img} failed, current size={len(valid_data)}")
                continue
            valid_data.append({"file_name": f"{img}.jpg", "text": caption})
    with open(train_path / "metadata.jsonl", "w") as jsonl_file:
        for entry in valid_data:
            jsonl_file.write(json.dumps(entry) + "\n")
    

from torchvision import transforms

def test():
    dataset = datasets.load_dataset("imagefolder", data_dir=str(data_path), split="train")
    height = []
    width = []
    fail = 0
    for i in range(len(dataset)):
        try:
            image = dataset[i]['image']
        except:
            fail += 1
            continue
        height.append(image.height)
        width.append(image.width)
    

# main()
test()

