import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClinicDBDataset(Dataset):
    """Hugging Face snapshot loader with deterministic image-mask pairing."""

    def __init__(self, args, data_dir, split, transform=None):
        self.args = args
        self.split = split
        self.transform = transform
        self.split_dir = Path(data_dir) / split
        metadata_path = self.split_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise RuntimeError(f"Metadata does not exist: {metadata_path}")

        pairs = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                image_path = self.split_dir / record["file_name"]
                mask_path = self.split_dir / record["mask_file_name"]
                if not image_path.is_file() or not mask_path.is_file():
                    raise RuntimeError(f"Missing pair: {image_path}, {mask_path}")
                pairs.append((image_path, mask_path))

        pairs.sort(key=lambda pair: pair[0].name)
        if split == "train":
            rng = np.random.default_rng(args.split_seed)
            order = rng.permutation(len(pairs))
            pairs = [pairs[index] for index in order]

        self.pairs = pairs
        self.image_normalization = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        print(f"Loaded {split} split with {len(self.pairs)} paired images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        label = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        if self.transform:
            if self.split == "train":
                key = "train_weak" if index < self.args.labeled_num else "train_strong"
                data = self.transform[key](image=image, mask=label)
            else:
                data = self.transform(image=image, mask=label)
            image, label = data["image"], data["mask"]

        label = (label >= 0.5).astype(np.int64)
        image = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1)
        image = self.image_normalization(image)
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return {
            "image": image,
            "label": label,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
