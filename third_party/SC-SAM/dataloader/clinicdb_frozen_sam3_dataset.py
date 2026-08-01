import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClinicDBFrozenSAM3Dataset(Dataset):
    """ClinicDB with fixed native-top SAM3 masks for unlabeled training data."""

    def __init__(self, args, data_dir, split, transform=None, candidate_npz=None, pseudo_dir=None):
        self.args = args
        self.split = split
        self.transform = transform
        self.split_dir = Path(data_dir) / split
        metadata_path = self.split_dir / "metadata.jsonl"
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
            order = np.random.default_rng(args.split_seed).permutation(len(pairs))
            pairs = [pairs[index] for index in order]
        self.pairs = pairs

        self.pseudo_masks = {}
        self.pseudo_audit = {}
        self.pseudo_accepted = {}
        if split == "train":
            if pseudo_dir:
                pseudo_dir = Path(pseudo_dir)
                self.pseudo_masks = {path.name: path for path in pseudo_dir.glob("*.png")}
            elif candidate_npz:
                archive = np.load(candidate_npz, allow_pickle=True)
                image_ids = archive["image_id"]
                names = archive["image_names"].astype(str)
                for image_id, name in enumerate(names):
                    group = np.flatnonzero(image_ids == image_id)
                    selected = group[np.argmax(archive["sam_score"][group])]
                    self.pseudo_masks[name] = archive["mask_lowres"][selected].astype(np.uint8)
                    self.pseudo_audit[name] = {
                        "sam_score": float(archive["sam_score"][selected]),
                        "dice": float(archive["dice"][selected]),
                    }
            else:
                raise ValueError("candidate_npz or pseudo_dir is required for training")
            required_pairs = pairs[self.args.labeled_num :]
            missing = [path.name for path, _ in required_pairs if path.name not in self.pseudo_masks]
            if missing:
                raise RuntimeError(f"SAM3 archive is missing {len(missing)} images: {missing[:5]}")
            if args.pseudo_quality_csv:
                with Path(args.pseudo_quality_csv).open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        self.pseudo_accepted[row["image"]] = bool(int(row["accepted"]))
                missing_quality = [
                    path.name for path, _ in required_pairs
                    if path.name not in self.pseudo_accepted
                ]
                if missing_quality:
                    raise RuntimeError(
                        f"Pseudo-quality CSV is missing {len(missing_quality)} images: "
                        f"{missing_quality[:5]}"
                    )

        self.accepted_unlabeled_indices = []
        if split == "train":
            for index in range(self.args.labeled_num, len(self.pairs)):
                name = self.pairs[index][0].name
                if self.pseudo_accepted.get(name, True):
                    self.accepted_unlabeled_indices.append(index)

        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        print(f"Loaded {split} split with {len(self.pairs)} paired images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        is_labeled = self.split != "train" or index < self.args.labeled_num

        if is_labeled:
            target = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        else:
            pseudo = self.pseudo_masks[image_path.name]
            if isinstance(pseudo, Path):
                target = np.asarray(Image.open(pseudo).convert("L"), dtype=np.float32) / 255.0
            else:
                target = cv2.resize(
                    pseudo, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                ).astype(np.float32)

        if self.transform:
            if self.split == "train":
                key = "train_weak" if is_labeled else "train_strong"
                data = self.transform[key](image=image, mask=target)
            else:
                data = self.transform(image=image, mask=target)
            image, target = data["image"], data["mask"]

        image = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1)
        target = torch.from_numpy(np.ascontiguousarray(target >= 0.5)).long()
        return {
            "image": self.normalize(image),
            "target": target,
            "is_labeled": torch.tensor(is_labeled, dtype=torch.bool),
            "image_name": image_path.name,
        }
