"""Paired restoration datasets and batch sampling for dynamic resolutions."""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import ToTensor

from utils.image_utils import data_augmentation


def parse_train_sizes(specification, patch_size=16):
    """Parse ``256x256,320x480`` into a list of (height, width)."""
    sizes = []
    for item in specification.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            height = width = int(item)
        else:
            height, width = (int(value) for value in item.split("x", 1))
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid training size: {item}")
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"Training size {height}x{width} must be divisible by "
                f"model patch size {patch_size}"
            )
        sizes.append((height, width))
    if not sizes:
        raise ValueError("--train_sizes must contain at least one size")
    return sizes


class MultiScaleBatchSampler(Sampler):
    """Emit one randomly selected spatial size for every complete batch."""

    def __init__(
        self,
        dataset_length,
        batch_size,
        sizes,
        shuffle=True,
        drop_last=True,
        seed=0,
    ):
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.sizes = tuple(sizes)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(self.dataset_length))
        if self.shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            height, width = rng.choice(self.sizes)
            yield [
                (index, height, width)
                for index in batch
            ]

    def __len__(self):
        if self.drop_last:
            return self.dataset_length // self.batch_size
        return (
            self.dataset_length + self.batch_size - 1
        ) // self.batch_size


class DynamicPairDataset(Dataset):
    def __init__(self, split, task_id, repeat=1):
        self.split = split
        self.task_id = task_id
        self.repeat = repeat
        self.to_tensor = ToTensor()

    @staticmethod
    def _read(path):
        return np.array(Image.open(path).convert("RGB"), copy=True)

    @staticmethod
    def _align_pair(degraded, clean):
        height = min(degraded.shape[0], clean.shape[0])
        width = min(degraded.shape[1], clean.shape[1])
        return degraded[:height, :width], clean[:height, :width]

    @staticmethod
    def _pad(image, target_h, target_w):
        height, width = image.shape[:2]
        pad_h = max(target_h - height, 0)
        pad_w = max(target_w - width, 0)
        if pad_h == 0 and pad_w == 0:
            return image
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        mode = (
            "reflect"
            if height > max(top, bottom) and width > max(left, right)
            else "edge"
        )
        return np.pad(
            image,
            ((top, bottom), (left, right), (0, 0)),
            mode=mode,
        )

    def _training_crop(self, degraded, clean, target_h, target_w):
        degraded = self._pad(degraded, target_h, target_w)
        clean = self._pad(clean, target_h, target_w)
        height, width = degraded.shape[:2]
        top = random.randint(0, height - target_h)
        left = random.randint(0, width - target_w)
        slices = np.s_[
            top : top + target_h,
            left : left + target_w,
        ]
        modes = range(8) if target_h == target_w else (0, 1, 4, 5)
        mode = random.choice(tuple(modes))
        degraded = data_augmentation(degraded[slices], mode).copy()
        clean = data_augmentation(clean[slices], mode).copy()
        return degraded, clean

    @staticmethod
    def _decode_index(index):
        if isinstance(index, tuple):
            if len(index) != 3:
                raise ValueError(f"Unexpected dynamic index: {index}")
            return index
        return index, None, None

    def _get_paths(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.sample_count * self.repeat

    def __getitem__(self, request):
        index, target_h, target_w = self._decode_index(request)
        degraded_path, clean_path = self._get_paths(
            index % self.sample_count
        )
        degraded, clean = self._align_pair(
            self._read(degraded_path),
            self._read(clean_path),
        )
        if self.split == "train":
            if target_h is None or target_w is None:
                raise RuntimeError(
                    "Training datasets must be used with "
                    "MultiScaleBatchSampler"
                )
            degraded, clean = self._training_crop(
                degraded,
                clean,
                target_h,
                target_w,
            )
        return (
            [degraded_path, self.task_id],
            self.to_tensor(np.ascontiguousarray(degraded)),
            self.to_tensor(np.ascontiguousarray(clean)),
        )


class DynamicSOTSDataset(DynamicPairDataset):
    def __init__(self, args):
        super().__init__("test", task_id=0)
        root = getattr(args, "sots_root", None) or os.path.join(
            args.data_file_dir,
            "dehazing/SOTS/outdoor",
        )
        hazy_dir = os.path.join(root, "hazy")
        clean_dir = os.path.join(root, "gt")
        self.pairs = []
        for hazy_path in sorted(glob.glob(os.path.join(hazy_dir, "*.jpg"))):
            scene = os.path.basename(hazy_path).split("_")[0]
            clean_path = os.path.join(clean_dir, f"{scene}.png")
            if os.path.isfile(clean_path):
                self.pairs.append((hazy_path, clean_path))
        if not self.pairs:
            raise FileNotFoundError(f"No valid SOTS pairs under {root}")
        self.sample_count = len(self.pairs)
        print(
            f"[DynamicSOTS] pairs={self.sample_count}, "
            "evaluation=native-resolution"
        )

    def _get_paths(self, index):
        return self.pairs[index]


class DynamicRESIDEDataset(DynamicPairDataset):
    def __init__(self, args):
        super().__init__(
            "train",
            task_id=0,
            repeat=getattr(args, "reside_repeat", 1),
        )
        root = getattr(args, "reside_root", None) or os.path.join(
            args.data_file_dir,
            "dehazing/RESIDE",
        )
        hazy_by_scene = {}
        for part in ("part1", "part2", "part3", "part4"):
            pattern = os.path.join(root, part, "*.jpg")
            for hazy_path in sorted(glob.glob(pattern)):
                scene = os.path.basename(hazy_path).split("_")[0]
                hazy_by_scene.setdefault(scene, []).append(hazy_path)
        self.samples = []
        for scene, hazy_paths in sorted(hazy_by_scene.items()):
            clean_path = os.path.join(root, "clear", f"{scene}.jpg")
            if os.path.isfile(clean_path):
                self.samples.append((hazy_paths, clean_path))
        if not self.samples:
            raise FileNotFoundError(f"No valid RESIDE pairs under {root}")
        self.sample_count = len(self.samples)
        print(
            f"[DynamicRESIDE] scenes={self.sample_count}, "
            f"repeat={self.repeat}"
        )

    def _get_paths(self, index):
        hazy_paths, clean_path = self.samples[index]
        return random.choice(hazy_paths), clean_path


class DynamicRainDataset(DynamicPairDataset):
    def __init__(self, args, split):
        if split == "train":
            root = getattr(args, "rain_train_root", None) or os.path.join(
                args.data_file_dir,
                "deraining/RainTrainL",
            )
            repeat = getattr(args, "rain_repeat", 10)
        elif split in ("val", "test"):
            root = getattr(args, "rain_test_root", None) or os.path.join(
                args.data_file_dir,
                "deraining/Rain100L",
            )
            repeat = 1
        else:
            raise ValueError(f"Unknown split: {split}")
        super().__init__(split, task_id=1, repeat=repeat)
        rainy_dir = os.path.join(root, "rainy")
        clean_dir = os.path.join(root, "gt")
        self.pairs = []
        for rainy_path in sorted(glob.glob(os.path.join(rainy_dir, "*.png"))):
            name = os.path.basename(rainy_path).replace(
                "rain-",
                "norain-",
                1,
            )
            clean_path = os.path.join(clean_dir, name)
            if os.path.isfile(clean_path):
                self.pairs.append((rainy_path, clean_path))
        if not self.pairs:
            raise FileNotFoundError(f"No valid rain pairs under {root}")
        self.sample_count = len(self.pairs)
        resolution = (
            "multi-scale crops"
            if split == "train"
            else "native-resolution"
        )
        print(
            f"[DynamicRain] split={split}, pairs={self.sample_count}, "
            f"repeat={self.repeat}, {resolution}"
        )

    def _get_paths(self, index):
        return self.pairs[index]
