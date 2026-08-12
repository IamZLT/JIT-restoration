"""Multi-scale All-in-One training dataset and balanced batch sampler."""

import math
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Sampler

from data.dataset_dynamic import parse_train_sizes
from data.dataset_utils import AIOTrainDataset
from utils.image_utils import crop_img, data_augmentation


TASK_ORDER = [
    "denoise_15",
    "denoise_25",
    "denoise_50",
    "derain",
    "dehaze",
]


class BalancedMultiScaleBatchSampler(Sampler):
    """Task-balanced indices with one shared HxW crop per batch."""

    def __init__(
        self,
        weights,
        samples_per_epoch,
        batch_size,
        sizes,
        num_replicas=1,
        rank=0,
        seed=0,
        drop_last=True,
    ):
        self.weights = weights
        self.batch_size = batch_size
        self.sizes = tuple(sizes)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = math.ceil(samples_per_epoch / num_replicas)
        if drop_last:
            self.num_samples = (
                self.num_samples // batch_size
            ) * batch_size
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        shard = indices[self.rank : self.total_size : self.num_replicas]
        rng = random.Random(self.seed + self.epoch + 17 + self.rank)
        for start in range(0, len(shard), self.batch_size):
            batch = shard[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            height, width = rng.choice(self.sizes)
            yield [(index, height, width) for index in batch]

    def __len__(self):
        return self.num_samples // self.batch_size


def make_balanced_multiscale_sampler(
    dataset,
    samples_per_epoch,
    batch_size,
    train_sizes,
    num_replicas=1,
    rank=0,
    seed=0,
):
    degradation_ids = [int(sample["de_type"]) for sample in dataset.lr]
    counts = np.bincount(degradation_ids, minlength=len(TASK_ORDER))
    if np.any(counts == 0):
        raise RuntimeError(f"Missing All-in-One task samples: {counts}")
    target_probabilities = np.array(
        [1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3],
        dtype=np.float64,
    )
    weights = torch.tensor(
        [
            target_probabilities[degradation_id]
            / counts[degradation_id]
            for degradation_id in degradation_ids
        ],
        dtype=torch.double,
    )
    print(
        "Raw All-in-One task counts:",
        dict(zip(TASK_ORDER, counts.tolist())),
    )
    print("Dynamic AIO training sizes:", train_sizes)
    return BalancedMultiScaleBatchSampler(
        weights,
        samples_per_epoch,
        batch_size,
        train_sizes,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
    )


class DynamicAIOTrainDataset(AIOTrainDataset):
    """Reuse AIO path discovery; crops follow MultiScale batch sizes."""

    @staticmethod
    def _pad_pair(image_1, image_2, target_h, target_w):
        height = min(image_1.shape[0], image_2.shape[0])
        width = min(image_1.shape[1], image_2.shape[1])
        image_1 = image_1[:height, :width]
        image_2 = image_2[:height, :width]
        pad_h = max(target_h - height, 0)
        pad_w = max(target_w - width, 0)
        if pad_h or pad_w:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            mode = (
                "reflect"
                if height > max(top, bottom) and width > max(left, right)
                else "edge"
            )
            padding = ((top, bottom), (left, right), (0, 0))
            image_1 = np.pad(image_1, padding, mode=mode)
            image_2 = np.pad(image_2, padding, mode=mode)
        return image_1, image_2

    def _crop_to_size(self, image_1, image_2, target_h, target_w):
        image_1, image_2 = self._pad_pair(
            image_1,
            image_2,
            target_h,
            target_w,
        )
        height, width = image_1.shape[:2]
        top = random.randint(0, height - target_h)
        left = random.randint(0, width - target_w)
        slices = np.s_[
            top : top + target_h,
            left : left + target_w,
        ]
        modes = range(8) if target_h == target_w else (0, 1, 4, 5)
        mode = random.choice(tuple(modes))
        image_1 = data_augmentation(image_1[slices], mode).copy()
        image_2 = data_augmentation(image_2[slices], mode).copy()
        return image_1, image_2

    def _crop_single(self, image, target_h, target_w):
        height, width = image.shape[:2]
        pad_h = max(target_h - height, 0)
        pad_w = max(target_w - width, 0)
        if pad_h or pad_w:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            mode = (
                "reflect"
                if height > max(top, bottom) and width > max(left, right)
                else "edge"
            )
            image = np.pad(
                image,
                ((top, bottom), (left, right), (0, 0)),
                mode=mode,
            )
            height, width = image.shape[:2]
        top = random.randint(0, height - target_h)
        left = random.randint(0, width - target_w)
        crop = image[top : top + target_h, left : left + target_w]
        modes = range(8) if target_h == target_w else (0, 1, 4, 5)
        mode = random.choice(tuple(modes))
        return data_augmentation(crop, mode).copy()

    def __getitem__(self, request):
        if not isinstance(request, tuple) or len(request) != 3:
            raise RuntimeError(
                "DynamicAIOTrainDataset requires BalancedMultiScaleBatchSampler"
            )
        index, target_h, target_w = request
        lr_sample = self.lr[index]
        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]

        if deg_type in ("denoise_15", "denoise_25", "denoise_50"):
            clean = crop_img(
                np.array(Image.open(lr_sample["img"]).convert("RGB")),
                base=16,
            )
            clean = self._crop_single(clean, target_h, target_w)
            # Noise is re-applied in the training loop for reproducibility.
            degraded = clean.copy()
        elif deg_type == "dehaze":
            degraded = crop_img(
                np.array(Image.open(lr_sample["img"]).convert("RGB")),
                base=16,
            )
            clean = crop_img(
                np.array(
                    Image.open(
                        self._get_nonhazy_name(lr_sample["img"])
                    ).convert("RGB")
                ),
                base=16,
            )
            degraded, clean = self._crop_to_size(
                degraded,
                clean,
                target_h,
                target_w,
            )
        else:
            hr_sample = self.hr[index]
            degraded = crop_img(
                np.array(Image.open(lr_sample["img"]).convert("RGB")),
                base=16,
            )
            clean = crop_img(
                np.array(Image.open(hr_sample["img"]).convert("RGB")),
                base=16,
            )
            degraded, clean = self._crop_to_size(
                degraded,
                clean,
                target_h,
                target_w,
            )

        return (
            [lr_sample["img"], de_id],
            self.toTensor(np.ascontiguousarray(degraded)),
            self.toTensor(np.ascontiguousarray(clean)),
        )


__all__ = [
    "TASK_ORDER",
    "BalancedMultiScaleBatchSampler",
    "DynamicAIOTrainDataset",
    "make_balanced_multiscale_sampler",
    "parse_train_sizes",
]
