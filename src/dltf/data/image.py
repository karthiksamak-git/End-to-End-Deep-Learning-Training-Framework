"""Fashion-MNIST pipeline: stratified splits, train-only augmentation, seeded loaders.

Images stay at their native 28x28 resolution: that is what the exported ONNX model
expects and what the browser playground draws on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dltf.config import DataConfig
from dltf.data import DataBundle
from dltf.data.splits import stratified_split, stratified_subsample
from dltf.seed import make_generator, seed_worker

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
IMAGE_SHAPE = (1, 28, 28)


def build_transforms(
    *,
    augmentation: bool,
    rotation_degrees: float = 10.0,
    translate: float = 0.1,
    mean: float = 0.2860,
    std: float = 0.3530,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transform, eval_transform)``.

    Augmentation is applied to the training split only; validation and test use the
    deterministic transform so metrics are comparable across epochs.
    """
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    if not augmentation:
        return eval_tf, eval_tf
    train_tf = transforms.Compose(
        [
            transforms.RandomRotation(rotation_degrees, fill=0),
            transforms.RandomAffine(degrees=0, translate=(translate, translate), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    return train_tf, eval_tf


def compute_normalization(
    dataset: datasets.FashionMNIST,
    indices: np.ndarray,
    *,
    seed: int = 42,
    max_samples: int = 4000,
) -> tuple[float, float]:
    """Estimate per-channel mean/std from (a sample of) the training split."""
    rng = np.random.default_rng(seed)
    sample = indices
    if indices.size > max_samples:
        sample = rng.choice(indices, size=max_samples, replace=False)
    total = 0.0
    total_sq = 0.0
    count = 0
    for idx in sample:
        img, _ = dataset[int(idx)]
        arr = np.asarray(img, dtype=np.float64)
        total += arr.sum()
        total_sq += (arr**2).sum()
        count += arr.size
    mean = total / count
    var = max(total_sq / count - mean**2, 1e-8)
    return float(mean), float(np.sqrt(var))


def make_image_bundle(
    cfg: DataConfig,
    *,
    data_dir: str | Path = "data",
    seed: int = 42,
    batch_size: int | None = None,
) -> DataBundle:
    """Build the Fashion-MNIST :class:`DataBundle`.

    Args:
        cfg: Data settings (subset caps, augmentation, fractions).
        data_dir: Download/cache directory (gitignored).
        seed: Controls the split and the DataLoader shuffle stream.
        batch_size: Overrides ``cfg.batch_size`` (used by the HPO trials).
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_train = datasets.FashionMNIST(
        root=str(root), train=True, download=cfg.download, transform=transforms.ToTensor()
    )
    test_set = datasets.FashionMNIST(
        root=str(root), train=False, download=cfg.download, transform=transforms.ToTensor()
    )
    targets = np.asarray(raw_train.targets, dtype=np.int64)

    split = stratified_split(
        targets,
        val_fraction=cfg.val_fraction,
        test_fraction=0.0,  # the official test split is the held-out set
        seed=seed,
    )
    train_idx = stratified_subsample(
        split["train"], targets, max_samples=cfg.subset_train, seed=seed
    )
    val_idx = stratified_subsample(split["val"], targets, max_samples=cfg.subset_val, seed=seed)
    mean, std = compute_normalization(raw_train, train_idx, seed=seed)
    train_tf, eval_tf = build_transforms(
        augmentation=cfg.augmentation,
        rotation_degrees=cfg.rotation_degrees,
        translate=cfg.translate,
        mean=mean,
        std=std,
    )

    train_base = datasets.FashionMNIST(
        root=str(root), train=True, download=False, transform=train_tf
    )
    eval_base = datasets.FashionMNIST(
        root=str(root), train=True, download=False, transform=eval_tf
    )
    test_base = datasets.FashionMNIST(
        root=str(root), train=False, download=False, transform=eval_tf
    )

    test_targets = np.asarray(test_set.targets, dtype=np.int64)
    test_idx = stratified_subsample(
        np.arange(test_targets.size), test_targets, max_samples=cfg.subset_test, seed=seed
    )

    generator = make_generator(seed)
    common: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "worker_init_fn": seed_worker,
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_base, train_idx.tolist()),
        batch_size=batch_size or cfg.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        Subset(eval_base, val_idx.tolist()),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        Subset(test_base, test_idx.tolist()),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        **common,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_shape=IMAGE_SHAPE,
        num_classes=len(FASHION_CLASSES),
        class_names=list(FASHION_CLASSES),
        task="image",
        test_dataset=test_loader.dataset,
        generator=generator,
        sizes={"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
        meta={
            "dataset": "Fashion-MNIST",
            "image_mean": mean,
            "image_std": std,
            "image_shape": list(IMAGE_SHAPE),
            "preprocessing": "ToTensor -> Normalize(mean,std); training split additionally "
            "gets RandomRotation(10) + RandomAffine(translate=0.1)",
            "split_indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            },
            "test_indices": test_idx.tolist(),
        },
    )
