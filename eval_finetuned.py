# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import torch
import torchvision.transforms

from maws.model_builder import build_model
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(description="ImageNet evaluation", add_help=False)
    parser.add_argument(
        "--batch_size", "-b", default=25, type=int, help="Batch size per step"
    )
    parser.add_argument("--img_size", "-i", type=int, help="Image size")
    parser.add_argument(
        "--no_rescale",
        "-nr",
        help="Do not rescale images and instead use resize short size + center crop",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="Number of data loading workers per process",
    )
    parser.add_argument(
        "--dataset_path",
        "-p",
        required=True,
        type=str,
        help="Path to the imagenet-1k val root folder",
    )
    parser.add_argument(
        "--model", "-m", default="vit_b16_xlmr_b", type=str, help="Model to evaluate"
    )
    parser.add_argument(
        "--device", "-d", default="cuda", type=str, help="Device to run evaluation on"
    )
    return parser


def numpy_load(path):
    with open(path, "rb") as fh:
        data = np.load(fh, allow_pickle=True)
    return data


def make_val_dataloader(args):
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=args.img_size, interpolation=3),
            torchvision.transforms.CenterCrop(size=args.img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
        if args.no_rescale
        else [
            torchvision.transforms.Resize(
                size=[args.img_size, args.img_size], interpolation=3
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    img_ds = ImageFolder(root=args.dataset_path, transform=image_transform)

    return torch.utils.data.DataLoader(
        img_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )


def compute_accuracy(total_top1, total_images):
    return 100 * total_top1 / total_images


def main(args):
    print("Downloading and building the model:", args.model)
    model = build_model(args.model, "maws")
    model = model.to(args.device)
    model = model.eval().half()

    val_loader = make_val_dataloader(args)

    total_top1 = 0
    total_images = 0
    print("Performing inference on In1k validation split...")
    tqdm_loader = tqdm(val_loader)
    with torch.amp.autocast(device_type=args.device):
        for batch in tqdm_loader:
            img_trans, target = batch
            img_trans = img_trans.to(args.device)
            target = target.to(args.device)
            logits_per_image = model(img_trans)
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += img_trans.size(0)
            tqdm_loader.set_description(
                f"top-1: {compute_accuracy(total_top1, total_images):.2f} %"
            )

    print("ImageNet-1k top-1 accuracy:", compute_accuracy(total_top1, total_images))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
