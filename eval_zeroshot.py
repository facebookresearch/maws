# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import torch
import torchvision.transforms

from maws.model_builder import build_model
from torchtext.utils import get_asset_local_path
from torchvision.datasets import ImageFolder
from tqdm import tqdm


IN1K_METADATA = {
    "english": {
        "templates": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/templates.npy",
        "classnames_zs": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/classnames_zs.npy",
    },
    "french": {
        "templates": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/templates_openai_fr.npy",
        "classnames_zs": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/classnames_zs_fr.npy",
    }
}


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Zero-shot ImageNet evaluation", add_help=False
    )
    parser.add_argument(
        "--batch-size", "-b", default=25, type=int, help="Batch size per step"
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
    parser.add_argument(
        "--language", "-l", default="english", type=str, help="The language used for the labels (default is english)"
    )
    return parser


def numpy_load(path):
    with open(path, "rb") as fh:
        data = np.load(fh, allow_pickle=True)
    return data


def gen_label_strings(templates_file_path, label_names_file_path):
    label_names = numpy_load(label_names_file_path)
    templates = numpy_load(templates_file_path)
    # label_names is a list/array of length num_classes
    # each element is a list of names for that class
    # e.g. [ ["dog", "puppy"], ["cat", "kitten, "tabby"]]
    if isinstance(templates, np.ndarray):
        assert templates.ndim == 1
        templates = templates.tolist()
    per_label_templates = []
    for label_id in range(len(label_names)):
        formatted_templates = [
            t.format(l) for t in templates for l in label_names[label_id]
        ]
        per_label_templates.append(formatted_templates)

    return per_label_templates


def get_per_label_text_embeddings(per_label_templates, clip_model):
    per_class_text_embeddings = []
    with torch.no_grad():
        for label_templates in tqdm(per_label_templates):
            class_embeddings = clip_model.encode_texts(
                texts=label_templates, normalize=False
            )
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            per_class_text_embeddings.append(class_embeddings)

    per_class_text_embeddings = torch.stack(per_class_text_embeddings, dim=0)
    return per_class_text_embeddings


def forward_val(images, clip_model, per_class_text_embeddings):
    image_feature = clip_model.encode_images(images=images)
    logits_per_image = image_feature @ per_class_text_embeddings.t()
    return logits_per_image


def make_val_dataloader(args):
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=224, interpolation=3),
            torchvision.transforms.CenterCrop(size=224),
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
    print("Downloading and building the clip model:", args.model)
    clip_model = build_model(args.model, "maws_clip")
    clip_model = clip_model.to(args.device)
    clip_model = clip_model.eval()

    print("Retrieving the ImageNet meta data for the chosen language...")
    language = args.language
    if language not in IN1K_METADATA:
        available = list(IN1K_METADATA.keys())
        raise ValueError(f"Unsupported language, please use one of: {available}")
    templates_path = get_asset_local_path(IN1K_METADATA[language]["templates"])
    labels_path = get_asset_local_path(IN1K_METADATA[language]["classnames_zs"])
    per_label_templates = gen_label_strings(templates_path, labels_path)

    print("Generating text embedding for class templates...")
    per_class_text_embeddings = get_per_label_text_embeddings(
        per_label_templates, clip_model
    )

    print("Loading the dataset...")
    val_loader = make_val_dataloader(args)

    total_top1 = 0
    total_images = 0
    print("Performing zero-shot inference on In1k validation split...")
    tqdm_loader = tqdm(val_loader)
    for batch in tqdm_loader:
        img_trans, target = batch
        img_trans = img_trans.to(args.device)
        target = target.to(args.device)
        logits_per_image = forward_val(img_trans, clip_model, per_class_text_embeddings)
        pred = logits_per_image.argmax(dim=1)
        correct = pred.eq(target).sum()
        total_top1 += correct.item()
        total_images += img_trans.size(0)
        tqdm_loader.set_description(f"top-1: {compute_accuracy(total_top1, total_images):.2f} %")

    print("Zero shot ImageNet-1k top-1 accuracy:", compute_accuracy(total_top1, total_images))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
