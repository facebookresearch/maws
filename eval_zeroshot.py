# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os

import numpy as np
import torch
import torchvision.transforms
from maws.inference_utils import start_inference_mode

from maws.model_builder import build_model
from PIL import Image
from torch.utils.data import Dataset
from torchtext.utils import get_asset_local_path
from tqdm import tqdm

S3_PATHS = {
    "templates_openai": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/templates_openai.npy",
    "classnames_zs": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/classnames_zs.npy",
    "val_labels": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/val_labels.npy",
    "val_images_local": "https://dl.fbaipublicfiles.com/maws/zero_shot_in1k_assets/val_images_local.npy",
}


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="0-shot In1k evaluations", add_help=False
    )
    parser.add_argument("--batch-size", default=32, type=int, help="batch_size")
    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument(
        "--in1k-dir",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="path to the imagenet-1k root folder containing train and val splits",
    )
    parser.add_argument(
        "--model", default="vit_b16_xlmr_b", type=str, help="model to evaluate on"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="device to evaluate on"
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


class In1kValDataset(Dataset):
    def __init__(
        self,
        image_file,
        label_file,
        img_folder,
        img_transforms=None,
    ):
        super().__init__()
        self.img_paths = numpy_load(image_file)
        self.labels = numpy_load(label_file)
        self.img_folder = img_folder
        self.img_transforms = img_transforms

        assert len(self.img_paths) == len(self.labels)

    def __getitem__(self, idx):
        # TODO: Add path in root and remove prefix
        img_path = os.path.join(self.img_folder, self.img_paths[idx])
        img = Image.open(img_path).convert("RGB")

        if self.img_transforms is not None:
            img = self.img_transforms(img)

        label = self.labels[idx]
        return {"imgs": img, "labels": label}

    def __len__(self):
        return len(self.img_paths)


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
    image_file = get_asset_local_path(S3_PATHS["val_images_local"])
    label_file = get_asset_local_path(S3_PATHS["val_labels"])

    img_ds = In1kValDataset(
        image_file, label_file, img_folder=args.in1k_dir, img_transforms=image_transform
    )

    return torch.utils.data.DataLoader(
        img_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )


def main(args):

    print("Downloading and building the clip model: ", args.model)
    clip_model = build_model(args.model, "maws_clip")
    clip_model = clip_model.to(args.device)
    clip_model = clip_model.eval()

    templates_path = get_asset_local_path(S3_PATHS["templates_openai"])
    labels_path = get_asset_local_path(S3_PATHS["classnames_zs"])
    per_label_templates = gen_label_strings(templates_path, labels_path)

    print("Generating text embedding for class templates...")
    per_class_text_embeddings = get_per_label_text_embeddings(
        per_label_templates, clip_model
    )

    val_loader = make_val_dataloader(args)

    total_top1 = 0
    total_images = 0
    print("Performing zero-shot inference on In1k validation split...")
    for batch in tqdm(val_loader):
        img_trans = batch["imgs"]
        img_trans = img_trans.to(args.device)
        target = batch["labels"].to(args.device)
        logits_per_image = forward_val(img_trans, clip_model, per_class_text_embeddings)
        pred = logits_per_image.argmax(dim=1)
        correct = pred.eq(target).sum()
        total_top1 += correct.item()
        total_images += img_trans.size(0)

    print("Zero shopt In1k top-1 accurarcy:", 100 * float(total_top1) / total_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "0-shot In1k evaluations", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
