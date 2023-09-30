# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.transforms


class RobertaIdentityHead(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


class CLIP(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        text_tokenizer: nn.Module,
        embed_dim: int,
        vision_encoder_width: Optional[int],
        text_encoder_width: Optional[int],
    ):
        """
        A CLIP style model which places images and texts in the same embedding space.
        CLIP reference: https://arxiv.org/abs/2103.00020

        Args:
            vision_encoder: vision encoding model
            text_encoder: text encoding model
            text_tokenizer: text tokenizer model
            embed_dim: the embedding dimension for the image-text CLIP space
            vision_encoder_width: width of the embeddings returned by the vision encoder
            text_encoder_width: width of the embeddings returned by the text encoder
        """
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if vision_encoder_width is not None:
            self.vision_projection = nn.Parameter(
                torch.empty(vision_encoder_width, embed_dim)
            )
            nn.init.normal_(self.vision_projection, std=vision_encoder_width**-0.5)
        else:
            self.vision_projection = None
        if text_encoder_width is not None:
            self.text_projection = nn.Parameter(
                torch.empty(text_encoder_width, embed_dim)
            )
            nn.init.normal_(self.text_projection, std=text_encoder_width**-0.5)
        else:
            self.text_projection = None

    def encode_images(self, images, normalize=True):
        if isinstance(images, (PIL.Image.Image, str)):
            images = [images]
        if isinstance(images, Sequence):
            assert len(images) > 0
            if isinstance(images[0], str):
                images = [PIL.Image.open(image).convert("RGB") for image in images]
            if isinstance(images[0], PIL.Image.Image):
                transform = self.get_image_transform()
                images = torch.stack([transform(image) for image in images], dim=0)
        else:
            assert isinstance(images, torch.Tensor)
        x = self.vision_encoder(images)
        if self.vision_projection is not None:
            x = x @ self.vision_projection
        if normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x

    @staticmethod
    def get_image_transform():
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=224, interpolation=3),
                torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def get_cropped_images(images):
        if isinstance(images, (PIL.Image.Image, str)):
            images = [images]
        assert isinstance(images, Sequence)
        assert len(images) > 0
        if isinstance(images[0], str):
            images = [PIL.Image.open(image).convert("RGB") for image in images]
        # now we have a list of PIL images
        crop_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=224, interpolation=3),
                torchvision.transforms.CenterCrop(size=224),
            ]
        )
        return [crop_transform(image) for image in images]

    def encode_texts(self, texts, normalize=True):
        texts_tokenized = self.text_tokenizer(texts)
        x = self.text_encoder(texts_tokenized)
        if self.text_projection is not None:
            x = x @ self.text_projection
        if normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x

    def get_logit_scale(self):
        return self.logit_scale.exp()

    def classify(
        self,
        images=None,
        texts=None,
        image_features=None,
        text_features=None,
        return_logits=False,
    ):
        assert (images is None) != (image_features is None)
        assert (texts is None) != (text_features is None)
        if image_features is None:
            image_features = self.encode_images(images)
        if text_features is None:
            text_features = self.encode_texts(texts)
        result = (image_features @ text_features.t()) * self.get_logit_scale()
        if return_logits:
            return result
        return torch.nn.functional.softmax(result, dim=-1)

    def forward(self, images, texts, normalize=True):
        image_features = self.encode_images(images, normalize=normalize)
        text_features = self.encode_texts(texts, normalize=normalize)
        return image_features, text_features, self.get_logit_scale()
