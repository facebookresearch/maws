# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


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
        """ """
        x = self.vision_encoder(images)
        if self.vision_projection is not None:
            x = x @ self.vision_projection
        if normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x

    def encode_texts(self, texts, normalize=True, skip_tokenizer=False):
        if not skip_tokenizer:
            texts_tokenized = self.text_tokenizer(texts)
        else:
            texts_tokenized = texts
        model_device = next(self.parameters()).device
        x = self.text_encoder(texts_tokenized.to(model_device))
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
