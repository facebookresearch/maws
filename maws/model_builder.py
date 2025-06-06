# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass

import torch
from timm.models.vision_transformer import VisionTransformer

try:
    from torchtext import transforms
    from torchtext.models.roberta.model import RobertaEncoderConf, RobertaModel

    TORCHTEXT_AVAILABLE = True
except ImportError:
    import warnings

    warnings.warn("torchtext not found, clip models will not work")
    TORCHTEXT_AVAILABLE = False

from .utils import get_asset_local_path

from .model import CLIP, RobertaIdentityHead


XLMR_SP_MODEL_URL = "https://dl.fbaipublicfiles.com/maws/pretrain/clip/xlmr_tokenizer/xlmr.sentencepiece.bpe.model"
XLMR_VOCAB_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/maws/pretrain/clip/xlmr_tokenizer/xlmr.vocab.pt"
)


@dataclass
class ViTConf:
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    num_classes: int = 0
    img_size: int = 224


@dataclass
class CLIPConf:
    embed_dim: int
    vision_encoder_width: int
    text_encoder_width: int


AVAILABLE_MODELS = {
    "mae_in1k": {
        "vit_2b14": "https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt",
    },
    "mae_in21k": {
        "vit_l16": "https://dl.fbaipublicfiles.com/maws/pretrain/mae_in21k/vit_l16.pt",
    },
    "mae": {
        "vit_b16": "https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_b16.pt",
        "vit_l16": "https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_l16.pt",
        "vit_h14": "https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_h14.pt",
        "vit_2b14": "https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_2b14.pt",
        "vit_6.5b14": "https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_6.5b14.pt",
    },
    "maws": {
        "vit_b16": "https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_b16.pt",
        "vit_l16": "https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_l16.pt",
        "vit_h14": "https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_h14.pt",
        "vit_2b14": "https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_2b14.pt",
        "vit_6.5b14": "https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_6.5b14.pt",
        "vit_b16_ft_in1k": "https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_b16_512.pt",
        "vit_l16_ft_in1k": "https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_l16_512.pt",
        "vit_h14_ft_in1k": "https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_h14_518.pt",
        "vit_2b14_ft_in1k": "https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_2b14_518.pt",
        "vit_6.5b14_ft_in1k": "https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_6.5b14_518.pt",
    },
    "maws_clip": {
        "vit_b16_xlmr_b": "https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_b16_xlmr_b.pt",
        "vit_l16_xlmr_l": "https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_l16_xlmr_l.pt",
        "vit_h14_xlmr_l": "https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_h14_xlmr_l.pt",
        "vit_2b14_xlmr_l": "https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_2b14_xlmr_l.pt",
    },
}

MODEL_CONFIGS = {
    "vit_b16": ViTConf(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    "vit_l16": ViTConf(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    "vit_h14": ViTConf(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    "vit_2b14": ViTConf(patch_size=14, embed_dim=2560, depth=24, num_heads=32),
    "vit_6.5b14": ViTConf(patch_size=14, embed_dim=4096, depth=32, num_heads=32),
    "vit_b16_ft_in1k": ViTConf(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
        img_size=512,
    ),
    "vit_l16_ft_in1k": ViTConf(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=1000,
        img_size=512,
    ),
    "vit_h14_ft_in1k": ViTConf(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        num_classes=1000,
        img_size=518,
    ),
    "vit_2b14_ft_in1k": ViTConf(
        patch_size=14,
        embed_dim=2560,
        depth=24,
        num_heads=32,
        num_classes=1000,
        img_size=518,
    ),
    "vit_6.5b14_ft_in1k": ViTConf(
        patch_size=14,
        embed_dim=4096,
        depth=32,
        num_heads=32,
        num_classes=1000,
        img_size=518,
    ),
}

if TORCHTEXT_AVAILABLE:
    MODEL_CONFIGS.update(
        {
            "vit_b16_xlmr_b": [
                CLIPConf(
                    embed_dim=768, vision_encoder_width=768, text_encoder_width=768
                ),
                ViTConf(patch_size=16, embed_dim=768, depth=12, num_heads=12),
                RobertaEncoderConf(
                    vocab_size=250002,
                    embedding_dim=768,
                    ffn_dimension=3072,
                    num_attention_heads=12,
                    num_encoder_layers=12,
                ),
            ],
            "vit_l16_xlmr_l": [
                CLIPConf(
                    embed_dim=1024, vision_encoder_width=1024, text_encoder_width=1024
                ),
                ViTConf(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
                RobertaEncoderConf(
                    vocab_size=250002,
                    embedding_dim=1024,
                    ffn_dimension=4096,
                    num_attention_heads=16,
                    num_encoder_layers=24,
                ),
            ],
            "vit_h14_xlmr_l": [
                CLIPConf(
                    embed_dim=1024, vision_encoder_width=1280, text_encoder_width=1024
                ),
                ViTConf(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
                RobertaEncoderConf(
                    vocab_size=250002,
                    embedding_dim=1024,
                    ffn_dimension=4096,
                    num_attention_heads=16,
                    num_encoder_layers=24,
                ),
            ],
            "vit_2b14_xlmr_l": [
                CLIPConf(
                    embed_dim=2048, vision_encoder_width=2560, text_encoder_width=1024
                ),
                ViTConf(patch_size=14, embed_dim=2560, depth=24, num_heads=32),
                RobertaEncoderConf(
                    vocab_size=250002,
                    embedding_dim=1024,
                    ffn_dimension=4096,
                    num_attention_heads=16,
                    num_encoder_layers=24,
                ),
            ],
        }
    )


def build_xlmr_tokenizer(sentence_piece_model_path, vocab_model_path, context_length):
    vocab_model = torch.load(get_asset_local_path(vocab_model_path), map_location="cpu")
    return transforms.Sequential(
        transforms.SentencePieceTokenizer(
            sp_model_path=get_asset_local_path(sentence_piece_model_path),
        ),
        transforms.VocabTransform(vocab_model),
        transforms.Truncate(context_length),
        transforms.AddToken(token=0, begin=True),
        transforms.AddToken(token=2, begin=False),
        transforms.ToTensor(padding_value=RobertaEncoderConf.padding_idx),
        transforms.PadTransform(
            max_length=context_length + 2, pad_value=RobertaEncoderConf.padding_idx
        ),
    )


def build_model(model_name, model_type, pretrained=True):
    assert model_type in AVAILABLE_MODELS
    assert model_name in AVAILABLE_MODELS[model_type]
    model_config = MODEL_CONFIGS[model_name]
    if model_type != "maws_clip":
        model = VisionTransformer(**asdict(model_config))
    else:
        vision_encoder = VisionTransformer(**asdict(model_config[1]))
        text_encoder_head = RobertaIdentityHead()
        text_encoder = RobertaModel(model_config[2], text_encoder_head)
        text_tokenizer = build_xlmr_tokenizer(
            sentence_piece_model_path=XLMR_SP_MODEL_URL,
            vocab_model_path=XLMR_VOCAB_MODEL_URL,
            context_length=100,
        )

        model = CLIP(
            vision_encoder, text_encoder, text_tokenizer, **asdict(model_config[0])
        )
    if pretrained:
        checkpoint = torch.load(
            get_asset_local_path(AVAILABLE_MODELS[model_type][model_name]),
            map_location="cpu",
        )
        model.load_state_dict(checkpoint)
    return model
