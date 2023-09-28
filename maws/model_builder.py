from timm.models.vision_transformer import VisionTransformer
from dataclasses import dataclass, asdict
from torchtext.models.roberta.model import RobertaEncoderConf, RobertaModel
from .model import RobertaIdentityHead, CLIP
import torchtext.transforms
import torch


@dataclass
class ViTConf:
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    num_classes: int = 0


AVAILABLE_MODEL_TYPES = [
    "mae",
    "maws",
    "maws_lit",
]

AVAILABLE_MODELS = {
    "mae": [
        "vit_b16",
    ],
    "maws": [
        "vit_b16",
    ],
    "maws_clip": [
        "vit_b16_xlmr_b",
    ],
}

MODEL_CONFIGS = {
    "vit_b16": ViTConf(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    "vit_b16_xlmr_b": [
        ViTConf(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        RobertaEncoderConf(vocab_size=250002, embedding_dim=768, ffn_dimension=3072, num_attention_heads=12, num_encoder_layers=12),
    ]
}


def build_xlmr_tokenizer(sentence_piece_model_path, vocab_model_path, context_length):
    with open(vocab_model_path, "rb") as f:
        vocab_model_state_dict = torch.load(f, map_location="cpu")

    return torchtext.transforms.Sequential(
        torchtext.transforms.SentencePieceTokenizer(
            sp_model_path=sentence_piece_model_path,
        ),
        torchtext.transforms.VocabTransform(vocab_model_state_dict),
        torchtext.transforms.Truncate(context_length),
        torchtext.transforms.AddToken(token=0, begin=True),
        torchtext.transforms.AddToken(token=2, begin=False),
        torchtext.transforms.ToTensor(padding_value=RobertaEncoderConf.padding_idx),
        torchtext.transforms.PadTransform(
            max_length=context_length + 2, pad_value=RobertaEncoderConf.padding_idx
        ),
    )

def build_model(model_name, model_type, pretrained=False):
    assert model_type in AVAILABLE_MODEL_TYPES
    assert model_name in AVAILABLE_MODELS[model_type]
    model_config = MODEL_CONFIGS[model_name]
    if model_type != "maws_clip":
        # just a ViT
        model = VisionTransformer(**asdict(model_config))
    else:
        vision_encoder = VisionTransformer(**asdict(model_config[0]))
        text_encoder_head = RobertaIdentityHead()
        text_encoder = RobertaModel(model_config[1], text_encoder_head)
        text_tokenizer = build_xlmr_tokenizer(None, None, 100)

        model = CLIP(
            vision_encoder,
            text_encoder,
            text_tokenizer,
            embed_dim=768,
            vision_proj_width=768,
            text_proj_width=768,
        )
    return model
