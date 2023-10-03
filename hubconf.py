from functools import partial

from maws.model_builder import AVAILABLE_MODELS, build_model


dependencies = ["torch", "torchvision", "torchtext", "timm"]


for model_type in AVAILABLE_MODELS:
    for model_name in AVAILABLE_MODELS[model_type]:
        globals()[f"{model_name}_{model_type}"] = partial(
            build_model, model_name, model_type
        )
