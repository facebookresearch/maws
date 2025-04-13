# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from maws.model_builder import AVAILABLE_MODELS, build_model


dependencies = ["torch", "torchvision", "torchtext", "timm"]


for model_type in AVAILABLE_MODELS:
    for model_name in AVAILABLE_MODELS[model_type]:
        globals()[f"{model_name}_{model_type}"] = partial(
            build_model, model_name, model_type
        )
