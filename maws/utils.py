# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torch
from torchtext.utils import get_asset_local_path
from matplotlib import pyplot as plt 
import numpy as np
import matplotlib

_STACK = None


def start_inference_mode(device="cpu"):
    global _STACK
    if _STACK is None:
        _STACK = contextlib.ExitStack()
        _STACK.enter_context(torch.inference_mode())


def reset_inference_mode():
    global _STACK
    _STACK.close()
    _STACK = None


def predict_probs_for_image(model, image_path, texts):
    image_path = get_asset_local_path(image_path)
    # cropped_image = model.get_cropped_images(image_path)
    return (model.classify(image_path, texts) * 100).tolist()[0]


def plot_probs(texts, probs, fig_ax, lang_type=None):
    # reverse the order to plot from top to bottom
    probs = probs[::-1]
    texts = texts[::-1]
    probs = np.array(probs)
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig, ax = fig_ax

    font_prop = matplotlib.font_manager.FontProperties(fname=lang_type_to_font_path(lang_type))

    ax.barh(texts, probs, color="darkslateblue", height=0.3)
    ax.barh(texts, 100 - probs, color="silver", height=0.3, left=probs)
    for bar, label, val in zip(ax.patches, texts, probs):
        ax.text(0, bar.get_y() - bar.get_height(), label, color="black", ha = 'left', va = 'center', fontproperties=font_prop) 
        ax.text(bar.get_x() + bar.get_width() + 1, bar.get_y()+bar.get_height()/2, f"{val:.2f} %", fontweight="bold", ha = 'left', va = 'center') 

    ax.axis("off")


def predict_probs_and_plot(model, image_path, texts, plot_image=True, fig_ax=None, lang_type=None):
    if plot_image:
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        ax_1.imshow(model.get_cropped_images(image_path))
        ax_1.axis("off")
    probs = predict_probs_for_image(model, image_path, texts)
    plot_probs(texts, probs, (fig, ax_2), lang_type=lang_type)


def lang_type_to_font_path(lang_type):
    mapping = {
        None: "https://cdn.jsdelivr.net/gh/notofonts/notofonts.github.io/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
        "cjk": "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
        "devanagari": "https://cdn.jsdelivr.net/gh/notofonts/notofonts.github.io/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
        # "emoji": "https://github.com/googlefonts/noto-emoji/blob/main/fonts/NotoColorEmoji-emojicompat.ttf",
    }
    return get_asset_local_path(mapping[lang_type])