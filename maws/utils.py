# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torch


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
