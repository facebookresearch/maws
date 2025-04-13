#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="maws",
    version="1.0.0",
    author="FAIR",
    description='Code and models for the paper "The effectiveness of MAE pre-pretraining for billion-scale pretraining" https://arxiv.org/abs/2303.13496',
    url="https://github.com/facebookresearch/maws",
    install_requires=[
        "torch",
        "torchvision",
        "torchtext",
        "timm==0.9.7",
    ],
    tests_require=[],
    packages=find_packages(exclude=("tests", "tests.*")),
)
