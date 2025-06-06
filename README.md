# MAWS

[[`Paper`](https://arxiv.org/abs/2303.13496)] [[`Colab`](https://colab.research.google.com/github/facebookresearch/maws/blob/main/clip_example.ipynb)] [[`BibTex`](#citation)] [[`Website`](https://facebookresearch.github.io/maws/)]

Repository for the strong foundational MAWS + MAE models at all sizes ranging from <100M parameters to >6.5B parameters, from the paper [The effectiveness of MAE pre-pretraining for billion-scale pretraining](https://arxiv.org/abs/2303.13496). Models are available for both MAE pre-pretraining and the follow up WSP pretraining, MAE→WSP a.k.a. MAWS (Masked Autoencoding → Weakly Supervised pretraining).
<p align="center">
  <img width="539" alt="image" src="https://github.com/facebookresearch/maws/assets/13458796/69afa2ca-9976-4c64-9814-1f906be05e36">
</p>

## Getting started

To get started with playing with our models immediately, we have a notebook available to play with on [Colab](https://colab.research.google.com/github/facebookresearch/maws/blob/main/clip_example.ipynb), or [locally](clip_example.ipynb) for running our models in zero-shot mode.

For building any of our models, select which model type you would like to build. We have models available for:
1. `model_type="maws"`: MAWS (MAE→WSP) pretraining, i.e. MAE pre-pretraining followed by WSP pretraining. We also have ImageNet-1k finetuned weights for MAWS models using the same model type.
1. `model_type="maws_clip"`: MAWS pretrained models along with LiT aligned text encoders for CLIP style zero shot classification
1. `model_type="mae"`: MAE pretrained models
1. `model_type="mae_in1k"`: MAE pretrained on ImageNet-1k models

To access a model, specify the model architecture and the model type: 
```python
from maws.model_builder import build_model

# build a MAWS model with CLIP capabilities (via an aligned text encoder)
clip_model = build_model("vit_b16_xlmr_b", "maws_clip")

# build a MAWS model
maws_model = build_model("vit_b16", "maws")

# build a MAWS model finetuned on IN1k
maws_in1k_model = build_model("vit_b16_ft_in1k", "maws")

# build an MAE model
mae_model = build_model("vit_b16", "mae")
```

The models are also available via torch.hub:
```python
# build a MAWS model with CLIP capabilities (via an aligned text encoder)
clip_model = torch.hub.load("facebookresearch/maws", model="vit_b16_xlmr_b_maws_clip")

# build a MAWS model
maws_model = torch.hub.load("facebookresearch/maws", model="vit_b16_maws")

# build a MAWS model finetuned on IN1k
maws_model = torch.hub.load("facebookresearch/maws", model="vit_b16_ft_in1k_maws")

# build an MAE model
mae_model = torch.hub.load("facebookresearch/maws", model="vit_b16_mae")
```

We list down all the available models and direct download links in the following section.

### Installation instructions

```bash
conda create --name maws python=3.10
conda activate maws
pip install torch torchvision torchtext
pip install timm==0.9.7
# for demo
pip install jupyter ipywidgets matplotlib
```
## WARNING
Torchtext has been deprecated which has broken clip model support. If you run without torchtext, all models which aren't clip based will work fine!


## Available models
### MAWS pretrained models

Model | Pretrained name + weights | IN1k 224px linear top-1 | IN1k 512/518px finetuned name + weights | IN1k 512/518px finetuned top-1 | Text encoder | 0-Shot name + weights | IN1k 224px 0-shot top-1
--- | --- | --- | --- | --- | --- | --- | ---
ViT-B | [vit_b16](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_b16.pt) | 83.3 | [vit_b16_ft_in1k](https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_b16_512.pt) | 86.8 | XLMR-B | [vit_b16_xlmr_b](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_b16_xlmr_b.pt) | 74.9
ViT-L | [vit_l16](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_l16.pt) | 86.1 | [vit_l16_ft_in1k](https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_l16_512.pt) | 88.8 | XLMR-L | [vit_l16_xlmr_l](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_l16_xlmr_l.pt) | 79.7
ViT-H | [vit_h14](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_h14.pt) | 87.5 | [vit_h14_ft_in1k](https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_h14_518.pt) | 89.5 | XLMR-L | [vit_h14_xlmr_l](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_h14_xlmr_l.pt) | 81.1
ViT-2B | [vit_2b14](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_2b14.pt) | 88.1 | [vit_2b14_ft_in1k](https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_2b14_518.pt) | 89.8 | XLMR-L | [vit_2b14_xlmr_l](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_2b14_xlmr_l.pt) | 82.1
ViT-6.5B | [vit_6.5b14](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_6.5b14.pt) | 88.6 | [vit_6.5b14_ft_in1k](https://dl.fbaipublicfiles.com/maws/finetune/in1k/maws/vit_6.5b14_518.pt) | 90.1

### MAE pretrained models

Model | Pretrained name + weights | IN1k 224px finetuned top-1
--- | --- | ---
ViT-B | [vit_b16](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_b16.pt) | 83.5
ViT-L | [vit_l16](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_l16.pt) | 86.1
ViT-H | [vit_h14](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_h14.pt) | 87.4
ViT-2B | [vit_2b14](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_2b14.pt) | 87.8
ViT-6.5B | [vit_6.5b14](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_6.5b14.pt) | 88.3

### MAE pretrained on ImageNet-1k

Model | Pretrained name + weights | IN1k 224px finetuned top-1
--- | --- | ---
ViT-2B | [vit_2b14](https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt) | 87.4

### MAE pretrained on ImageNet-21k

Model | Model name + weights | IN1k 512px finetuned
--- | --- | ---
ViT-L | [vit_l16](https://dl.fbaipublicfiles.com/maws/pretrain/mae_in21k/vit_l16.pt) | 86.9

## Evaluation on ImageNet-1k

### Finetuned
We share weights for the MAWS models finetuned on ImageNet-1k at high resolution (512px for ViT-B, ViT-L and 518px for ViT-H, ViT-2B, ViT-6.5B). `$IN1K_VAL_PATH` should be the path to the ImageNet-1k val root folder.

```bash
python eval_finetuned.py -m vit_b16_ft_in1k -i 512 -b 25 -p $IN1K_VAL_PATH
# ImageNet-1k top-1 accuracy: 86.832

python eval_finetuned.py -m vit_l16_ft_in1k -i 512 -b 10 -p $IN1K_VAL_PATH
# ImageNet-1k top-1 accuracy: 88.796

python eval_finetuned.py -m vit_h14_ft_in1k -i 518 -b 5 -p $IN1K_VAL_PATH
# ImageNet-1k top-1 accuracy: 89.502

python eval_finetuned.py -m vit_2b14_ft_in1k -i 518 -b 5 -p $IN1K_VAL_PATH
# ImageNet-1k top-1 accuracy: 89.752

python eval_finetuned.py -m vit_6.5b14_ft_in1k -i 518 -b 5 -p $IN1K_VAL_PATH
# ImageNet-1k top-1 accuracy: 90.064
```

### Zero-shot
Please refer to all the available model names in the [MAWS Pretrained models](#maws-pretrained-models) section. `$IN1K_VAL_PATH` should be the path to the ImageNet-1k val root folder.

```bash
python eval_zeroshot.py -m vit_b16_xlmr_b -b 25 -p $IN1K_VAL_PATH
# Zero shot ImageNet-1k top-1 accuracy: 74.888

# Trying the french language instead with a larger model on a 32GB V100
python eval_zeroshot.py -m vit_2b14_xlmr_l --language french -b 5 -p $IN1K_VAL_PATH
# Zero shot ImageNet-1k top-1 accuracy: 62.622
```

## Citation

If you use our models or if the work is useful in your research, please give us a star and cite:

```bibtex
@inproceedings{singh2023effectiveness,
    title={The effectiveness of MAE pre-pretraining for billion-scale pretraining},
    author={Singh, Mannat and Duval, Quentin and Alwala, Kalyan Vasudev and Fan, Haoqi and Aggarwal, Vaibhav and Adcock, Aaron and Joulin, Armand and Doll{\'a}r, Piotr and Feichtenhofer, Christoph and Girshick, Ross and Girdhar, Rohit and Misra, Ishan},
    booktitle={ICCV},
    year={2023}
}
```

## License
Our models are released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details.
