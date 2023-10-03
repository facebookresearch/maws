# MAWS
Models for the paper "The effectiveness of MAE pre-pretraining for billion-scale pretraining" for both MAE pre-pretraining and the follow up WSP pretraining, MAE→WSP, which we call MAWS (Masked Autoencoding → Weakly Supervised pretraining).

## Getting started

To get started, select which model type you would like to build. We have models available for:
1. `model_type="maws"`: MAWS (MAE→WSP) pretraining, i.e. MAE pre-pretraining followed by WSP pretraining
1. `model_type="maws_clip"`: MAWS pretrained models along with LiT aligned text encoders for CLIP style zero shot classification
1. `model_type="mae"`: MAE pretrained models
1. `model_type="mae_in1k"`: MAE pretrained on ImageNet-1k models

To access a model, specify the model architecture and the model type: 
```python
from maws.model import build_model

# build a MAWS model with CLIP capabilities (via an aligned text encoder)
clip_model = build_model("vit_b16_xlmr_b", "maws_clip")

# build a MAWS model
maws_model = build_model("vit_b16", "maws")

# build an MAE model
mae_model = build_model("vit_b16", "mae")
```

We also have an example notebook for using the models in a zero shot manner, please refer to [`clip_example.ipynb`](clip_example.ipynb).

The models are also available via torch.hub:
```python
# build a MAWS model with CLIP capabilities (via an aligned text encoder)
clip_model = torch.hub.load("facebookresearch/maws", model="vit_b16_maws_clip")

# build a MAWS model
maws_model = torch.hub.load("facebookresearch/maws", model="vit_b16_maws")

# build an MAE model
mae_model = torch.hub.load("facebookresearch/maws", model="vit_b16_mae")
```

We list down all the available models and direct download links in the next section.

### Installation instructions

```bash
conda create --name maws python=3.10
conda activate maws
pip install torch torchvision torchtext
pip install timm==0.9.7
# for demo
pip install jupyter ipywidgets matplotlib
```

## Available models
### MAWS pretrained models

Model | Model name + weights | IN1k 224px linear | IN1k 512/518px finetuned | Text encoder | Model name + weights | IN1k 224px 0-shot 
--- | --- | --- | --- | --- | --- | ---
ViT-B | [vit_b16](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_b16.pt) | 83.3 | 86.4 | XLMR-B | [vit_b16_xlmr_b](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_b16_xlmr_b.pt) | 74.9
ViT-L | [vit_l16](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_l16.pt) | 86.1 | 88.8 | XLMR-L | [vit_l16_xlmr_l](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_l16_xlmr_l.pt) | 79.7
ViT-H | [vit_h14](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_h14.pt) | 87.5 | 89.4 | XLMR-L | [vit_h14_xlmr_l](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_h14_xlmr_l.pt) | 81.1
ViT-2B | [vit_2b14](https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_2b14.pt) | 88.1 | 89.7 | XLMR-L | [vit_2b14_xlmr_l](https://dl.fbaipublicfiles.com/maws/pretrain/clip/vit_2b14_xlmr_l.pt) | 82.1

### MAE pretrained models

Model | Model name + weights | IN1k 224px finetuned
--- | --- | ---
ViT-B | [vit_b16](https://dl.fbaipublicfiles.com/mae/pretrain/mae/vit_b16.pt) | 83.5
ViT-L | [vit_l16](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_l16.pt) | 86.1
ViT-H | [vit_h14](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_h14.pt) | 87.4
ViT-2B | [vit_2b14](https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_2b14.pt) | 87.8

### MAE pretrained on ImageNet-1k

Model | Model name + weights | IN1k 224px finetuned
--- | --- | ---
ViT-2B | [vit_2b14](https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt) | 87.4


## Zero-shot evaluation on ImageNet-1k

Please refer to all the available model names in the [MAWS Pretrained models section](maws-pretrained-models). `$IN1K_VAL_PATH` should be the path to the ImageNet-1k val root folder.

```bash
python eval_zeroshot.py -m vit_b16_xlmr_b -p $IN1K_VAL_PATH
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
