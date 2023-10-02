# MAWS
Code and models for the paper "The effectiveness of MAE pre-pretraining for billion-scale pretraining"

# Using the Models

We have models available for:
1. MAE (pre-)pretraining
1. MAWS (MAE→WSP) pretraining, i.e. MAE (pre-)pretraining followed by WSP pretraining
1. MAWS pretrained models along with LiT aligned text encoders for CLIP style zero shot classification

To access a model, specify the model architecture and the model type: 
```python
from maws.model import build_model

# build a MAWS model with CLIP capabilities
clip_model = build_model("vit_b16_xlmr_b", "maws_clip")

# build a MAWS model
maws_model = build_model("vit_b16", "maws")

# build an MAE model
mae_model = build_model("vit_b16", "mae")
```

We also have an example notebook for using the models in a zero shot manner, please refer to [`clip_example.ipynb`](clip_example.ipynb).

We list down all the available models, their names, and direct download links next.

**TODO: Add torchhub support**


# MAWS pretrained models

Model | IN1k 224px linear | IN1k 512/518px finetuned | Text encoder | IN1k 224px 0-shot 
--- | --- | --- | --- | --- 
ViT-B | 83.3 | 86.4* | XLMR-B | 74.9
ViT-L | 86.1 | 88.8* | XLMR-L | 79.7
ViT-H | 87.5 | 89.4 | XLMR-L | 81.1
ViT-2B | 88.1 | 89.7 | XLMR-L | 82.1

*tentative

# MAE pretrained models

Model | IN1k 224px finetuned
--- | --- 
ViT-B | 83.5
ViT-L | 86.1
ViT-H | 87.4
ViT-2B | 87.8
ViT-2B-IN1k | 87.4

# Installation instructions

```bash
conda create --name maws python=3.10
conda activate maws
pip install torch torchvision torchtext
pip install timm==0.9.7
pip install jupyter ipywidgets
```

# Zero-shot evaluation on In1k
```bash
python eval_zeroshot.py --model vit_b16_xlmr_b --in1k-dir <Path to your 1n1k root folder containing "train" and "va" directories>

# For evaluating on other models, please refer to all the available model names in maws/model_builder.py
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
