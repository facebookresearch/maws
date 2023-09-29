# MAWS
Code and models for the paper "The effectiveness of MAE pre-pretraining for billion-scale pretraining"

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

## Citation

If you use the SWAG models or if the work is useful in your research, please give us a star and cite:  

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
