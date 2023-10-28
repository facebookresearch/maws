---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

{% include open_links.html %}

\
\
**Abstract:**

_This paper revisits the standard pretrain-then-finetune paradigm used in computer vision for visual recognition tasks. Typically, state-of-the-art foundation models are pretrained using large scale (weakly) supervised datasets with billions of images. We introduce an additional pre-pretraining stage that is simple and uses the self-supervised MAE technique to initialize the model. While MAE has only been shown to scale with the size of models, we find that it scales with the size of the training dataset as well. Thus, our MAE-based pre-pretraining scales with both model and data size making it applicable for training foundation models. Pre-pretraining consistently improves both the model convergence and the downstream transfer performance across a range of model scales (millions to billions of parameters), and dataset sizes (millions to billions of images). We measure the effectiveness of pre-pretraining on 10 different visual recognition tasks spanning image classification, video recognition, object detection, low-shot classification and zero-shot recognition. Our largest model achieves new state-of-the-art results on iNaturalist-18 (91.3%), 1-shot ImageNet-1k (62.1%), and zero-shot transfer on Food-101 (96.2%). Our study reveals that model initialization plays a significant role, even for web-scale pretraining with billions of images._

![image tooltip here](/assets/MAWS.png){:width="500" style="display:block; margin-left:auto; margin-right:auto"}

**Released Models:**

We release the following models at various ViT sizes (Base, Large, Huge, and 2 Billion):
- <ins>Strong foundational MAWS (MAE→WSP) models</ins> with the following accuracy on ImageNet-1k: 89.7% finetuned, 88.1% linear probe, and 82.1% zero shot
- <ins>Improved Masked Autoencoder (MAE) models</ins> pretrained on 3 billion images, which outperform MAE models pretrained on ImageNet-1k