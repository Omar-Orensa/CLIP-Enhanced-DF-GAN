# CLIP-Enhanced DF-GAN

This repository is a modified version of [DF-GAN](https://github.com/tobran/DF-GAN), originally developed by Ming Tao et al., and licensed under the Apache 2.0 License.

This fork integrates **CLIP-based loss** into the training process of DF-GAN to improve semantic alignment between text and image.

<img src="framework.png" width="804px" height="380px"/>

---

## Modifications in This Fork

This repository introduces the following modifications to the original DF-GAN:

- Integration of **CLIP-based loss** into the generator’s objective function.
- Updated training loop to support CLIP loss.
- Documentation adjusted to reflect these changes.

These changes were implemented as part of a bachelor thesis project to improve semantic alignment in text-to-image generation.

---

## Requirements

- python 3.8
- Pytorch 1.9
- At least 1x12GB NVIDIA GPU

## Installation

Clone this repo.

```
git clone https://github.com/Omar-Orensa/CLIP-Enhanced-DF-GAN.git
pip install -r requirements.txt
cd CLIP-Enhanced-DF-GAN/code/
```

## Preparation

### Datasets

1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`

## Training

```
cd DF-GAN/code/
```

### Train the DF-GAN model

- For bird dataset: `bash scripts/train.sh ./cfg/bird.yml`
- For coco dataset: `bash scripts/train.sh ./cfg/coco.yml`

### Resume training process

If your training process is interrupted unexpectedly, set **resume_epoch** and **resume_model_path** in train.sh to resume training.

### TensorBoard

Our code supports automatic FID evaluation during training, the results are stored in TensorBoard files under ./logs. You can change the test interval by changing **test_interval** in the YAML file.

- For bird dataset: `tensorboard --logdir=./code/logs/bird/train --port 8166`
- For coco dataset: `tensorboard --logdir=./code/logs/coco/train --port 8177`

## Evaluation

### Download Pretrained Model

- [DF-GAN for bird](https://drive.google.com/file/d/1rzfcCvGwU8vLCrn5reWxmrAMms6WQGA6/view?usp=sharing). Download and save it to `./code/saved_models/bird/`
- [DF-GAN for coco](https://drive.google.com/file/d/1e_AwWxbClxipEnasfz_QrhmLlv2-Vpyq/view?usp=sharing). Download and save it to `./code/saved_models/coco/`

### Evaluate DF-GAN models

We synthesize about 3w images from the test descriptions and evaluate the FID between **synthesized images** and **test images** of each dataset.

```
cd CLIP-Enhanced-DF-GAN/code/
```

- For bird dataset: `bash scripts/calc_FID.sh ./cfg/bird.yml`
- For coco dataset: `bash scripts/calc_FID.sh ./cfg/coco.yml`
- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).

### Some tips

- Our evaluation codes do not save the synthesized images (about 3w images). If you want to save them, set **save_image: True** in the YAML file.
- Since we find that the IS can be overfitted heavily through Inception-V3 jointed training, we do not recommend the IS metric for text-to-image synthesis.

## 📈 Evaluation Summary

We evaluate our CLIP-enhanced DF-GAN model using both **FID** (Fréchet Inception Distance) and **CLIP Score** to assess image quality and semantic alignment. Experiments were conducted on the CUB dataset.

| Dataset | Metric       | Original DF-GAN | Best Score (CLIP-enhanced) | Epoch | λ<sub>CLIP</sub> |
| ------- | ------------ | --------------- | -------------------------- | ----- | ---------------- |
| **CUB** | FID ↓        | 12.23           | **12.71**                  | 1355  | 0.01             |
|         | CLIP Score ↑ | 28.54           | **29.11**                  | 1350  | 0.01             |

> **Notes:**
>
> - Lower FID indicates better visual fidelity.
> - Higher CLIP score indicates stronger semantic alignment between text and image.
> - The CLIP Score is reported on a **logit scale** (i.e., raw cosine similarity logits from the CLIP model), which is why values typically fall in the 18–30 range.

### Original DF-GAN Benchmark Results

The released model achieves better performance than the CVPR paper version.

| Model                    | CUB-FID↓  | COCO-FID↓ | NOP↓    |
| ------------------------ | --------- | --------- | ------- |
| DF-GAN(paper)            | 14.81     | 19.32     | 19M     |
| DF-GAN(pretrained model) | **12.10** | **15.41** | **18M** |

## Sampling

```
cd CLIP-Enhanced-DF-GAN/code/
```

### Synthesize images from example captions

- For bird dataset: `bash scripts/sample.sh ./cfg/bird.yml`
- For coco dataset: `bash scripts/sample.sh ./cfg/coco.yml`

### Synthesize images from your text descriptions

- Replace your text descriptions into the ./code/example_captions/dataset_name.txt
- For bird dataset: `bash scripts/sample.sh ./cfg/bird.yml`
- For coco dataset: `bash scripts/sample.sh ./cfg/coco.yml`

The synthesized images are saved at ./code/samples.

<img src="selected_samples.jpg" width="804px" height="306px"/>

---

### Citing DF-GAN and CLIP

```
@inproceedings{tao2022df,
  title={DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis},
  author={Tao, Ming and Tang, Hao and Wu, Fei and Jing, Xiao-Yuan and Bao, Bing-Kun and Xu, Changsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16515--16525},
  year={2022}
}
```

```
@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={ICML},
  year={2021}
}
```

The code is released for academic research use only. For commercial use, please contact [Ming Tao](mingtao2000@126.com).

**Reference**

- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) [[code]](https://github.com/hanzhanggit/StackGAN-v2)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) [[code]](https://github.com/taoxugit/AttnGAN)
- [DM-GAN: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1904.01310) [[code]](https://github.com/MinfengZhu/DM-GAN)
