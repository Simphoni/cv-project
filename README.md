## 主要功能

本仓库包含微调模型的训练代码、模型的推理与评测代码。

## 功能介绍

* `training` 文件夹中主要包含微调模型的代码
* `inference` 文件夹中主要包含模型推理生成图像的代码
* `eval` 文件夹中主要包含多种评测指标的代码

## 参考代码

我们的代码参考了开源仓库的若干实现，在此记录并表示感谢：

Train:
 * Diffusers: https://github.com/huggingface/diffusers
 * PEFT: https://github.com/huggingface/peft

Eval: 
 * FID: https://github.com/mseitzer/pytorch-fid
 * Inception Score: https://github.com/sbarratt/inception-score-pytorch
 * Perceptual Similarity: https://github.com/richzhang/PerceptualSimilarity
 * CLIP Score: https://github.com/Taited/clip-score

## 致谢

衷心感谢老师和助教的辛勤付出！在课程和实验的过程中我收获良多。
