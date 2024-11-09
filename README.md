# 通过上下文敏感提示和细粒度标签检测仇恨备忘录 [[English](./README.md) | 中文]

该资源库包含我们研究中使用的代码和数据集。

<img src="misc/framework.png" width="350" />  

*图 1. 概念框架*

## 论文

**发表:** [AAAI 2025](https://aaai.org/conference/aaai/aaai-25/) (学生摘要，受邀口头汇报)  

**作者:** [Rongxin Ouyang (欧阳荣鑫)](https://rongxin.me/cv)$^1$, [Kokil Jaidka](https://discovery.nus.edu.sg/17291-kokil-jaidka)$^1$$^2$, [Subhayan Mukerjee](https://discovery.nus.edu.sg/19113-subhayan-mukerjee)$^1$$^2$, and Guangyu Cui$^2$

  $^1$ 新加坡国立大学传播与新媒体系  
  $^2$ 新加坡国立大学可信互联网和社区中心

**论文链接:**

- 正文: [TBD]
- 补充材料: [TBD]

## 数据集

由于原始数据集的大小和版权限制，请使用提供的链接访问我们研究的数据集。

- [仇恨备忘录挑战和数据集](https://ai.meta.com/tools/hatefulmemes/)

- [Hugging Face 备份](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded)

## 文件结构

- `./dataset/`
  - `./dataset/raw/hateful_memes_expanded/` Meta 仇恨备忘录 元数据
  - `./dataset/raw/hateful_memes_expanded/img/` Meta 仇恨备忘录图片文件
  - `...`
- `./process/`
  - `./process/internvl_finetuned/` 微调后 InternVL 模型
  - `...`
- `./script/`
  - `./script/1.finetune.distilbert.sample.ipynb` 微调 DistilBERT (单模态)
  - `./script/2.finetune.internvl.sample.sh` 微调 Internvl 2.0 8B (多模态)
  - `./script/3.evaluation.batch.py` 评估所有模型
  - `...`

## 汇报问题

- 如果您遇到任何问题，请随时联系荣鑫 (rongxin$u.nus.edu)。 😄

## 引用格式

TBD

## 开源协议

MIT License
