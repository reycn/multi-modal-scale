# Hateful Meme Detection through Context-Sensitive Prompting and Fine-Grained Labeling

This repository contains the codes and datasets used in our study.

<img src="misc/framework.png" width="350" />  

*Figure 1. The conceptual framework*

## Paper

**Publication:** [AAAI 2025](https://aaai.org/conference/aaai/aaai-25/) (Student Abstract, Oral)  

**Authors:** [Rongxin Ouyang](https://rongxin.me/cv)$^1$, [Kokil Jaidka](https://discovery.nus.edu.sg/17291-kokil-jaidka)$^1$$^2$, [Subhayan Mukerjee](https://discovery.nus.edu.sg/19113-subhayan-mukerjee)$^1$$^2$, and Guangyu Cui$^2$

  $^1$ Department of Communications and New Media, National University of Singapore  
  $^2$ Centre for Trusted Internet \& Community, National University of Singapore

**Link to Paper:**

- Main: [TBD]
- Supplementary Information: [TBD]

## Dataset

Due to the size and copyright restrictions of the original dataset, please use the provided links to access the dataset for our study.

- [Hateful Memes Challenge and Dataset](https://ai.meta.com/tools/hatefulmemes/)

- [Hugging Face](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded)

## File Structure

- `./dataset/`
  - `./dataset/raw/hateful_memes_expanded/` Meta Hateful Memes Meta Data
  - `./dataset/raw/hateful_memes_expanded/img/` Meta Hateful Memes Images
  - `...`
- `./process/`
  - `./process/internvl_finetuned/` Finetuned InternVL models
  - `...`
- `./script/`
  - `./script/1.finetune.distilbert.sample.ipynb` Finetuning DistilBERT (unimodal)
  - `./script/2.finetune.internvl.sample.sh` Finetuning Internvl 2.0 8B (multi-modal)
  - `./script/3.evaluation.batch.py` Evaluations of all models
  - `...`

## Bug Reports

- If you encountered any questions, feel free to reach out to Rongxin (rongxin$u.nus.edu). 😄

## Citation

TBD

## License

MIT License
