# Codes and Dataset
This repository contains the code relevant to our submission.

## File Structure
- `./dataset/`
  - `./dataset/raw/hateful_memes_expanded/` Meta Hateful Memes
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

## Dataset
- [Hateful Memes Challenge and Dataset](https://ai.meta.com/tools/hatefulmemes/)

## Bugs
- Please report bugs to: rongxin@u.nus.edu, thanks!
## License
MIT License