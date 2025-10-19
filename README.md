# SCRAMBLe

This is the main repository containing code for the paper "SCRAMBLe : Enhancing Multimodal LLM Compositionality with Synthetic Preference Data". \[[arxiv](https://arxiv.org/abs/2504.04740)\] \[[poster](https://samarth4149.github.io/projects/scramble/poster.pdf)\].

### Install

```bash
conda create -n scramble python=3.10
conda activate scramble
pip install -r requirements.txt
```

### Chat interface

The 2 scripts `chat_llava.py` and `chat_molmo.py` can be used to download the SCRAMBLe tuned models from huggingface and chat with them.

### Data

The synthetic training data for SCRAMBLe is available at `./data/synthetic_data.json`. 
It uses images from the COCO-2017 train dataset. One convenient download location is [kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/data).

### Citation

If you found this work useful, please consider citing:

```bibtex
@inproceedings{mishra2025scramble,
  title        = {SCRAMBLe: Enhancing Multimodal LLM Compositionality with Synthetic Preference Data},
  author       = {Mishra, Samarth and Saenko, Kate and Saligrama, Venkatesh},
  booktitle    = {ICCV 2025 Findings},
  year         = {2025},
  note         = {arXiv preprint arXiv:2504.04740},
  url          = {https://arxiv.org/abs/2504.04740},
  doi          = {10.48550/arXiv.2504.04740}
}
```

