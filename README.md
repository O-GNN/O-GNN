# $\mathcal{O}$-GNN: Incorporating Ring Priors into Molecular Modeling

This repository contains the code for [$\mathcal{O}$-GNN: Incorporating Ring Priors into Molecular Modeling](https://openreview.net/forum?id=5cFfz6yMVPU), which is introduced in ICLR2023.

If you find this work helpful in your research, please cite as:
```
@inproceedings{
zhu2023mathcalognn,
title={\${\textbackslash}mathcal\{O\}\$-{GNN}: incorporating ring priors into molecular modeling},
author={Jinhua Zhu and Kehan Wu and Bohan Wang and Yingce Xia and Shufang Xie and Qi Meng and Lijun Wu and Tao Qin and Wengang Zhou and Houqiang Li and Tie-Yan Liu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=5cFfz6yMVPU}
}
```
## Requirements and Installation
- PyTorch
- Torch-Geometric
- RDKit 

You can build a Docker image with our [Dockerfile](Dockerfile), and install our code and develop it locally
```
pip install -e . 
```

## Training 
We provided the example training command on PCQM4Mv1 dataset
```
bash run_training.sh --prefix iclr \
    --batch-size 512 --dropout 0.0 --pooler-dropout 0.0 \
    --init-face --use-bn --epochs 300 --num-layers 12 --lr 0.0003 \
    --weight-decay 0.1 --beta2 0.999 --num-workers 1 \
    --mlp-hidden-size 512 --lr-warmup \
    --use-adamw --node-attn --period 25
```