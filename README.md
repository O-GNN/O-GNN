# $\mathcal{O}$-GNN: Incorporating Ring Priors into Molecular Modeling

This repository contains the code for *$\mathcal{O}$-GNN: Incorporating Ring Priors into Molecular Modeling*.


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