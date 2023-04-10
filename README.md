# IEF-DGN

This is the official PyTorch implementation for the [paper](https://arxiv.org/abs/2204.11067):
> Zhihao Han. Interest-Evolution-Framework-with-Delta-Generative-Network-for-Session-based-Recommendation.

## Overview

In this work, we described IEF-DGN -- a candidate generation system that was designed as part of our ads multistage ranking system. Firstly, we proposed Interest-Evolution Framework with Delta Generative Network(IEF-DGN). Next, we design two intention generators for producing virtual items utilising multiple concepts. Finally, taking into account of item embeddings in interaction sequences, we propose the interest evolving matrix to capture interest evolving process.

<div  align="center"> 
<img src='asset/abs.png' width="50%">
</div>

## Requirements

```
recbole==1.0.1
python==3.7
pytorch==1.7.1
cudatoolkit==10.1
```

## Datasets

you can download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1dlJ3PzcT5SCN8-Mocr_AIQPGk9DVgTWB?usp=sharing). Then,
```bash
mv DATASET.zip dataset
unzip DATASET.zip
```

`DATASET` can be one of
* `diginetica`
* `nowplaying`
* `retailrocket`
* `tmall`
* `yoochoose`


## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole) and [RecBole-GNN](https://github.com/RUCAIBox/RecBole-GNN).

