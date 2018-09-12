## Pervasive Attention: 2D Convolutional Networks for Sequence-to-Sequence Prediction


This is an open source PyTorch implementation of the pervasive attention model described in:

Maha Elbayad, Laurent Besacier, and Jakob Verbeek. 2018. **[Pervasive Attention: 2D Convolutional Networks for Sequence-to-Sequence Prediction](https://arxiv.org/abs/1808.03867)**. In Proceedings of the 22nd Conference on Computational Natural Language Learning (CoNLL 2018)


### Requirements
```
pytorch (tested with v0.4.1)
subword-nmt
h5py (2.7.0)
tensorboardX 
```

### Usage:

#### IWSLT'14 pre-processing:
```
cd data
./prepare-iwslt14.sh
cd ..
python preprocess.py -d iwslt14
```

#### Training:
```
mkdir -p save events
python train.py -c config/l24.yaml
```

#### Generation & evaluation
```
python generate.py -c config/l24.yaml

```



