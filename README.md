<p>
  <img src="img/fig.png" width="1000">
  <br />
</p>

<hr>

<h1> BatchSampler: Sampling Mini-Batches for Contrastive Learning in Vision, Language, and Graphs </h1>



Source code for KDD'23 paper:  [BatchSampler: Sampling Mini-Batches for Contrastive Learning in Vision, Language, and Graphs](https://arxiv.org/abs/2306.03355).


BatchSampler is a simple and general generative method to sample mini-batches of hard-to-distinguish (i.e., hard and true negatives to each other) instances, which can be directly plugged into in-batch contrastive models in vision, language, and graphs. 

<h2>Dependencies </h2>

* Python >= 3.7
* [Pytorch](https://pytorch.org/) >= 1.9.0 

<h2>Quick Start </h2>


```bash
sh train.sh
```



<h2> Datasets </h2>

We conduct experiments on five datasets across three modalities. For vision modality, we use a large-scale dataset [ImageNet](https://www.image-net.org/), two medium-sacle datasets: [STL10](https://cs.stanford.edu/~acoates/stl10/) and [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100), and two small-scale datasets: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html). For language modality, we use 7 semantic textual similarity (STS) tasks. For graphs modality, we conduct graph-level classification experiments on 7 benchmark datasets: IMDB-B, IMDB- M, COLLAB, REDDIT-B, PROTEINS, MUTAG, and NCI1.

<h2> Experimental Results </h2>
Vision Modality

|        Method      | 100ep        | 400ep        | 800ep        |
| ------------------ | ------------ | ------------ | ------------ | 
| SimCLR             | 64.0         | 68.1         | 68.7         | 
| **w/BatchSampler** | **64.7**     | **68.6**     | **69.2**     | 
| Moco v3            | 68.9         | 73.3         | 73.8         | 
| **w/BatchSampler** | **69.5**     | **73.7**     | **74.2**     | 

language Modality

|          Method    | STS12        | STS13        | STS14        | STS15          | STS16          | STS-B          |   SICK-R       |   Avg.           |
| ------------------ | ------------ | ------------ | ------------ | -------------- | -------------- | -------------- | -------------- | -------------- | 
| SimCSE-BERT<sub>{BASE} |   68.62   | 80.89     | 73.74     | 80.88     | 77.66     | **77.79**      | **69.64** | 75.60 | 
| w/kNN Sampler      | 63.62     | 74.86    | 69.79    | 79.17 | 76.24           | 74.73            | 67.74 | 72.31  |
| **w/BatchSampler** | **72.37**     | **82.08**    |  **75.24**    | **83.10**     | **78.43**     |   77.54  | 68.05  | **76.69** |
| DCL-BERT<sub>{BASE} |   65.22   | 77.89    | 68.94   | 79.88     | **76.72**    |73.89      | **69.54** | 73.15 | 
| w/kNN Sampler      | 66.34    | 76.66    | 72.60    | 78.30 | 74.86         | 73.65            | 67.92 | 72.90  |
| **w/BatchSampler** | **69.55**     | **82.66**    |  **73.37**    | **80.40**     | 75.37    |   **75.43**  | 66.76  | **74.79** |
| HCL-BERT<sub>{BASE} |   62.57   | 79.12     | 69.70     | 78.00     | 75.11    | 73.38     | 69.74 | 72.52| 
| w/kNN Sampler      | 61.12    | 75.73    | 68.43    | 76.64 | 74.78          | 71.22            | 68.04 | 70.85  |
| **w/BatchSampler** | **66.87**     | **81.38**    |  **72.96**    | **80.11**     | **77.99**     |   **75.95**  | 70.89 | **75.16** |



Graphs Modality

|          Method    | IMDB-B        | IMDB-M        | COLLAB        | REDDIT-B          | PROTEINS          | MUTAG         |   NCI1       | 
| ------------------ | ------------ | ------------ | ------------ | -------------- | -------------- | -------------- | -------------- | 
| GraphCL |   70.90±0.53    | 48.48±0.38    | 70.62±0.23      | 90.54±0.25    | 74.39±0.45     | 86.80±1.34      | 77.87±0.41  | 
| w/kNN Sampler      | 70.72±0.35      | 47.97±0.97     | 70.59±0.14     | 90.21±0.74  | 74.17±0.41           | 86.46±0.82          | 77.27±0.37  | 
| **w/BatchSampler** | **71.90±0.46**      | **48.93±0.28**     | **71.48±0.28**    | **90.88±0.16**  | **75.04±0.67**           | **87.78±0.93**          | **78.93±0.38**  | 
| DCL|    71.07±0.36    | 48.93±0.32    | **71.06±0.51**      | 90.66±0.29    | 74.64±0.48     | 88.09±0.93      | 78.49±0.48  | 
| w/kNN Sampler   |   70.94±0.19    | 48.47±0.35    | 70.49±0.37      | 90.26±1.03    | 74.28±0.17     | 87.13±1.40      | 78.13±0.52  | 
| **w/BatchSampler** |  **71.32±0.17**      | **48.96±0.25**     | 70.44±0.35    | **90.73±0.34**  | **75.02±0.61**           | **89.47±1.43**          | **79.03±0.32**  | 
| HCL|    **71.24±0.36**    | 48.54±0.51    | 71.03±0.45      | 90.40±0.42    | 74.69±0.42     | 87.79±1.10      | 78.83±0.67  | 
| w/kNN Sampler    |  71.14±0.44    | 48.36±0.93   | 70.86±0.74      | 90.64±0.51    | 74.06±0.44     | 87.53±1.37      | 78.66±0.48  | 
| **w/BatchSampler** |  71.20±0.38      | **48.76±0.39**     | **71.70±0.35**    | **91.25±0.25**  | **75.11±0.63**           | **88.31±1.29**          | **79.17±0.27**  | 
| MVGRL|    74.20±0.70    | 51.20±0.50    |-     | 84.50±0.60    | -    | 89.70±1.10      | - | 
| w/kNN Sampler   |   73.30±0.34    | 50.70±0.36    | -      | 82.70±0.67    | -    | 85.08±0.66      | - | 
| **w/BatchSampler** |  **76.70±0.35**      | **52.40±0.39**     | - | **87.47±0.79**  | -          | **91.13±0.81**          | -  | 

<h2> Citing </h2>
If you find our work is helpful to your research, please consider citing our paper:

```
@article{yang2023batchsampler,
  title={BatchSampler: Sampling Mini-Batches for Contrastive Learning in Vision, Language, and Graphs},
  author={Yang, Zhen and Huang, Tinglin and Ding, Ming and Dong, Yuxiao and Ying, Rex and Cen, Yukuo and Geng, Yangliao and Tang, Jie},
  journal={arXiv preprint arXiv:2306.03355},
  year={2023}
}
```
