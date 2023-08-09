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


<h2> Datasets </h2>

We conduct experiments on five datasets across three modalities. For vision modality, we use a large-scale dataset [ImageNet](https://www.image-net.org/), two medium-sacle datasets: STL10 and ImageNet-100, and two small-scale datasets: CIFAR10 and CIFAR100. For language modality, we use 7 semantic textual similar- ity (STS) tasks. For graphs modality, we conduct graph-level classification experiments on 7 benchmark datasets: IMDB-B, IMDB- M, COLLAB, REDDIT-B, PROTEINS, MUTAG, and NCI1.

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
