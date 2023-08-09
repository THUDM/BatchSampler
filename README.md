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
