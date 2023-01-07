<div align="center">   
  
# RSA: Reducing Semantic Shift from Aggressive Augmentations for Self-supervised Learning
[![Paper](https://img.shields.io/badge/paper-NeurIPS-green)](https://openreview.net/pdf?id=Cgmk9CicWFl)

</div>

Official implementation of [RSA: Reducing Semantic Shift from Aggressive Augmentations for Self-supervised Learning](https://openreview.net/pdf?id=Cgmk9CicWFl) (NeurIPS 2022).

Most recent self-supervised learning methods learn visual representation by contrasting different augmented views of images. Compared with supervised learning, more aggressive augmentations have been introduced to further improve the diversity of training pairs. However, aggressive augmentations may distort images' structures leading to a severe semantic shift problem that augmented views of the same image may not share the same semantics, thus degrading the transfer performance. To address this problem, we propose a new SSL paradigm, which counteracts the impact of semantic shift by balancing the role of weak and aggressively augmented pairs. Specifically, semantically inconsistent pairs are of minority, and we treat them as noisy pairs. Note that deep neural networks (DNNs) have a crucial memorization effect that DNNs tend to first memorize clean (majority) examples before overfitting to noisy (minority) examples. Therefore, we set a relatively large weight for aggressively augmented data pairs at the early learning stage. With the training going on, the model begins to overfit noisy pairs. Accordingly, we gradually reduce the weights of aggressively augmented pairs. In doing so, our method can better embrace aggressive augmentations and neutralize the semantic shift problem. Experiments show that our model achieves 73.1% top-1 accuracy on ImageNet-1K with ResNet-50 for 200 epochs, which is a 2.5% improvement over BYOL. Moreover, experiments also demonstrate that the learned representations can transfer well for various downstream tasks.


<p float="left" align="center">
<img src="arch.png" width="800" /> 
<figcaption align="center">
The illustration of our proposed method (RSA). We utilize an asymmetric-style framework, including an online network and a target network. The online network is optimized by gradients, and the target network is updated with the exponential moving average strategy. We first adopt the weak augmentation to generate two views, then adopt the aggressive augmentations to further generate another two views. Subsequently, we make aggressive-augmented views to keep consistent with their corresponding weak- and aggressive-augmented views in the embedding space. On the right of the image, we compare RSA with classical SSL methods. RSA forces learned representations to a balance between weak- and aggressive-augmented views.
</figcaption>
</p>


## Requirements
- This codebase is written for `python3` and 'pytorch'.
- To install necessary python packages, run `pip install -r requirements.txt`.


## Experiments
### Data
- Please download and place all datasets into the data directory. 


### Training
To train RSA on CIFAR-100

```
python train_single.py --dataset cifar100 --beta 0.3
```

To train RSA on STL-10

```
python train_single.py --dataset stl10
```


To train RSA on ImageNet-100

```
python train_multi.py --dataset ImageNet-100 --data_root data/ImageNet-100/
```


To train RSA on ImageNet-1K

```
python train_multi.py --dataset ImageNet --lr 0.6 --wd 1e-6 --batch-size 2048 --warmup-epochs 10 --data_root data/ImageNet/
```


## License and Contributing
- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github. 


## Reference
If you find the code useful in your research, please consider citing our paper:


<pre>
@inproceedings{
    bai2022rsa,
    title={RSA: Reducing Semantic Shift from Aggressive Augmentations for Self-supervised Learning},
    author={Yingbin Bai and Erkun Yang and Zhaoqing Wang and Yuxuan Du and Bo Han and Cheng Deng and Dadong Wang and Tongliang Liu},
    booktitle={NeurIPS},
    year={2022},
}
</pre>
