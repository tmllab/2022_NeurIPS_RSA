# RSA



## Experiments

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Please download and place all datasets into the data directory. Note that our train program will invoke the linear evaluation by default.


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


## Cite RSA
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
