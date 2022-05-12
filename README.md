# relu_swish
This repo is a supplementary material for [author's blog post (Japanese)](). Through a few examples, we investigate the relationship between [Swish](http://arxiv.org/abs/1710.05941) and a linear activation, also Swish and [ReLU](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) activation. 

## Purpose
Swish (or [SiLU](https://arxiv.org/abs/1702.03118)) activation function can be considered as a generalization of a linear map and ReLU. 


Activation functions are essential to introduce non-linearity to DNNs i.e. DNN approximations are heavily dependent on the properties of the selected activation functions. This repo builds neural networks to learn several functions with 3 different activation functions, namely, ReLU, Swish, and tanh. Networks have different parameter initializations, [Glorot normal](https://proceedings.mlr.press/v9/glorot10a.html) for tanh activation, [He normal](https://arxiv.org/abs/1502.01852) for ReLU ([Nair (2010)](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)) and Swish ([Ramachandran (2017)](http://arxiv.org/abs/1710.05941), [Elfwing (2018)](https://arxiv.org/abs/1702.03118)). 

## Example
ReLU network has zig-zag inference result, while others (tanh & Swish) have smooth approximations. This is due to their natures, or continuity to be exact. 


## Dependencies
TensorFlow environment:
|Library/Package|Version|
|:---:|:---:|
|keras|2.8.0|
|matplotlib|3.5.1|
|numpy|1.22.1|
|pandas|1.4.0|
|scipy|1.7.3|
|tensorflow|2.8.0|

## References
[1] [author's blog post](link). 
[2] Ramachandran, P., Zoph, B., Le, Q.V.: Swish: a Self-Gated Activation Function, *arXiv: 1710.05941*, 2017. ([paper](http://arxiv.org/abs/1710.05941))
[3] Nair, V., Hinton, G.E.: Rectified Linear Units Improve Restricted Boltzmann Machines, *International Conference on Machine Learning (ICML)*, pp. 807–814, 2010. [[paper](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)]
[3] Elfwing, S., Uchibe, E., Doya, K.: Sigmoid-weighted linear units for neural network function approximation in reinforcement learning, Vol. 107, pp. 3-11, *Neural Networks*, 2018. ([paper](https://arxiv.org/abs/1702.03118))

[7] Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, *arXiv: 1810.04805*, 2018. ([paper](https://arxiv.org/abs/1810.04805))
[8] He, K., Zhang, X., Ren, S., Sun, J.: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, *International Conference on Computer Vision (ICCV)*, pp. 1026-1034, 2015. ([paper](https://arxiv.org/abs/1502.01852))
[9] 以前の記事: [DNNフィッティングと活性化関数の選択](https://qiita.com/ShotaDeguchi/items/751a8ec86b7bc7ec34ed)
