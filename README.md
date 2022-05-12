# relu_swish
This repo is a supplementary material for [author's blog post (Japanese)](link). Through a few examples, we investigate the relationship between [Swish](http://arxiv.org/abs/1710.05941) and a linear activation, also Swish and [ReLU](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) activation. 

## Purpose
Swish (equivalent to [SiLU](https://arxiv.org/abs/1702.03118)) activation function can be considered as a generalization of a linear map and ReLU. In other words, e can have Swish approach to linear / ReLU by varying its characteristic factor, beta (beta -> 0 or beta -> inf). Considering its similarity to ReLU, we employed [He normal](https://arxiv.org/abs/1502.01852) for weight matrix initialization. 

## Example
Linear activation-employed DNN (Deep Neural Network) returns linear signal (which is very natural), and ReLU-equipped DNN yields zig-zag result. In addition, Swish inference approaches to linear DNN / ReLU DNN depending on beta values. 

<img src="">

## Dependencies
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
<br>
[2] Ramachandran, P., Zoph, B., Le, Q.V.: Swish: a Self-Gated Activation Function, *arXiv: 1710.05941*, 2017. ([paper](http://arxiv.org/abs/1710.05941))
<br>
[3] Nair, V., Hinton, G.E.: Rectified Linear Units Improve Restricted Boltzmann Machines, *International Conference on Machine Learning (ICML)*, pp. 807–814, 2010. [[paper](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)]
<br>
[4] Elfwing, S., Uchibe, E., Doya, K.: Sigmoid-weighted linear units for neural network function approximation in reinforcement learning, Vol. 107, pp. 3-11, *Neural Networks*, 2018. ([paper](https://arxiv.org/abs/1702.03118))
<br>
[5] He, K., Zhang, X., Ren, S., Sun, J.: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, *International Conference on Computer Vision (ICCV)*, pp. 1026-1034, 2015. ([paper](https://arxiv.org/abs/1502.01852))
