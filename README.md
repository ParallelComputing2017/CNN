# ConvNet Parallelization

This work explores the parallelization of Convolutional Neural Networks (ConvNets), on CPU and GPU. The experiment parallelize a ConvNet using Pthreads, OpenMP and CUDA, and uses the MNIST dataset to validated the results.

## Requirements

* A copy of the MNIST dataset (check "data" folder)
* Boost library (system and timer)
* CUDA toolkit 8


## Description

The experiment consist in take a sequential version of a ConvNet and parallelize it using different strategies and technologies. The sequential version used is a object oriented variation of [Simple Convolutional Neural Network Library](https://github.com/can1357/simple_cnn). For the parallelization 2 strategies were applied: data parallelism on CPU and matrix multiplication parallelism on GPU. The experiment is validated using the [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist) and the simple topology
