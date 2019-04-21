# MLP-OpenCL

## Introduction
Goal for this project is to get to know Multiprocessor systems, OpenMP and OpenCL, while comparing performance of a CPU and GPU, as well as comparing OpenCL and Tensorflow.
Multilayer Perceptron implemented in bare C, OpenMP and OpenCL.
Three fully connected layers are used. Problem is classification of a handwritten numbers, MLP is trained with Tensorflow's MNIST dataset.
Only forward propagation is implemented for now.

***Wiki*** will contain theoretical knowledge and methodology, so feel free to check it out.

## Expectations
Hypothesis 1: MLP on CPU programmed with OpenMP will run faster than MLP programmed in bare C

Hypothesis 2: Every implementation on GPU will run faster than any implementation on CPU

Hypothesis 3: Size of a network (ie. number of layers and operations that are needed to be calculated) greatly impacts the speedup of GPU implementation. Larger networks should get more speedup than smaller when network is beeing run on GPU.

Hypothesis 4: Implementations in OpenCL and Tensorflow should be comparable. Tensorflow will be probably a bit faster, since this project is not about trying to optimize as much as possible.

## Methodology
1) The network is trained in Tensorflow.
2) The MLP algorithm is written in bare C, for CPU, without any parallelization.
3) OpenMP is used for accelerating the C implementation, ran on CPU.
4) OpenCL implementation, but ran on GPU.
5) Comparison of results
6) Conclusion
