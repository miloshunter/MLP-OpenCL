# MLP-OpenCL

## Introduction
Goal for this project is to get to know Multiprocessor systems, OpenMP and OpenCL, while comparing performance of a CPU and GPU, as well as comparing OpenCL and Tensorflow.
Multilayer Perceptron implemented in bare C, OpenMP and OpenCL.
Three fully connected layers are used. Problem is classification of a handwritten numbers, MLP is trained with Tensorflow's MNIST dataset.
Only forward propagation is implemented for now.

[***Wiki***](https://github.com/miloshunter/MLP-OpenCL/wiki) will contain theoretical knowledge and methodology, so feel free to check it out.

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

## File tree at the end of the project
```C
├── c_implementation
│   ├── load_parameters.c
│   ├── load_parameters.h
│   ├── makefile
│   ├── opencl_implementation
│   │   ├── makefile
│   │   ├── new_kernel.cl
│   │   ├── opencl_mlp.c
│   │   ├── opencl_utils.c
│   │   ├── opencl_utils.h
│   │   └── test_kernel.c
│   ├── read_image.c
│   ├── read_image.h
│   └── simple_mlp.c
├── default.conf
├── LICENSE
├── makefile
├── parameters
├── python_training
│   ├── makefile
│   ├── MNIST_data
│   │   ├── t10k-images-idx3-ubyte.gz
│   │   ├── t10k-labels-idx1-ubyte.gz
│   │   ├── train-images-idx3-ubyte.gz
│   │   └── train-labels-idx1-ubyte.gz
│   ├── mnist_train.py
│   └── requirements.txt
├── README.md
```
## Results
### Hypothesis 1:
_MLP on CPU programmed with OpenMP will run faster than MLP programmed in bare C_

Speedup for OpenMP on CPU depends on the network size but generaly goes from 1.5 on two threads and small networks to around 5 on maximum thread number (Ryzen 5 2600x has 6 cores and 12 threads).
This hypothesis is proved to be right.

### Hypothesis 2:
_Every implementation on GPU will run faster than any implementation on CPU_

This hypothesis proved right again, but greatly depends on network size.

For small network where CPU speedup is 2.5, GPU speedup is 2.7. For smaller networks probably would be even worse.

But for large network where CPU speedup is 5, GPU speedup is 17.

### Hypothesis 3:
_Size of a network greatly impacts the speedup of GPU implementation_

Size of the network indeed impacts the GPU implementation speedup. It goes from 2.7 on small network to 17 on large network.

### Hypothesis 4:
_Implementations in OpenCL and Tensorflow should be comparable_

Need to check Tensorflow time so this stays unanswered for now. :(
