# MLP-OpenCL

## Introduction
Goal for this project is to get to know Multiprocessor systems, OpenMP and OpenCL, while comparing performance of a CPU and GPU, as well as comparing OpenCL and Tensorflow.
Multilayer Perceptron implemented in bare C, OpenMP and OpenCL.
Three fully connected layers are used. Problem is classification of a handwritten numbers, MLP is trained with Tensorflow's MNIST dataset.
Only forward propagation is implemented for now.

[***Wiki***](https://github.com/miloshunter/MLP-OpenCL/wiki) will contain theoretical knowledge and methodology, so feel free to check it out.

## Requirements
 * Python3 depencencies are noted in [python_training/requirements.txt](https://github.com/miloshunter/MLP-OpenCL/blob/master/python_training/requirements.txt) file

 `
 pip3 install -r python_training/requirements.txt
 `

 * Requires [libpng](http://www.libpng.org/pub/png/) to be installed 
 
 `
 Ubuntu: apt install libpng-dev
 `

 * OpenMP and OpenCL should be installed on system
 
 To check if OpenCL drivers and devices exist, one can use: **clinfo** package

## Training, compiling and execution
To test everything with default setup just run command:

```C
make train default_test
```

This command will train default network, compile code, download test image and run Single Core test, MultiCore OpenMP test and OpenCL test, if all dependencies are satisfied.

If not using OpenCL, just testing single core vs multi core CPU performance run following command:

```C
make train 
make run_single_core         #  For single thread
make run_openmp N_THR=4      # For 4 thread test
```

Checkout file tree structure below. Everything is done by using **make** commands.

Configuration of a network is done by using **\<network-name\>.conf** file (template default.conf is provided). One should not change the form of a file, just add/remove layers.

When particular network is used, please provide every **make** command with **CONF=\<network-name\>.conf** variable.

```C
4       -   number of layers
Size    | Type
784     - input
256      - fully connected
128      - fully connected
64      - fully connected
10     - output (softmax)
```
### Train network
Network is trained on MNIST dataset, which is automatically downloaded when needed.

In order to train network use **make train** command. For example to train network for *5* epochs and if network is defined in mlp.conf file, one should use:

```make train CONF=mlp.conf EPOCH=5```

If no CONF is provided, default.conf is used and trained for 2 epochs.

### Build code
To build C code, just run **make** in root directory.

### Run programs
Before running program, please make sure that the network is trained (by checking **parameters/** dir for **\<network-name\>_weights.bin** file.

There are few options on what to run:
```C
make run_single_core 
make run_openmp
make run_opencl
```

To choose on which **image** (28x28 .png) network will be used, one can set variable IMG=<path_to_image>/image.png.

To download 10 test images run: ```make download_test_pics```

For example to use network default.conf on image ~/images/example.png using single core and multicore (6 threads OpenMP):

```C make run_single_core run_openmp CONF=default.conf IMG=~/images/example.png N_THR=6```
### Cleaning up
To clean compiled files run **make clean** command.

To clean everything except original source files run **make clean_all** command. This will wipe out all parameters, test files, downloaded MNIST dataset etc.

## File tree
```C
|-- c_implementation
|   |-- load_parameters.c
|   |-- load_parameters.h
|   |-- makefile
|   |-- opencl_implementation
|   |   |-- makefile
|   |   |-- new_kernel.cl
|   |   |-- opencl_mlp.c
|   |   |-- opencl_utils.c
|   |   |-- opencl_utils.h
|   |-- read_image.c
|   |-- read_image.h
|   `-- simple_mlp.c
|-- default.conf
|-- LICENSE
|-- makefile
|-- python_training
|   |-- makefile
|   |-- mnist_train.py
|   `-- requirements.txt
`-- README.md
```



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
