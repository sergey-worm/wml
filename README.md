# Tiny ML Framework by Worm

![The picture](./imgs/worm_at_work.jpg "Worm at work!")

## Decription

This project containts a Tiny ML (WML) framework and an example of its use.
The project is written on plain C and developed to be used on baremetal
systems with FPU.

File structure of the project:

1. **wml** — contains header and source files of WML framework
1. **main.c** — the entry point of the project, containts unit tests and examples
1. **Makefile** — the root makefile of the project
1. **refs** — python files with reference code of training and testing
   neural network with pytorch

# WML

WML is a Tiny ML framework by Worm. It contains:<br>

1. **wml_mat** — representation of Matrix and implementation of operations with it
1. **wml_utils** — collection of utilities (stack memory allocator, random generator, etc)
1. **wml_layers** — representation of layers of neural network (Linear, ReLU, Softmax)
1. **wml_data_loader** — implementation of dataloaders (from array, from file)
1. **wml_plot** — a wrapper over `gnuplot`


WML allows to train and use neural networks (NN) on microcontrollers and CPUs.
For NN training the availability of FPU is mandatory.

## How to use

To build project on Unix system you need to follow the next steps:<br>

1. Edit `main()` function in `main.c` file and uncomment one of functions
   (unit\_tests, iris, mnist). Also you can change hyperparameters in the
   appropriate function.
1. Download MNIST dataset (for MNIST example only) by executing `./dl_mnist.sh`.
1. Execute `make` in root directory.
1. Start training and testing by executing `./bld/prog`.

As a result you will train and test the model and will see its accuracy.

**NOTE:**<br>
The `Makefile` by default does debug build and use GCC sanitizers. This allows
to make robust project but consumes CPU time. You can modify `Makefile` to
change this.

## Unit tests

To test parts of WML you can run unit\_tests:

1. Edit `main()` function in `main.c` file and uncomment `unit\_tests()` function.
1. Build the project by executing `make`.
1. Run the tests by executing the application `./bld/prog`.

As a result unit test will be completed.

## IRIS Example

The first example project is NN with one hidden layer and Fisher Iris dataset.
The NN classifies Irises by its dementions and reaches 100% accuracy.<p>

The dataset is embedded in the header file `wml_ds_iris.h`.

To reproduce Iris example follow the next steps:

1. Edit `main()` function in `main.c` file and uncomment `iris()` function.
1. Build the project by executing `make`.
1. Run the tests by executing the application `./bld/prog`.

## MNIST Example

The second example project is NN with two hidden layers and
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.
The NN classifies pictures of digits and reaches accuracy about 94%.<p>

The dataset should be loaded by script `dl_mnist.sh`.

To reproduce MNIST example follow the next steps:

1. Edit `main()` function in `main.c` file and uncomment `mnist()` function.
1. Download MNIST dataset by executing the script `./dl_mnist.sh`
1. Build the project by executing `make`.
1. Run the tests by executing the application `./bld/prog`.

**NOTE:**<br>
The training process can take about 2 hours.

## Contacts

Sergey Worm, <sergey.worm@gmail.com>
