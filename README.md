# keras-conv2d

## Description

Code for training and evaluating 2D convolutional neural network with Keras.  Use multiple channels and filters to explore conv1d options for [HLS4ML project](https://github.com/hls-fpga-machine-learning/keras-training).

This code just uses the Keras MNIST handwritten digits example.  inference.py implements the inference by hand as a test before writing code for FPGAs. 

## Setup

Install all dependencies using miniconda (from [HLS4ML project](https://github.com/hls-fpga-machine-learning/keras-training))):

Install `miniconda2` by sourcing `install_miniconda.sh` in your home directory. Log out and log back in after this.
```bash
cp install_miniconda.sh ~/
cd ~
source install_miniconda.sh
```

Install the rest of the dependencies:
```bash
cd ~/keras-conv2d
source install.sh
```

Each time you log in set things up:
```bash
source setup.sh
```

## Run
```
python mnist_cnn.py
python inference.py
```
