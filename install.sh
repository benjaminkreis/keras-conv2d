#!/bin/bash
conda create --copy --name keras-training2d python=2.7.13
conda install --name keras-training2d --file keras-training2d.conda 
source activate keras-training2d
pip install -r keras-training2d.pip
cp activateROOT.sh  $CONDA_PREFIX/etc/conda/activate.d/activateROOT.sh 
