import os
from optparse import OptionParser
from keras.models import load_model, Model
from keras.models import model_from_json
from argparse import ArgumentParser
from keras import backend as K
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import yaml
import math


json_string = open('model.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('my_model_weights.h5')

x_image_test = np.empty(shape=(1,8,8,1))
x_image_test.fill(0)

print(x_image_test.shape)
print(x_image_test)
print(model.predict(x_image_test))
