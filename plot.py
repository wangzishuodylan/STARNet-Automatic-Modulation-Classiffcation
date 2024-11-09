import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.regularizers import *
from keras.optimizers import adam_v2
from keras.models import model_from_json,Model
import mltools,dataset2016

import starnet as mcl
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from keras import backend as K
import tensorflow as tf
import pandas as pd

mltools.show_history(hist)