"Adapted from the code (https://github.com/leena201818/radioml) contributed by leena201818"
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

import starnet as star
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from keras import backend as K
import tensorflow as tf
import pandas as pd

signal_len = 128
modulation_num = 11

data_format = 'channels_first'
def zero_mask(X_train,Y_train, p):
    y = Y_train
    num = int(X_train.shape[1] * p)
    res = X_train.copy()
    index = np.array([[i for i in range(X_train.shape[1])] for _ in range(X_train.shape[0])])
    for i in range(index.shape[0]):
        np.random.shuffle(index[i, :])

    for i in range(res.shape[0]):
        res[i, index[i, :num], :] = 0
    res = np.vstack((X_train,res))
    y_DA =y.transpose((1, 0))
    print(y_DA.shape)



    y_DA = np.tile(y_DA, (1, 2))
    print(y_DA.shape)
    y_DA = y_DA.T
    return res,y_DA
concat_axis = 3
def get_amp_phase(data):
    X_train_cmplx = data[:, 0, :] + 1j * data[:, 1, :]
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(data[:, 1, :], data[:, 0, :]) / np.pi
    X_train_amp = np.reshape(X_train_amp, (-1, 1, signal_len))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, signal_len))
    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))
    for i in range(X_train.shape[0]):
        X_train[i, :, 0] = X_train[i, :, 0] / np.linalg.norm(X_train[i, :, 0], 2)

    return X_train
if __name__ == "__main__":
    # Set up some params
    parser = argparse.ArgumentParser(description="STARNET")
    parser.add_argument("--epoch", type=int, default=10000, help='Max number of training epochs')
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size")
    parser.add_argument("--filepath", type=str, default='./weights.h5', help='Path for saving and reloading the weight')
    parser.add_argument("--datasetpath", type=str, default='./RML2016.10a_dict.pkl', help='Path for the dataset')
    parser.add_argument("--data", type=int, default=0, help='Select the RadioML2016.10a or RadioML2016.10b, 0 or 1')
    opt = parser.parse_args()

    # Set Keras data format as channels_last
    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    (mods,snrs,lbl),(X_train,Y_train,X_pure),(X_val,Y_val, X_valpure),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
        dataset2016.load_data(opt.datasetpath,opt.data)




    model = star.STARNET(input_shape=(128, 2), classes=11)


    lam = 0.6

    learning_rate = 0.003
    model.compile(loss=['mean_squared_error','categorical_crossentropy'],
                  loss_weights=[lam,1-lam],
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
    model.summary()
    filename = f'PET'
    checkpoint = ModelCheckpoint(opt.filepath,
                                 verbose=1,
                                 save_best_only=True)
    # print(y_train.shape)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.7, verbose=1, patience=15, min_lr=0.0000001)
    X_train_masked,y = zero_mask(X_train,Y_train , 0.1)


    x = np.vstack((X_pure,X_pure))

    hist = model.fit(x=X_train_masked ,
                     y=[x, y],
                     batch_size=600,
                     epochs=300,
                     verbose=2,
                     validation_data=(X_val, [X_val, Y_val]),
                     callbacks=[checkpoint, rl]
                     )
    train_test_list = [hist.history['softmax_loss'], hist.history['val_softmax_loss'], hist.history['softmax_accuracy'],
                       hist.history['val_softmax_accuracy']]
    train_test_array = np.array(train_test_list).T
    df = pd.DataFrame(train_test_array, columns=['Training Loss', 'Test Loss', 'Training Acc', 'Test Acc'])
    df.to_excel(f'loss/{filename}.xlsx', index=False)
    # We re-load the best weights once training is finished
    model.load_weights(opt.filepath)
    mltools.show_history(hist)

    # Show simple version of performance
    score = model.evaluate([X_test], [X_test,Y_test], verbose=1, batch_size=opt.batch_size)
    print(score)


