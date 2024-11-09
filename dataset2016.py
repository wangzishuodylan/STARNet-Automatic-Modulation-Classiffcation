"Adapted from the code (https://github.com/leena201818/radioml) contributed by leena201818"
import pickle
import numpy as np
from keras.utils import to_categorical
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
# import mltools,dataset2016
# import MCLDNN as mcl
import argparse
def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)
signal_len =128

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

def rotate_matrix(theta):
    m = np.zeros((2,2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    print(m)
    return m

def Rotate_DA(x, y):
    [N, L, C] = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3*np.pi/2))


    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))
    print(y.shape)
    y_DA =y.transpose((1, 0))
    print(y_DA.shape)



    y_DA = np.tile(y_DA, (1, 4))
    print(y_DA.shape)
    y_DA = y_DA.T
    print(y_DA.shape)
    print(x_DA.shape)
    # y_DA = y_DA.reshape(-1)
    # print(y_DA.shape)
    # y_DA = y_DA.T
    # print(y_DA.shape)
    return x_DA, y_DA


def load_data(filename,data):
    # RadioML2016.10a: (220000,2,128), mods*snr*1000, total 220000 samples;
    # RadioML2016.10b: (1200000,2,128), mods*snr*6000, total 1200000 samples;
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ]
    X = []
    lbl = []
    train_idx=[]
    val_idx=[]
    np.random.seed(2016)
    a=0
    train_rate = 0.5
    X_label = []
    for mo in mods:
        # print(mo)
        for sn in snrs:
            if sn>=18:
                for c in range(20):
                    X_label.append(Xd[(mo,sn)])

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            if data==0:
                train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
                val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
            elif data==1:
                train_idx+=list(np.random.choice(range(a*6000,(a+1)*6000), size=3600, replace=False))
                val_idx+=list(np.random.choice(list(set(range(a*6000,(a+1)*6000))-set(train_idx)), size=1200, replace=False))
            a+=1
    X = np.vstack(X)
    X_label = np.vstack(X_label)

    # Scramble the order between samples
    # and get the serial number of training, validation, and test sets
    n_examples = X.shape[0]
    test_idx=list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    X_train =X[train_idx]
    X_pure =X_label[train_idx]
    X_val=X[val_idx]
    X_valpure =X_label[val_idx]
    X_test =X[test_idx]
    # print(X_train.shape)


    # transfor the label form to one-hot
    def to_onehot(yy):
        yy1=np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy]=1
        return yy1
    Y_train=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),train_idx)))

    Y_val=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),val_idx)))
    Y_test=to_onehot(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    X_train = X_train.transpose((0, 2, 1))
    X_pure = X_pure.transpose((0, 2, 1))



    X_train, Y_train = Rotate_DA(X_train, Y_train)
    X_pure, Y_pure = Rotate_DA(X_pure, Y_train)

    X_train = X_train.transpose((0, 2, 1))
    X_pure = X_pure.transpose((0, 2, 1))


    X_train = get_amp_phase(X_train)
    X_pure = get_amp_phase(X_pure)




    X_val = get_amp_phase(X_val)
    X_valpure = get_amp_phase(X_valpure)
    X_test = get_amp_phase(X_test)
    # X_val = X_val.transpose((0, 2, 1))
    # X_test = X_test.transpose((0, 2, 1))
    print(X_train.shape)
    print(X_pure.shape)
    # print(X_val.shape)
    # print(X_test.shape)


    return (mods,snrs,lbl),(X_train,Y_train, X_pure),(X_val,Y_val, X_valpure),(X_test,Y_test),(train_idx,val_idx,test_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MCLDNN")
    parser.add_argument("--epoch", type=int, default=10000, help='Max number of training epochs')
    parser.add_argument("--batch_size", type=int, default=400, help="Training batch size")
    parser.add_argument("--filepath", type=str, default='./weights.h5', help='Path for saving and reloading the weight')
    parser.add_argument("--datasetpath", type=str, default='./RML2016.10a_dict.pkl', help='Path for the dataset')
    parser.add_argument("--data", type=int, default=0, help='Select the RadioML2016.10a or RadioML2016.10b, 0 or 1')
    opt = parser.parse_args()

    # Set Keras data format as channels_last
    K.set_image_data_format('channels_last')
    print(K.image_data_format())
    #
    (mods, snrs, lbl), (X_train, Y_train, X_pure), (X_val, Y_val, X_valpure), (X_test, Y_test), (train_idx, val_idx, test_idx) = load_data(
        opt.datasetpath, opt.data)
    # X1_train = np.expand_dims(X_train[:, 0, :], axis=2)
    # X1_test = np.expand_dims(X_test[:, 0, :], axis=2)
    # X1_val = np.expand_dims(X_val[:, 0, :], axis=2)
    #
    # X2_train = np.expand_dims(X_train[:, 1, :], axis=2)
    # X2_test = np.expand_dims(X_test[:, 1, :], axis=2)
    # X2_val = np.expand_dims(X_val[:, 1, :], axis=2)
    #
    # X_train = np.expand_dims(X_train, axis=3)
    # X_test = np.expand_dims(X_test, axis=3)
    # X_val = np.expand_dims(X_val, axis=3)
    #
    # print(X_train.shape)
    # print(X1_train.shape)
    # print(X2_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)
    # print(Y_train.shape)
    # print(Y_val.shape)
    # print(Y_test.shape)

