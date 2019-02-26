import numpy as np
import keras
import sklearn.metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from model import unet_1d

from datetime import datetime
import pickle
import os

class DataGenerator:
    def __init__(self):
        self.patient_X = np.load('patient_X.npy') # (852, 10, 10000)
        self.patient_y= np.load('patient_y.npy') # (103, 10, 10000)
        self.X = np.swapaxes(self.patient_X, 1, 2)
        self.y = np.swapaxes(self.patient_y, 1, 2)

    @staticmethod
    def normalize(X, means_and_stds=None):
        if means_and_stds is None:
            means = [ X[..., i].mean(dtype=np.float32) for i in range(X.shape[-1]) ]
            stds = [ X[..., i].std(dtype=np.float32) for i in range(X.shape[-1]) ]
        else:
            means = means_and_stds[0]
            stds = means_and_stds[1]

        normalized_X = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[-1]):
            normalized_X[..., i] = X[..., i].astype(np.float32) - means[i]
            normalized_X[..., i] = normalized_X[..., i] / stds[i]
        return normalized_X, (means, stds)

    @staticmethod
    def split(X, y, rs=42):
        # do patient split
        # patient_training_set, patient_valid_set, patient_test_set  = patient_split(X[:852, ...], y[:852, ...])

        # do normal split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

        # combine
        '''
        X_train = np.append(X_train, patient_training_set[0], axis=0)
        y_train = np.append(y_train, patient_training_set[1], axis=0)

        X_valid = np.append(X_valid, patient_valid_set[0], axis=0)
        y_valid = np.append(y_valid, patient_valid_set[1], axis=0)

        X_test = np.append(X_test, patient_test_set[0], axis=0)
        y_test = np.append(y_test, patient_test_set[1], axis=0)
        '''
        return [X_train, y_train], [X_valid, y_valid], [X_test, y_test]

    def X_shape(self):
        return self.X.shape[1:]

    def data(self):
        return self.X, self.y

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def train():
    g = DataGenerator()
    X, y = g.data()
    train_set, valid_set, test_set = g.split(X, y)

    model_checkpoints_dirname = 'model_checkpoints/'+datetime.now().strftime('%Y%m%d%H%M%S')
    tensorboard_log_dirname = model_checkpoints_dirname + '/logs'
    os.makedirs(model_checkpoints_dirname)
    os.makedirs(tensorboard_log_dirname)

    # do normalize using means and stds from training data
    train_set[0], means_and_stds = DataGenerator.normalize(train_set[0])
    valid_set[0], _ = DataGenerator.normalize(valid_set[0], means_and_stds)
    test_set[0], _ = DataGenerator.normalize(test_set[0], means_and_stds)

    # save means and stds
    with open(model_checkpoints_dirname + '/means_and_stds.txt', 'wb') as f:
        pickle.dump(means_and_stds, f)

    model = unet_1d()
    model.summary()

    callbacks = [
        # EarlyStopping(patience=5),
        ModelCheckpoint(model_checkpoints_dirname + '/{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
        TensorBoard(log_dir=tensorboard_log_dirname)
    ]

    print(1. - valid_set[1][:, 0].sum() / valid_set[1][:, 0].shape[0])
    print(1. - train_set[1][:, 0].sum() / train_set[1][:, 0].shape[0])

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=200, validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True)

    y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
    y_true = test_set[1][:, 1]
    print(sklearn.metrics.classification_report(y_true, y_pred))

    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['normal', 'patient'])

if __name__ == '__main__':
    train()