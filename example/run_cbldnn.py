import os
import sys

import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, ZeroPadding2D, Dropout, Flatten, Dense, Reshape, \
    LSTM, Bidirectional
from tensorflow.keras import Model
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"


def getConfusionMatrixPlot(true_labels, predicted_labels, mods):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm, 2)
    print(cm)

    print()
    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = mods
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt


def gendata(fp, snr_choice):
    Xd = pickle.load(open(fp, 'rb'), encoding='latin')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    print(mods, snrs)
    for mod in mods:
        if snr_choice:
            for snr in [snr_choice, 0]:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        else:
            for snr in snrs:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
    X = np.vstack(X)
    np.random.seed(2020)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.75)
    print(n_train)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    if not snr_choice:
        # uniformly random selection across all snrs
        X_train_reduced = np.empty((0, 2, 128))
        Y_train_reduced = np.empty((0, 10))
        train_SNRs = map(lambda x: lbl[x][1], train_idx)
        train_snr = lambda snr: X_train[np.where(np.array(train_SNRs) == snr)]
        test_snr = lambda snr: Y_train[np.where(np.array(train_SNRs) == snr)]
        for snr in snrs:
            X_train_i = train_snr(snr)
            n_examples = X_train_i.shape[0]
            per_snr_size = n_examples // 128
            train_idx = np.random.choice(range(0, n_examples), size=per_snr_size, replace=False)
            X_train_reduced = np.append(X_train_reduced, X_train_i[train_idx], axis=0)
            Y_train_i = test_snr(snr)
            Y_train_reduced = np.append(Y_train_reduced, Y_train_i[train_idx], axis=0)

        X_train = X_train_reduced
        Y_train = Y_train_reduced
    print('SHAPE OF X_TRAIN:', X_train.shape)
    print('SHAPE OF X_TEST:', X_test.shape)
    return X_train, X_test, Y_train, Y_test, mods


class CBLDNN(Model):
    def __init__(self, in_shp, dropout):
        super(CBLDNN, self).__init__()
        self.r1 = Reshape(in_shp + [1], input_shape=in_shp)
        self.p1 = ZeroPadding2D((0, 2))
        self.c1 = Conv2D(kernel_initializer="glorot_uniform", name="conv1", activation="relu",
                         padding="valid", filters=256, kernel_size=(1, 3))
        self.d1 = Dropout(dropout)
        self.p2 = ZeroPadding2D((0, 2))

        self.c2 = Conv2D(kernel_initializer="glorot_uniform", name="conv2", activation="relu",
                         padding="valid", filters=256, kernel_size=(2, 3))
        self.d2 = Dropout(dropout)
        self.p3 = ZeroPadding2D((0, 2))
        self.c3 = Conv2D(kernel_initializer="glorot_uniform", name="conv3", activation="relu",
                         padding="valid", filters=80, kernel_size=(1, 3))
        self.d3 = Dropout(dropout)
        self.p4 = ZeroPadding2D((0, 2))
        self.c4 = Conv2D(kernel_initializer="glorot_uniform", name="conv4", activation="relu",
                         padding="valid", filters=80, kernel_size=(1, 3))
        self.d4 = Dropout(dropout)
        self.p5 = ZeroPadding2D((0, 2))

        self.flatten = Flatten()
        self.r2 = Reshape((1, 11200))
        self.l1 = Bidirectional(LSTM(50))
        self.f1 = Dense(128, activation='relu', kernel_initializer='he_normal', name="Dense1")
        self.d5 = Dropout(dropout)
        self.f2 = Dense(10, activation='softmax', kernel_initializer='he_normal', name="Dense1")

    def call(self, x):
        x = self.r1(x)
        x = self.p1(x)
        x = self.c1(x)
        x = self.d1(x)

        x = self.p2(x)
        x = self.c2(x)
        x = self.d2(x)

        x = self.p3(x)
        x = self.c3(x)
        x = self.d3(x)
        x = self.p4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.p5(x)

        x = self.flatten(x)
        x = self.r2(x)
        x = self.l1(x)

        x = self.f1(x)
        x = self.d5(x)
        y = self.f2(x)
        return y


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 1. config parameter
    dropout = 0.6
    snr = 18
    batch_size = 1024
    epochs = 500
    prefix_dir = '/tmp/project/mcft/'

    if sys.argv.__len__() == 3:
        snr = int(sys.argv[1])
        epochs = int(sys.argv[2])

    str_snr = str(snr)
    if snr < 0:
        str_snr = str_snr.replace('-', 'i')
    print(snr, epochs)
    # 2. generate input data for model
    file_name = "/tmp/project/mcft/data/Dataprocess/RML2016_10b/RML2016.10b_dict.dat"
    X_train, X_test, Y_train, Y_test, classes = gendata(file_name, snr)
    in_shp = list(X_train.shape[1:])

    # 2. Define Model,train,predict and evaluate
    model = CBLDNN(in_shp=in_shp, dropout=dropout)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    log_dir = prefix_dir + 'cbldnn_' + '_' + str(epochs)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ckpt_dir = prefix_dir + 'output/cbldnn_%s/' % str_snr
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = ckpt_dir + 'cbldnn_%s_%s_%s.ckpt' % (str_snr, str(dropout), str(epochs))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_file, monitor='val_loss', verbose=0, save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='auto')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.25
                        , callbacks=[logs, checkpoint, early_stop])

    model.load_weights(ckpt_file)

    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    print(score)

    ###############################################    show   ###############################################

    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    logitPic = prefix_dir + "figures/cbldnn/logit_%s_%s_%s.png" % (str(snr), str(dropout), str(epochs))

    plt.savefig(logitPic)

    plt.show()

    # estimate classes
    test_Y_i_hat = np.array(model.predict(X_test))
    acc = {}
    width = 4.1
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(Y_test, 1), np.argmax(test_Y_i_hat, 1), classes)
    plt.gcf().subplots_adjust(bottom=0.15)
    confusionPic = prefix_dir + "figures/cbldnn/confusion_%s_%s_%s.png" % (str(snr), str(dropout), str(epochs))
    plt.savefig(confusionPic)

    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, X_test.shape[0]):
        j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_i_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)
    print(acc)
    plt.show()
