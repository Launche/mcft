import numpy as np
import pickle


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
