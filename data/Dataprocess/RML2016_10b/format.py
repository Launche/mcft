import pickle

import numpy as np


def format_data_choice_snr(snr_choice=18):
    f = "/tmp/project/mcft/data/Dataprocess/RML2016_10b/RML2016.10b_dict.dat"

    mod_dict = {
        1: '8PSK', 2: 'AM-DSB', 3: 'BPSK', 4: 'CPFSK', 5: 'GFSK', 6: 'PAM4', 7: 'QAM16', 8: 'QAM64', 9: 'QPSK',
        10: 'WBFM'
    }

    mod_dict2 = {
        '8PSK': 1, 'AM-DSB': 2, 'BPSK': 3, 'CPFSK': 4, 'GFSK': 5, 'PAM4': 6, 'QAM16': 7, 'QAM64': 8, 'QPSK': 9,
        'WBFM': 10
    }

    Xd = pickle.load(open(f, 'rb'), encoding='latin')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    print(mods, snrs)
    for mod in mods:
        if snr_choice:
            for snr in [snr_choice]:
                raw = Xd[(mod, snr)]
                data = raw.reshape(raw.shape[0], -1)
                target = np.array([mod_dict2[mod]] * data.shape[0])
                data = np.insert(data, 0, values=target, axis=1)
                X.append(data)
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        else:
            for snr in snrs:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
    X = np.vstack(X)
    return X


X = format_data_choice_snr(18)
print(X[0])