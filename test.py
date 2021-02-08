import pickle
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def format_data_choice_snr(snr_choice=None):
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


X = format_data_choice_snr()
print(X.shape)
# data = pd.DataFrame(X)
# n = list(map(lambda x: str(x), range(0, 257)))
# n[0] = 'label'
# data.columns = n
#
# # # data['1'] = list(map(lambda x: x*10000000000, data['1']))
# # v = data['1'].value_counts()
# # print(v)
# #
# data = data['200']
# data.plot()
# plt.show()


# TAG_SET = list(key2index.keys())  ##表示所有items的集合##
#
#
# def sparse_from_csv(csv):
#     post_tags_str = list(csv)
#     table = tf.lookup.StaticHashTable(
#         tf.lookup.KeyValueTensorInitializer(list(key2index.keys()), list(key2index.values())),
#         default_value=-1)  ## 这里构造了个查找表 ##
#     split_tags = tf.strings.split(csv, " ")
#     split_tags = tf.RaggedTensor.to_sparse(split_tags)
#     return tf.SparseTensor(
#         indices=split_tags.indices,
#         values=table.lookup(split_tags.values),  ## 这里给出了不同值通过表查到的index ##
#         dense_shape=split_tags.dense_shape)
#
#
# EMBEDDING_DIM = 16  ##embedding向量的维数##
# embedding_params = tf.Variable(tf.random.truncated_normal([len(TAG_SET) + 1, EMBEDDING_DIM]))
# tags = sparse_from_csv(train['item_id'])
# embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)
