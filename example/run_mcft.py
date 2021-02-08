import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split


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
raw = pd.DataFrame(X)
feats = list(map(lambda x: str(x), range(0, 257)))
feats[0] = 'label'
raw.columns = feats

dense_feats = feats[1:]

feature_names = dense_feats


def process_dense_feats(data, feats):
    d = data.copy()
    d = d[feats].fillna(0.0)
    # for f in feats:
    #     d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    return d


data = process_dense_feats(raw, dense_feats)
data['label'] = raw['label']

# embedding层
k = 8  # embedding 维度

# 构造输入
dense_inputs = []
for f in dense_feats:
    _input = Input([1], name=f)
    dense_inputs.append(_input)
# 对输入进行嵌入
dense_kd_embed = []
for i, _input in enumerate(dense_inputs):
    f = dense_feats[i]
    embed = tf.Variable(tf.random.truncated_normal(shape=(1, k), stddev=0.01), name=f)

    scaled_embed = tf.expand_dims(_input * embed, axis=1)

    dense_kd_embed.append(scaled_embed)

print(dense_kd_embed)
input_embeds = dense_kd_embed
# 构建feature map
embed_map = Concatenate(axis=1)(input_embeds)  # ?, 257, 8

print(embed_map)
def auto_interacting(embed_map, d=6, n_attention_head=2):
    """
    实现单层 AutoInt Interacting Layer
    @param embed_map: 输入的embedding feature map, (?, n_feats, n_dim)
    @param d: Q,K,V映射后的维度
    @param n_attention_head: multi-head attention的个数
    """
    assert len(embed_map.shape) == 3, "Input embedding feature map must be 3-D tensor."

    k = embed_map.shape[-1]

    # 存储多个self-attention的结果
    attention_heads = []
    W_Q = []
    W_K = []
    W_V = []

    # 1.构建多个attention
    for i in range(n_attention_head):
        # 初始化W_Q, W_K, W_V
        W_Q.append(tf.Variable(tf.random.truncated_normal(shape=(k, d)), name="query_" + str(i)))  # k, d 8，6
        W_K.append(tf.Variable(tf.random.truncated_normal(shape=(k, d)), name="key_" + str(i)))  # k, d
        W_V.append(tf.Variable(tf.random.truncated_normal(shape=(k, d)), name="value_" + str(i)))  # k, d

    for i in range(n_attention_head):
        # 映射到d维空间
        embed_q = tf.matmul(embed_map, W_Q[i])  # ?, 39, d
        embed_k = tf.matmul(embed_map, W_K[i])  # ?, 39, d
        embed_v = tf.matmul(embed_map, W_V[i])  # ?, 39, d

        # 计算attention
        energy = tf.matmul(embed_q, tf.transpose(embed_k, [0, 2, 1]))  # ?, 39, 39
        attention = tf.nn.softmax(energy)  # ?, 39, 39

        attention_output = tf.matmul(attention, embed_v)  # ?, 39, d
        attention_heads.append(attention_output)

    # 2.concat multi head
    multi_attention_output = Concatenate(axis=-1)(attention_heads)  # ?, 39, n_attention_head*d

    # 3.ResNet
    w_res = tf.Variable(tf.random.truncated_normal(shape=(k, d * n_attention_head)),
                        name="w_res_" + str(i))  # k, d*n_attention_head
    output = Activation("relu")(multi_attention_output + tf.matmul(embed_map, w_res))  # ?, 39, d*n_attention_head)

    return output


def build_autoint(x0, n_layers):
    xl = x0
    for i in range(n_layers):
        xl = auto_interacting(xl, d=6, n_attention_head=2)

    return xl


# 构建3层interacting layer
autoint_layer = build_autoint(embed_map, 3)

autoint_layer = Flatten()(autoint_layer)

output_layer = Dense(10, activation="softmax")(autoint_layer)

# 声明模型：指定输入和输出
# model = Model(dense_inputs + sparse_inputs, output_layer)
model = Model(dense_inputs, output_layer)

# 模型的编译，指定loss、优化器、评估指标
# model.compile(optimizer="adam",
#               loss="binary_crossentropy",
#               metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

# model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
#               optimizer=tf.keras.optimizers.Adam(lr=0.001),
#               metrics=['accuracy'])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# train_data = data.loc[:50000 - 1]
# valid_data = data.loc[50000:]
#
# train_dense_x = [train_data[f].values for f in dense_feats]
# # train_sparse_x = [train_data[f].values for f in sparse_feats]
#
# train_label = [train_data['label'].values]
#
# val_dense_x = [valid_data[f].values for f in dense_feats]
# # val_sparse_x = [valid_data[f].values for f in sparse_feats]
# val_label = [valid_data['label'].values]

target = ['label']
train, test = train_test_split(data, test_size=0.2, random_state=2020)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}
#
# model.fit(train_dense_x,
#           train_label, epochs=5, batch_size=256,
#           validation_data=(val_dense_x, val_label), )
# # callbacks=[tbCallBack])

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

score = model.evaluate(test_model_input, test[target].values, verbose=0, batch_size=256)
print(score)
