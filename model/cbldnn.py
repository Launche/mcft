from tensorflow.keras.layers import Conv2D, ZeroPadding2D, MaxPool2D, Dropout, Flatten, Dense, Reshape, \
    LSTM, Bidirectional
from tensorflow.keras import Model


# Build VT-CNN2 Neural Net model using Keras primitives --
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization


# Build the NN Model

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
