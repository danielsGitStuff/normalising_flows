from __future__ import annotations

from keras import Input, Model
from keras.layers import Dense
from keras.losses import BinaryCrossentropy

from keta.lazymodel import LazyModel
from maf.mixlearn.ClassifierTrainingProcess import BinaryClassifierCreator


class MiniBooneBinaryClassifierCreator(BinaryClassifierCreator):
    def __init__(self):
        super().__init__()

    def create_classifier(self, input_dims: int) -> LazyModel:
        ins = Input(shape=(input_dims,))
        b = Dense(512, activation='relu')(ins)
        # b = Dropout()
        b = Dense(512, activation='relu')(b)
        b = Dense(1, activation='sigmoid')(b)
        # b = Dense(100, activation='relu', name='DenseRELU0')(ins)
        # b = Dense(100, activation='relu')(b)
        # b = BatchNormalization()(b)
        # b = Dense(100, activation='relu')(b)
        # b = Dense(1, activation='linear', name='out')(b)
        model = Model(inputs=[ins], outputs=[b])
        lm = LazyModel.Methods.wrap(model)
        lm.compile(optimizer='adam', loss=BinaryCrossentropy(), lr=0.001, metrics=['accuracy'])
        return lm