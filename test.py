#!/usr/bin/env python3

from hrr import *
import tensorflow as tf
import numpy as np

hrr_size = 5
alpha = 0.1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(hrr_size, activation='sigmoid', input_dim=hrr_size))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Nadam(alpha),
                metrics=['accuracy'])

model.summary()

test = [
    [1,1,1,1,1],
    [1,1,1,1,1]
]

print(model.predict(test))
