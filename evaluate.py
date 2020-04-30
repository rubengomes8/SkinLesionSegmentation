import os
import pandas as pd
import DataGen
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Model

# Tutorial: https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


dataset_path = "/home/ruben/Documentos/dataset/"

train_path = os.path.join(dataset_path, "train/")

train_csv = pd.read_csv(train_path + "train.csv")
train_ids = train_csv["id"].values # coluna do nome da imagem

image_size = 224
batch_size = 10
val_data_size = 200

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

gen = DataGen.DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)


# Model
model_class = Model.Model()
model = model_class.ResUNet()
model.summary()
model.load_weights("/home/ruben/Desktop/Segmentation/ResUnet/cp.ckpt")

train_gen = DataGen.DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen.DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)



for i in range(0, len(valid_gen)):
    x, y = valid_gen.__getitem__(i)
    result = model.predict(x)
    result = result > 0.4
    for j in range(len(x)):
        plt.imshow(np.reshape(y[j] * 255, (image_size, image_size)), cmap="gray")
        plt.show()
        plt.imshow(np.reshape(result[j] * 255, (image_size, image_size)), cmap="gray")
        plt.show()