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
x, y = gen.__getitem__(0)
# print(x.shape, y.shape)

r = random.randint(0, len(x)-1)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(x[r])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[r]*255, (image_size, image_size)), cmap="gray")


# Model
model_class = Model.Model()
model = model_class.ResUNet()
adam = tf.keras.optimizers.Adam(lr=1e-5)
model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

# Train and load the best Model
train_gen = DataGen.DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen.DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

epochs = 40

checkpoint_path = "/home/ruben/Desktop/Segmentation/ResUnet/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, save_best_only=True)

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                    epochs=epochs, callbacks=[cp_callback])

model.load_weights(checkpoint_path)

# Predict and plot
print("\n      Ground Truth            Predicted Value")

for i in range(1, 5, 1):
    ## Dataset for prediction
    x, y = valid_gen.__getitem__(i)
    result = model.predict(x)
    result = result > 0.4

    for i in range(len(result)):

        fig = plt.plot()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(np.reshape(y[i] * 255, (image_size, image_size)), cmap="gray")

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(np.reshape(result[i] * 255, (image_size, image_size)), cmap="gray")

'''
        f, axarr = plt.subplots(2, 2)
        
        axarr[0, 0].imshow(image_datas[0])
        axarr[0, 1].imshow(image_datas[1])
        axarr[1, 0].imshow(image_datas[2])
        axarr[1, 1].imshow(image_datas[3])
        plt.plot(x, bacc, marker='o', label='Mixed')
        hier = [0.643] * 21
        plt.plot(x, hier, 'r--', marker='', label='Hier')
        flat = [0.645] * 21
        plt.plot(x, flat, 'g:', label='Flat')
        # plt.ylabel('BACC.', fontsize=16)
        plt.xlabel('η (%)', fontsize=16)
'''