from tensorflow import keras

class Blocks():


    def bn_act(self, x, act=True):
        x = keras.layers.BatchNormalization()(x)
        if act == True:
            x = keras.layers.Activation("relu")(x)
        return x

    def conv_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = self.bn_act(x)
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        output = keras.layers.Add()([conv, shortcut])
        return output

    def residual_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        res = self.conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = self.conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)

        output = keras.layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(self, x, xskip):
        u = keras.layers.UpSampling2D((2, 2))(x)
        c = keras.layers.Concatenate()([u, xskip])
        return c



