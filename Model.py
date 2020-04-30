from tensorflow import keras
from Blocks import Blocks

class Model():

    def ResUNet(self):
        f = [16, 32, 64, 128, 256]
        image_size = 224
        inputs = keras.layers.Input((image_size, image_size, 3))
        blocks = Blocks()
        ## Encoder
        e0 = inputs
        e1 = blocks.stem(e0, f[0])
        e2 = blocks.residual_block(e1, f[1], strides=2)
        e3 = blocks.residual_block(e2, f[2], strides=2)
        e4 = blocks.residual_block(e3, f[3], strides=2)
        e5 = blocks.residual_block(e4, f[4], strides=2)

        ## Bridge
        b0 = blocks.conv_block(e5, f[4], strides=1)
        b1 = blocks.conv_block(b0, f[4], strides=1)

        ## Decoder
        u1 = blocks.upsample_concat_block(b1, e4)
        d1 = blocks.residual_block(u1, f[4])

        u2 = blocks.upsample_concat_block(d1, e3)
        d2 = blocks.residual_block(u2, f[3])

        u3 = blocks.upsample_concat_block(d2, e2)
        d3 = blocks.residual_block(u3, f[2])

        u4 = blocks.upsample_concat_block(d3, e1)
        d4 = blocks.residual_block(u4, f[1])

        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
        model = keras.models.Model(inputs, outputs)
        return model