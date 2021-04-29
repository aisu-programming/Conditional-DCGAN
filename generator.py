''' Libraries '''
import tensorflow as tf


''' Functions '''
BCE = tf.keras.losses.BinaryCrossentropy()
def generator_loss(f_predict):
    loss = BCE([[0.9]]*len(f_predict), f_predict)
    return tf.math.reduce_mean(loss)


class MyGenerator(tf.keras.layers.Layer):

    def __init__(self, layer_num):
        super(MyGenerator, self).__init__()

        self.layer_num = layer_num

        self.CtgEmbed = tf.keras.layers.Embedding(10, 1024)
        self.Dense    = tf.keras.layers.Dense(4 * 4 * 512*(2**(self.layer_num-3)), use_bias=False)
        self.Reshape  = tf.keras.layers.Reshape((4, 4, 512*(2**(self.layer_num-3))))

        self.BatchNorms = [
            tf.keras.layers.BatchNormalization() for _ in range(layer_num) ]
        self.LeakyReLUs = [
            tf.keras.layers.LeakyReLU(alpha=0.2) for _ in range(layer_num) ]

        if layer_num == 3: self.ConvTrans = [
            # (None,  4,  4, 256)
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False),
            # (None,  8,  8, 128)
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False),
            # (None, 16, 16,  64)
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', use_bias=False, activation='tanh'),
            # (None, 32, 32,   3)
        ]
        elif layer_num == 4: self.ConvTrans = [
            # (None,  4,  4, 512)
            tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', use_bias=False),
            # (None,  8,  8, 256)
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False),
            # (None, 16, 16, 128)
            tf.keras.layers.Conv2DTranspose(64, 3, padding='same', use_bias=False),
            # (None, 16, 16,  64)
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', use_bias=False, activation='tanh'),
            # (None, 32, 32,   3)
        ]
        elif layer_num == 5: self.ConvTrans = [
            # (None,  4,  4, 1024)
            tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', use_bias=False),
            # (None,  8,  8,  512)
            tf.keras.layers.Conv2DTranspose(256, 3, padding='same', use_bias=False),
            # (None,  8,  8,  256)
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False),
            # (None, 16, 16,  128)
            tf.keras.layers.Conv2DTranspose(64, 3, padding='same', use_bias=False),
            # (None, 16, 16,   64)
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', use_bias=False, activation='tanh'),
            # (None, 32, 32,    3)
        ]
        elif layer_num == 6: self.ConvTrans = [
            # (None,  4,  4, 2048)
            tf.keras.layers.Conv2DTranspose(1024, 3, padding='same', use_bias=False),
            # (None,  4,  4, 1024)
            tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', use_bias=False),
            # (None,  8,  8,  512)
            tf.keras.layers.Conv2DTranspose(256, 3, padding='same', use_bias=False),
            # (None,  8,  8,  256)
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False),
            # (None, 16, 16,  128)
            tf.keras.layers.Conv2DTranspose(64, 3, padding='same', use_bias=False),
            # (None, 16, 16,   64)
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', use_bias=False, activation='tanh'),
            # (None, 32, 32,    3)
        ]

    def call(self, noise, catergory=None):

        assert noise.shape[1:] == [ 1024 ]
        if catergory is None:
            x = noise
        else:
            catergory = tf.squeeze(self.CtgEmbed(catergory))
            assert catergory.shape[1:] == [ 1024 ]
            x = noise + catergory

        x = self.Dense(x)
        assert x.shape[1:] == [ 4 * 4 * 512*(2**(self.layer_num-3)) ]
        x = self.Reshape(x)
        assert x.shape[1:] == [ 4, 4, 512*(2**(self.layer_num-3)) ]

        for i in range(self.layer_num):
            x = self.BatchNorms[i](x)
            x = self.LeakyReLUs[i](x)
            x = self.ConvTrans[i](x)

        x = ((x + 1) / 2) * 255
        return x