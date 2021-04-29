''' Libraries '''
import tensorflow as tf


''' Functions '''
BCE = tf.keras.losses.BinaryCrossentropy()
def discriminator_loss(r_predict, f_predict):
    # r_loss = BCE([[1.]]*len(r_predict), r_predict)
    r_loss = BCE([[0.9]]*len(r_predict), r_predict)
    # f_loss = BCE([[0.]]*len(f_predict), f_predict)
    f_loss = BCE([[0.1]]*len(f_predict), f_predict)
    return tf.math.reduce_mean(r_loss + f_loss) / 2


class MyDiscriminator(tf.keras.layers.Layer):
    def __init__(self, layer_num, dropout):
        super(MyDiscriminator, self).__init__()
        self.layer_num = layer_num
        self.Convs = []
        for i in range(layer_num):
            if i % 2 == 0: self.Convs.append(
                tf.keras.layers.Conv2D(64*(2**i), 3, strides=(2, 2), padding='same'))
            else: self.Convs.append(
                tf.keras.layers.Conv2D(64*(2**i), 3, padding='same'))
        self.LeakyReLUs = [ 
            tf.keras.layers.LeakyReLU(alpha=0.2) for _ in range(layer_num) ]
        self.Flatten    = tf.keras.layers.Flatten()
        self.Dropout    = tf.keras.layers.Dropout(dropout)
        self.CtgEmbed   = tf.keras.layers.Embedding(10, 32768)
        self.RealOrFake = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    def call(self, x, catergory=None, training=False):
        assert x.shape[1:] == [ 32, 32, 3 ]
        for i in range(self.layer_num):
            x = self.Convs[i](x)
            x = self.LeakyReLUs[i](x)
            assert x.shape[1:] == [
                int(16*(0.5**(i//2))), int(16*(0.5**(i//2))), 64*(2**i) ]
        x = self.Flatten(x)
        assert x.shape[1:] == [ 32768 ]
        if catergory is not None:
            x += tf.squeeze(self.CtgEmbed(catergory))
        x = self.Dropout(x, training=training)
        x = self.RealOrFake(x)
        return x