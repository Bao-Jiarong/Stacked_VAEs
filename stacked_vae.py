'''
    ------------------------------------
    Author : Bao Jiarong
    Date   : 2020-08-30
    Project: Variational AE (stacked)
    Email  : bao.salirong@gmail.com
    ------------------------------------
'''
import tensorflow as tf

class STACKED_VAE(tf.keras.Model):
    #................................................................................
    # Constructor
    #................................................................................
    def __init__(self, image_size = 28, latent_dim = 200):
        super(STACKED_VAE, self).__init__(name = "STACKED_VAE")

        self.image_size = image_size  # height and weight of images
        self.linear_dim = (image_size ** 2) >> 1
        self.latent_dim = latent_dim
        self.my_in_shape= [image_size, image_size, 3]

        # Encoder Layers
        self.flatten  = tf.keras.layers.Flatten()
        self.en_dense1= tf.keras.layers.Dense(units = self.linear_dim   , activation="relu", name = "en_fc1")
        self.en_dense2= tf.keras.layers.Dense(units = self.linear_dim>>1, activation="relu", name = "en_fc2")
        self.en_dense3= tf.keras.layers.Dense(units = self.latent_dim   , name = "en_fc3")
        self.en_act   = tf.keras.layers.Activation("relu", name = "en_main_out")

        # Decoder Layers
        self.de_dense1= tf.keras.layers.Dense(units = self.linear_dim>>1, activation="relu", name = "de_fc1")
        self.de_dense2= tf.keras.layers.Dense(units = self.linear_dim   , activation="relu", name = "de_fc2")
        self.de_dense3= tf.keras.layers.Dense(units = (image_size ** 2) * 3, name = "de_fc3")
        self.de_act   = tf.keras.layers.Activation("sigmoid")
        self.reshape  = tf.keras.layers.Reshape(self.my_in_shape, name = "de_main_out")

    #................................................................................
    # Encoder Space
    #................................................................................
    def encoder(self, x, training = None):
        # Encoder Space
        x = self.flatten(x)
        x = self.en_dense1(x)
        x = self.en_dense2(x)
        # Latent Space
        x = self.en_dense3(x)
        x = self.en_act(x)
        return x

    #................................................................................
    # Decoder Space
    #................................................................................
    def decoder(self, x, training = None):
        # Decoder Space
        x = self.de_dense1(x)
        x = self.de_dense2(x)
        x = self.de_dense3(x)
        x = self.de_act(x)
        x = self.reshape(x)
        return x

    #................................................................................
    #
    #................................................................................
    def call(self, inputs, training = None):
        # inputs = self.in_layer(inputs)
        self.encoded = self.encoder(inputs, training)

        shape = self.encoded.shape[1:]
        x = tf.random.uniform(shape, minval=0.0, maxval=1.0)
        de_input = self.encoded + x

        self.decoded = self.decoder(de_input, training)
        return self.decoded
