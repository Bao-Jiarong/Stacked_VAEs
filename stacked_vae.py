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

        # Latent Space
        self.la_dense1= tf.keras.layers.Dense(units = self.latent_dim   , name = "la_fc1")
        self.la_dense2= tf.keras.layers.Dense(units = self.latent_dim   , name = "la_fc2")

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
    # Latent Space
    #................................................................................
    def latent_space(self,x):
        mu  = self.la_dense1(x)
        std = self.la_dense2(x)
        shape = mu.shape[1:]
        eps = tf.random.normal(shape, 0.0, 1.0)
        x = mu + eps * (tf.math.exp(std/2.0))
        return x

    #................................................................................
    #
    #................................................................................
    def call(self, inputs, training = None):
        # inputs = self.in_layer(inputs)
        # self.encoded = self.encoder(inputs, training)
        #
        # self.latent_space = self.latent_space(self.encoded)
        #
        # self.decoded = self.decoder(self.latent_space, training)

        x = self.encoder(inputs, training)

        x = self.latent_space(x)

        x = self.decoder(x, training)
        return x
