import math
from math import e
from math import sqrt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sc
from scipy import integrate
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from time import time

from ODE import ODE

DTYPE = "float32"


class NN:
    """
    A class to represent a neural net

    Attributes:
        set_b: boolean
            determines, weighting function is used
        batchsize: int
            batch size for training data
        t_0: float
            initial time
        t_f: float
            end time
        ode: ode instance from ODE.py
            an instance of given ODE system
        t_dim: int
            dimension of time
        x_dim: int
            dimension of x
        rho_dim: int
            dimension of other parameters (physical parameters etc.)
        input_data: matrix
            batchsize x (t_dim + x_dim + rho_dim) matrix of data, which is used to train
        g_function: function
            G(x, x´,t,rho) = 0
        a_function: function
            a(t) = 1 - exp(t_0 -t)
        x_approx_function: function
            x(t,x_0,rho) = x_0 + a(t) * model(input_data)
        lam: float
            lambda > 0
        mu_function: function
            mu(t) = exp(lambda * (t_0 -t))
        model: tf.keras.Sequential()
            sequential neural net model with tensorflow, keras
        neurons_per_layer: array
            array of shape [n^(1), ..., n^(l-1)] with n^(l) being the amount of neurons per layer and
            len(neurons_per_layer) the amount of hidden layers
        epochs: int
            epoch count
        learning_rate: float
            learning rate of neural net
    """

    def __init__(self, time_interval, batchsize, ode, set_b=True):
        """
        initializes the neural net
        :param time_interval: array
            array of shape [t_0, t_f] with t_0 being initial time and t_f end time
        :param batchsize: int
            batch size
        :param ode: instance of ODE.py class
            used to get all ODE specific data
        :param set_b: boolean
            configures weighting function mu
        """
        self.set_b = set_b
        self.batchsize = batchsize
        (self.t_0, self.t_f) = time_interval
        self.ode = ode

        self.input_data = None
        self.g_function = None
        self.x_approx_function = None
        self.a_function = None
        self.mu_function = None

        self.lam = None
        self.t_dim = None
        self.x_dim = None
        self.rho_dim = None

        self.model = None
        self.neurons_per_layer = None
        self.hidden_layer = None
        self.epochs = None
        self.learning_rate = None

    def set_functions(self):
        """
        initializes functions and some attributes
        """
        self.lam = 0.5
        self.g_function = self.ode.ml_G
        self.a_function = self.a
        self.x_approx_function = self.x_Approx
        self.mu_function = self.mu
        self.t_dim, self.x_dim, self.rho_dim = self.ode.dimension_list

    def init_model(self, neurons_per_layer, epochs):
        """
        initializes the neural net model
        :param neurons_per_layer: array
            array of shape [n^(1), ..., n^(l-1)] with n^(l) being the amount of neurons per layer and
            len(neurons_per_layer) the amount of hidden layers
        :param epochs: int
            amount of epochs
        """
        self.neurons_per_layer = neurons_per_layer
        self.hidden_layer = len(neurons_per_layer)
        self.epochs = epochs

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(self.x_dim + self.t_dim + self.rho_dim,)))

        for layer in range(self.hidden_layer):
            self.model.add(tf.keras.layers.Dense(
                self.neurons_per_layer[layer],
                activation=tf.keras.activations.get("tanh"),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
        self.model.add(tf.keras.layers.Dense(2))

    def get_time_derivative(self, data):
        """
        function to calculate the derivative of the approximation x_hat
        :param data: matrix
            matrix of shape batchsize x (t_dim + x_dim + rho_dim) of data containing t,x and rho
        :return:
            tf.concat([u_t_1, u_t_2], axis=1): tf.array
                tf array containing the first and second variable of the derivative of approximation x_hat
        """
        t = data[:, 0:self.t_dim]
        x = data[:, self.t_dim:self.t_dim + self.x_dim]
        rho = data[:, self.t_dim + self.x_dim: self.t_dim + self.x_dim + self.rho_dim]

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(t)
            u_1 = x[:, 0:1] + tf.math.multiply(self.a(t), self.model(tf.concat([t, x, rho], axis=1))[:, 0:1])
        u_t_1 = tape.gradient(u_1, t)
        del tape

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(t)
            u_2 = x[:, 1:2] + tf.math.multiply(self.a(t), self.model(tf.concat([t, x, rho], axis=1))[:, 1:2])
        u_t_2 = tape.gradient(u_2, t)
        del tape
        return tf.concat([u_t_1, u_t_2], axis=1)

    def a(self, t):
        """
        function for a(t)
        :param t: array
            array of points of the time interval
        :return:
            a(t): tf.array
                returning the a(t) = 1 - exp(t_0 - t)
        """
        return 1. - tf.math.exp(-tf.math.subtract(t, self.t_0))

    def x_Approx(self, data):
        """
        function for the approximation x_hat
        :param data: matrix
            matrix of shape batchsize x (t_dim + x_dim + rho_dim) of data containing t,x and rho
        :return:
            x(t,x_0,rho): matrix
                returning x(t,x_0,rho) = x_0 + a(t) * N(t,x_0,rho,weights)
        """
        return self.input_data[:, self.t_dim:self.t_dim + self.x_dim] + \
               self.a(self.input_data[:, 0:self.t_dim]) * self.model(data)

    def mu(self, t):
        """
        function for the weighting function mu
        :param t: array
            array of points of the time interval
        :return:
            mu(t): array
                returning mu(t) = exp(lambda * (t_0 - t))
        """
        if self.set_b is True:
            return tf.math.exp(-self.lam * t)
        else:
            return 1.

    def C(self):
        """
        calculates the loss for given training data
        :return:
            Loss: float
                returns the Loss C
        """
        return tf.math.reduce_mean(
            self.mu(self.input_data[:, 0:self.t_dim]) *
            tf.square(
                tf.reshape(tf.norm(
                    self.g_function(
                        self.x_approx_function(self.input_data),
                        self.get_time_derivative(self.input_data),
                        self.input_data[:, 0:self.t_dim],
                        self.input_data[:, self.t_dim + self.x_dim:self.t_dim + self.x_dim + self.rho_dim]
                    ), axis=1), shape=(-1, 1))
            )
        )

    def C_derivative(self):
        """
        calculates the derivative of the loss function C
        :return:
            loss_weight: Tensor
                weights
            C_der: Tensor
                Gradient of loss function
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss_weights = self.C()
        C_der = tape.gradient(loss_weights, self.model.trainable_variables)
        del tape
        return loss_weights, C_der

    def optimize(self, learning_rate, t, X, sol):
        """
        trains the neural net
        :param learning_rate: float
            learning rate
        :param t: array
            time array, used for error calculation
        :param X: matrix
            data of x and rho, used for error calculation
        :param sol: matrix
            exact solution
        :return:
            history: array
                loss history
            error: array
                error history
            time: float
                training time
        """
        self.learning_rate = learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step():
            loss_weights, L_der = self.C_derivative()
            optimizer.apply_gradients(zip(L_der, self.model.trainable_variables))
            return loss_weights

        history = []
        error = []
        t0 = time()

        self.ode.initialize_data(batchsize=self.batchsize)
        self.input_data = self.ode.initial_data
        for i in range(self.epochs):  # training
            t_c = (t[-1].numpy() * tf.math.log(10. * i / self.epochs + 1.)) / tf.math.log(11.)  # curriculum learning
            self.lam = 4 / (t_c + 5)
            loss = train_step()
            history.append(loss.numpy())
            if sol is not None:
                error.append(tf.norm((X[:, 1:3] + self.a(t) * self.model(X)) - tf.transpose(tf.constant(sol))).numpy())
            self.ode.initialize_data(batchsize=self.batchsize, t_c=t_c)
            self.input_data = self.ode.initial_data
            if i % 50 == 0:
                print('It {:05d}: loss = {:10.8e}'.format(i, loss))
                print('\nComputation time: {} seconds'.format(time() - t0))
        return history, error, time() - t0
