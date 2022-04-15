import numpy as np
import tensorflow as tf

DTYPE = "float32"


class ODE:
    """
    A class to initialize all needed data for a ODE system

    Attributes:
        ode_name: string
            name of the ODE
        numerical_f: function
            the right side f(t,x) of the ODE
        ml_G: function
            function G(x,x´,t,rho)=0
        ode_list: array
            array of ode_names
        dimension_list: array
            array of shape [t_dim, x_dim, rho_dim] with t_dim being the dimension of t, x_dim the dimension of x and
            tho_dim the dimension of rho
        param_list_num: array
            array of (physical) parameters
        param_list: array
            array of all (physical) parameters, that don´t get trained by the neural net
        initial_data: matrix
            training data for the neural net
        t0: float
            initial time
        tf: float
            end time
    """
    def __init__(self, ode_name, t0, tf):
        """
        initializes an instance of the class
        :param ode_name: string
            name of the ODE
        :param t0: float
            initial time
        :param tf: float
            end time
        """
        self.ode_name = ode_name
        self.numerical_f = self.numerical_func  # right side f of ODE
        self.ml_G = self.ml_G_func  # G = dx - f
        self.ode_list = ['harmonicoscillator', 'reboundpendulum', 'stiff']
        self.dimension_list = None  # dimension_list = [t_dim, x_dim, rho_dim]
        self.param_list_num = None
        self.param_list = None
        self.initial_data = None
        self.t0 = t0
        self.tf = tf
        if ode_name == 'harmonicoscillator':
            self.dimension_list = [1, 2, 1]
            self.param_list_num = [1., 1.]  # m=1, k=1
            self.param_list = 1.  # m=1
        elif ode_name == 'reboundpendulum':
            self.dimension_list = [1, 2, 2]
            self.param_list_num = [1., 1., 1., 3.0, 1.]  # l=1, m=1, g=1, k=3, c=1
            self.param_list = [1., 1., 1.]  # l=1, m=1, g=1
        elif ode_name == 'stiff':
            self.dimension_list = [1, 2, 2]
            self.param_list_num = [-100., -1., 3., 4.]  # lam_1=-100, lam_2=-1, c_1=3, c_2=4

    def numerical_func(self, t, x):
        """
        returns the right side of the ODE
        :param t: float
            one point of time
        :param x: array
            space variables [x1, ..., xn]
        :return:
            f: array
                calculated right side
        """
        if self.ode_name == 'harmonicoscillator':
            m, k = self.param_list_num
            return [
                x[1],
                -(k * x[0]) / m
            ]
        elif self.ode_name == 'reboundpendulum':
            l, m, g, k, c = self.param_list_num
            return [
                x[1],
                - (g * np.sin(x[0])) / l + np.heaviside(-x[0], 0.) * np.maximum(
                    - np.multiply(l * k / m, x[0])
                    - np.multiply(c, x[1]), 0)
            ]
        elif self.ode_name == 'stiff':
            lam1, lam2 = self.param_list_num[0:2]
            return [
                ((lam1 + lam2) * x[0] + (lam1 - lam2) * x[1]) / 2,
                ((lam1 - lam2) * x[0] + (lam1 + lam2) * x[1]) / 2
            ]
        return None

    def ml_G_func(self, x, dx, t, rho):
        """
        returns the calculated function G(x,x´,t,rho)=0
        :param x: matrix
            space variables in shape (batchsize x x_dim)
        :param dx: matrix
            derivative of the space variables in shape (batchsize x x_dim)
        :param t: array
            time array of shape (batchsize x 1)
        :param rho: matrix
            parameter matrix of shape (batchsize x rho_dim)
        :return:
            G: matrix
                calculated function G=0
        """
        if self.ode_name == 'harmonicoscillator':
            m = self.param_list
            k = rho
            return tf.concat(
                [
                    x[:, 1:2] - dx[:, 0:1],
                    -(k * x[:, 0:1]) / m - dx[:, 1:2]
                ],
                axis=1
            )
        elif self.ode_name == 'reboundpendulum':
            sigma = x[:, 0:1]
            omega = x[:, 1:2]
            l, m, g = self.param_list
            return tf.concat(
                [
                    omega - dx[:, 0:1],
                    - (g * tf.math.sin(sigma)) / l + tf.experimental.numpy.heaviside(-sigma, 0.) *
                    tf.math.maximum(
                        - (l * rho[:, 0:1] * sigma)/m
                        - rho[:, 1:2] * omega, 0) - dx[:, 1:2]
                ],
                axis=1
            )
        elif self.ode_name == 'stiff':
            lam1 = rho[:, 0:1]
            lam2 = rho[:, 1:2]
            return tf.concat(
                [
                    ((lam1 + lam2) * x[:, 0:1] + (lam1 - lam2) * x[:, 1:2]) / 2 - dx[:, 0:1],
                    ((lam1 - lam2) * x[:, 0:1] + (lam1 + lam2) * x[:, 1:2]) / 2 - dx[:, 1:2]
                ],
                axis=1
            )
        return None

    def initialize_data(self, batchsize=None, t_c=None):
        """
        initializes several data matrices
        :param batchsize: int
            batch size
        :param t_c: float
            curriculum learning time
        """
        t_0 = self.t0
        t_f = self.tf
        if self.ode_name == 'harmonicoscillator':
            self.initial_data_ode = [       # x_0=0, v_0=-1
                0.,
                -1.,
            ]
            if batchsize is not None:
                if t_c is not None:
                    t_f = t_c
                self.initial_data = tf.concat([
                    tf.random.uniform((batchsize, 1), t_0, t_f, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), -1.0, 1.0, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), -1.0, 1.0, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), 0.5, 2., dtype=DTYPE)
                ], axis=1)
        elif self.ode_name == 'reboundpendulum':
            self.initial_data_ode = [       # \theta_0=1, \omega_0=0.2
                1.0,
                0.2,
            ]
            if batchsize is not None:
                if t_c is not None:
                    t_f = t_c
                self.initial_data = tf.concat([
                    tf.random.uniform((batchsize, 1), t_0, t_f, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), 0.0, 1.0, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), -0.2, 0.2, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), 2.0, 5.0, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), 0.0, 2.0, dtype=DTYPE)
                ], axis=1)
        elif self.ode_name == 'stiff':
            eps = 0.25
            c1, c2 = self.param_list_num[2:4]
            lam1, lam2 = self.param_list_num[0:2]
            self.initial_data_ode = [       # x_1,0=c1+c2, x_2,0=c1-c2
                c1 + c2,
                c1 - c2,
            ]
            if batchsize is not None:
                if t_c is not None:
                    t_f = t_c
                self.initial_data = tf.concat([
                    tf.random.uniform((batchsize, 1), t_0, t_f, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), (c1 + c2) - eps, (c1 + c2) + eps, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), (c1 - c2) - eps, (c1 - c2) + eps, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), lam1 - eps, lam1 + eps, dtype=DTYPE),
                    tf.random.uniform((batchsize, 1), lam2 - eps, lam2 + eps, dtype=DTYPE)
                ], axis=1)

    def construct_test_matrix(self, t):
        """
        constructs test matrix for plotting
        :param t: array
            time array
        :return:
            test_matrix: matrix
                matrix of shape (batchsize x (t_dim + x_dim + rho_dim))
        """
        time_dimension = len(t)
        if self.ode_name == 'harmonicoscillator':
            return np.concatenate(
                    [
                        [t],
                        [np.full(time_dimension, self.initial_data_ode[0], dtype=DTYPE)],
                        [np.full(time_dimension, self.initial_data_ode[1], dtype=DTYPE)],
                        [np.full(time_dimension, self.param_list_num[1:2], dtype=DTYPE)]
                    ], axis=0)
        elif self.ode_name == 'reboundpendulum':
            return np.concatenate(
                    [
                        [t],
                        [np.full(time_dimension, self.initial_data_ode[0], dtype=DTYPE)],
                        [np.full(time_dimension, self.initial_data_ode[1], dtype=DTYPE)],
                        [np.full(time_dimension, self.param_list_num[3:4], dtype=DTYPE)],
                        [np.full(time_dimension, self.param_list_num[4:5], dtype=DTYPE)]
                    ], axis=0)
        elif self.ode_name == 'stiff':
            lam1, lam2 = self.param_list_num[0:2]
            return np.concatenate(
                    [
                        [t],
                        [np.full(time_dimension, self.initial_data_ode[0], dtype=DTYPE)],
                        [np.full(time_dimension, self.initial_data_ode[1], dtype=DTYPE)],
                        [np.full(time_dimension, lam1, dtype=DTYPE)],
                        [np.full(time_dimension, lam2, dtype=DTYPE)]
                    ], axis=0)

    def solution(self, t):
        """
        calculates and returns exact solution, if defined
        :param t: float
            point of time
        :return:
            sol: array
                space variables of exact solution
        """
        if self.ode_name == 'harmonicoscillator':
            t_0 = self.t0
            return [
                    - np.sin(t - t_0),
                    - np.cos(t - t_0)
                ]
        elif self.ode_name == 'reboundpendulum':
            return None
        elif self.ode_name == 'stiff':
            lam1, lam2 = self.param_list_num[0:2]
            c1, c2 = self.param_list_num[2:4]
            return [
                c1 * np.exp(lam1 * t) + c2 * np.exp(lam2 * t),
                c1 * np.exp(lam1 * t) - c2 * np.exp(lam2 * t)
            ]

    def print_odes(self):
        """
        prints list of ODE names that can be used
        :return:
            print: function
                prints list of ODE names
        """
        return print(self.ode_list)
