import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import count
import numpy as np
import numpy.version
import pandas as pd
import scipy.version
import tensorflow as tf
import numerical_approx as na
import neural_net as nn
from math import e
from ODE import ODE

DTYPE = "float32"

# -------------------------------------------- configurations----------------------------------------------------------


ode_name = 'stiff'  # ['harmonicoscillator', 'reboundpendulum', 'stiff']
preset = True
save_data = not preset

epochs = 100000

# -------------------------------------------- initial conditions-------------------------------------------------------

# time start and ending
t_0 = 0.
t_f = 10.
if ode_name == 'harmonicoscillator':
    t_f = 2 * np.pi

# tolerances for numeric library
rtol = 10e-7
atol = 10e-14
if ode_name == 'stiff':
    rtol = 10e-2
    atol = 10e-7
if ode_name == 'reboundpendulum':
    rtol = 10e-5
    atol = 10e-7

# batchsize and time axis for plotting
batchsize = 10000
t = np.linspace(t_0, t_f, batchsize, dtype=DTYPE)

# init ODE with given conditions
ode = ODE(ode_name=ode_name,t0=t_0, tf=t_f)
ode.initialize_data()

# test data for plotting
X = tf.constant(np.transpose(ode.construct_test_matrix(t)))

# ----------------------------------------init solution, if given ------------------------------------------------------

sol = ode.solution(t)

# ----------------------------------------init numerical models---------------------------------------------------------

num_approx = na.NUM(
    ode_name=ode_name,
    time_interval=[t_0, t_f],
    rtol=rtol,
    first_step=None,
    max_step=np.inf,
    atol=atol,
    ode=ode
)
num_approx.set_functions()
num_approx.init_model()
approximation = num_approx.approx(10000)

if ode_name == 'reboundpendulum':
    rtol = 10e-14
    atol = 10e-14
    num_approx.reset(rtol, atol)
    num_approx.set_functions()
    num_approx.init_model()
    approximation2 = num_approx.approx(10000)

# ----------------------------------------init neural net---------------------------------------------------------------

set_b = True
if ode_name == 'harmonicoscillator':
    set_b = False
neural_net = nn.NN(
    time_interval=[t_0, t_f],
    batchsize=batchsize,
    ode=ode,
    set_b=set_b
)
neural_net.set_functions()

iterations = np.linspace(0, epochs, num=epochs)

plot_path = "Output/" + ode_name + "/"
os.makedirs(plot_path, exist_ok=True)

# ----------------------------------------neural net training & plotting------------------------------------------------

def initialize_neural_net(load_pretrained, save_new_data, neural_net_model, layer_count, neurons_count, lr, index):
    """ initializes and trains the neural net or loads trained data
    :param load_pretrained: boolean
        determines, if pretrained data is loaded
    :param save_new_data: boolean
        determines, if new rained data is saved
    :param neural_net_model: neural_net instance
        the initialized neural net from neural_net.py
    :param layer_count: int
        amount of layers for the neural net
    :param neurons_count: int
        amount of neurons per layer for the neural net
    :param lr: float
        learning rate for the neural net
    :param index: string
        path extension for saving/loading right data
    :return:
        loss_array: array
            array of the loss function
        error_array: array/None
            array of the global error of the neural net, if exact solution is defined
        prediction: array
            array of the output for the neural net
        calc_time: float/None
            training time of the neural net
    """
    loss_array = None
    error_array = None
    prediction = None
    calc_time = None
    path = "Output/" + ode_name + "/" + index + "/"
    if load_pretrained is False:
        structure_array = np.full(layer_count, neurons_count)
        neural_net_model.init_model(neurons_per_layer=structure_array, epochs=epochs)
        loss_array, error_array, calc_time = \
            neural_net_model.optimize(learning_rate=lr, t=tf.constant((np.reshape(t, (-1, 1)))), X=X, sol=sol)
        prediction = X[:, 1:3] + neural_net_model.a(tf.constant((np.reshape(t, (-1, 1))))) * neural_net_model.model(X)
        if save_new_data is True:
            os.makedirs(path + "new_trained_data/", exist_ok=True)
            path = path + "new_trained_data/"
            np.save(path + "l-" + str(layer_count) + "-" + "n" + str(neurons_count) + "-loss", loss_array)
            np.save(path + "l-" + str(layer_count) + "-" + "n" + str(neurons_count) + "-error", error_array)
            np.save(path + "l-" + str(layer_count) + "-" + "n" + str(neurons_count) + "-predict", prediction)
            np.save(path + "l-" + str(layer_count) + "-" + "n" + str(neurons_count) + "-time", calc_time)
    else:
        loss_array = np.load(path + "l-" + str(layer_count) + "-n-" + str(neurons_count) + "-loss.npy")
        prediction = np.load(path + "l-" + str(layer_count) + "-n-" + str(neurons_count) + "-predict.npy")
        if os.path.exists(path + "l-" + str(layer_count) + "-n-" + str(neurons_count) + "-error.npy"):
            error_array = np.load(path + "l-" + str(layer_count) + "-n-" + str(neurons_count) + "-error.npy")
        if os.path.exists(path + "l-" + str(layer_count) + "-n-" + str(neurons_count) + "-time.npy"):
            calc_time = np.load(path + "l-" + str(layer_count) + "-n-" + str(neurons_count) + "-time.npy")

    return loss_array, error_array, prediction, calc_time

def plotting(loss_data, error_data, predict_data, system_name):
    """ handles different plotting cases for given ODE´s
    :param loss_data: array
        the loss array of the trained neural net
    :param error_data: array/None
        the global error array of the trained neural net, if exact solution is defined
    :param predict_data: array
        the output array of the neural net
    :param system_name: string
        the name of the ODE system
    """
    path = "Output/" + ode_name + "/Plots/"
    os.makedirs(path, exist_ok=True)
    if system_name == 'stiff':
        nn_1 = predict_data[:, 0:1]
        nn_2 = predict_data[:, 1:2]

        rk_norm = np.linalg.norm(np.subtract([approximation[0], approximation[1]], ode.solution(approximation[2]))
                                 , axis=0)

        bdf_norm = np.linalg.norm(np.subtract([approximation[3], approximation[4]], ode.solution(approximation[5]))
                                  , axis=0)

        plt.plot(nn_1, nn_2, label="ML-Approximation")
        plt.plot(approximation[0], approximation[1], label="RK-Approximation")
        plt.plot(approximation[3], approximation[4], label="BDF-Approximation")
        plt.plot(sol[0], sol[1], label="exakte Lösung")
        plt.xlabel('$x_{1}$')
        plt.ylabel('$x_{2}$')
        plt.legend(loc=3, prop={'size': 10})
        plt.title('Trajektorien')
        plt.savefig(path + 'trajectories' + '.png')

        plt.clf()

        avr_error = pd.DataFrame({'error': error_data}).rolling(1000).mean()
        plt.yscale('log', base=10)
        # plt.xscale('log', base=10)
        plt.plot(avr_error, label="ML-Fehler")
        plt.legend(loc=1, prop={'size': 10})
        plt.xlabel('Epochen')
        plt.ylabel('$\Vert \cdot \Vert_2$-Fehler')
        plt.title('ML-Fehler')
        plt.savefig(path + 'ML_error' + '.png')

        plt.clf()

        avr_loss = pd.DataFrame({'loss': loss_data}).rolling(1000).mean()
        plt.yscale('log', base=10)
        # plt.xscale('log', base=10)
        plt.plot(avr_loss, label="Kostenfunktion")
        plt.legend(loc=1, prop={'size': 10})
        plt.xlabel('Epochen')
        plt.ylabel(r'$\mathcal{C}$')
        plt.title('ML-Kostenfunktion')
        plt.savefig(path + 'ML_Loss' + '.png')

        plt.clf()

        plt.yscale('log', base=10)
        plt.plot(t, tf.norm(predict_data - tf.transpose(tf.constant(sol)), axis=1), label="ML")
        plt.plot(approximation[2], rk_norm, label="RK")
        plt.plot(approximation[5], bdf_norm, label="BDF")
        plt.legend(loc=1, prop={'size': 10})
        plt.xlabel('$t$')
        plt.ylabel('$\Vert \cdot \Vert_2$-Fehler')
        plt.title('Zeitabhängiger Fehler')
        plt.savefig(path + 'Error_in_time' + '.png')

    elif system_name == 'harmonicoscillator':
        i = 0
        for loss_d in loss_data:
            avr_loss = pd.DataFrame({'loss': loss_d}).rolling(12000).mean()
            plt.plot(avr_loss, label="Netzwerk " + str(i + 1))
            i += 1

        plt.yscale('log', base=10)
        # plt.xscale('log', base=10)
        plt.legend(loc=3, prop={'size': 10})
        plt.ylabel(r'$\mathcal{C}$')
        plt.xlabel('Epochen')
        plt.title('Kostenfunktion')
        plt.savefig(path + 'Loss_' + str(len(loss_data)) + '.png')

        plt.clf()
        i = 0
        for error_d in error_data:
            avr_error = pd.DataFrame({'error': error_d}).rolling(12000).mean()
            plt.plot(avr_error, label="Netzwerk " + str(i + 1))
            i += 1

        plt.yscale('log', base=10)
        # plt.xscale('log', base=10)
        plt.legend(loc=3, prop={'size': 10})
        plt.ylabel('$\Vert \cdot \Vert_2$-Fehler')
        plt.xlabel('Epochen')
        plt.title('Globaler Fehler')
        plt.savefig(path + 'error_' + str(len(loss_data)) + '.png')

        plt.clf()

        i = 0
        for predict_d in predict_data:
            plt.plot(t, tf.norm(predict_d - tf.transpose(tf.constant(sol)), axis=1), label="Netzwerk " + str(i + 1))
            i += 1

        rk_norm_harm = np.linalg.norm(np.subtract([approximation[0], approximation[1]], ode.solution(approximation[2]))
                                      , axis=0)
        plt.xlim([0.6, t_f])
        plt.yscale('log', base=10)
        plt.plot(approximation[2], rk_norm_harm, label="RK")
        plt.legend(loc=4, prop={'size': 10})
        plt.xlabel('$t$')
        plt.ylabel('$\Vert \cdot \Vert_2$-Fehler')
        plt.title('Zeitabhängiger Fehler')
        plt.savefig(path + 'Error_in_time_' + str(len(loss_data)) + '.png')

        plt.clf()

        i = 0
        for predict_d in predict_data:
            nn_1 = predict_d[:, 0:1]
            nn_2 = predict_d[:, 1:2]
            plt.plot(nn_1, nn_2, label="Netzwerk " + str(i + 1))
            i += 1

        plt.plot(approximation[0], approximation[1], label='RK')
        plt.plot(sol[0], sol[1], label='exakte Lösung')
        plt.legend(loc=2, prop={'size': 10})
        plt.ylabel('v')
        plt.xlabel('x')
        plt.title('Trajektorien')
        plt.savefig(path + 'trajectories' + str(len(loss_data)) + '.png')

        plt.clf()

        i = 0
        for predict_d in predict_data:
            nn_1 = predict_d[:, 0:1]
            nn_2 = predict_d[:, 1:2]
            plt.plot(t, nn_1, label="Netzwerk " + str(i + 1))
            i += 1

        plt.plot(approximation[2], approximation[0], label="RK")
        plt.plot(t, sol[0], label='exakte Lösung')
        plt.legend(loc=2, prop={'size': 10})
        plt.xlabel('$t$')
        plt.ylabel('x')
        plt.title('$x(t)$-Graph')
        plt.savefig(path + 'trajectories_in_time_' + str(len(loss_data)) + '.png')

        plt.clf()

    else:
        nn_1 = predict_data[:, 0:1]
        nn_2 = predict_data[:, 1:2]
        plt.plot(nn_1, nn_2, label="ML")
        plt.plot(approximation[0], approximation[1], label="RK")
        plt.plot(approximation2[0], approximation2[1], label="Referenzlösung")
        plt.legend(loc=2, prop={'size': 10})
        plt.title('Phasenraum des gedämpften Pendels')
        plt.xlabel(r'$\theta$')
        plt.ylabel('$\omega$')
        plt.savefig(path + 'trajectories' + '.png')

        plt.clf()

        plt.yscale('log', base=10)
        # plt.xscale('log', base=10)
        plt.plot(iterations, loss_data, label="ML Loss")
        plt.legend(loc=1, prop={'size': 10})
        plt.xlabel('Epochen')
        plt.ylabel(r'$\mathcal{C}$')
        plt.title('Kostenfunktion')
        plt.savefig(path + 'ML_Loss' + '.png')

        plt.clf()

        plt.plot(t, nn_1, label="ML")
        plt.plot(approximation[2], approximation[0], label="RK")
        plt.plot(approximation2[2], approximation2[0], label="Referenzlösung")
        plt.legend(loc=1, prop={'size': 10})
        plt.xlabel('$t$')
        plt.ylabel(r'$\theta$')
        plt.title(r'$\theta(t)-$Graph')
        plt.savefig(path + 'trajectories_in_time' + '.png')

        plt.clf()

        avr_loss = pd.DataFrame({'loss': loss_data}).rolling(12000).mean()
        plt.yscale('log', base=10)
        # plt.xscale('log', base=10)
        plt.plot(avr_loss, label="ML Loss")
        plt.legend(loc=1, prop={'size': 10})
        plt.xlabel('Epochen')
        plt.ylabel(r'$\mathcal{C}$')
        plt.title('Kostenfunktion')
        plt.savefig(path + 'avr_loss' + '.png')

    print("done plotting")


if ode_name == 'harmonicoscillator':    # training neural net for harmonic oscillator and plotting
    neurons_per_layer = [4, 8, 16, 32]
    layers = [4, 8, 16]
    learning_rate = 0.0001
    neuron_loss = []
    neuron_errors = []
    neuron_predictions = []
    for i in range(len(neurons_per_layer)):
        loss, error, predict, time = initialize_neural_net(preset, save_data, neural_net, 4, neurons_per_layer[i],
                                                           learning_rate, "var_neurons")
        neuron_loss.append(loss)
        neuron_errors.append(error)
        neuron_predictions.append(predict)

    plotting(neuron_loss, neuron_errors, neuron_predictions, ode_name)
    layers_loss = []
    layers_errors = []
    layers_predictions = []
    for i in range(len(layers)):
        loss, error, predict, time = initialize_neural_net(preset, save_data, neural_net, layers[i], 32, learning_rate,
                                                           "var_layers")
        layers_loss.append(loss)
        layers_errors.append(error)
        layers_predictions.append(predict)

    plotting(layers_loss, layers_errors, layers_predictions, ode_name)


elif ode_name == 'stiff':    # training neural net for stiff ODE and plotting
    n_count = 32
    l_count = 8
    learning_rate = 0.0001
    loss, error, predict, time = initialize_neural_net(preset, save_data, neural_net, l_count, n_count, learning_rate,
                                                       "")
    plotting(loss, error, predict, ode_name)


else:       # training neural net for rebound pendulum and plotting
    n_count = 128
    l_count = 4
    learning_rate = 0.001
    loss, error, predict, time = initialize_neural_net(preset, save_data, neural_net, l_count, n_count, learning_rate,
                                                       "")
    plotting(loss, error, predict, ode_name)
