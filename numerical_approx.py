import numpy as np
import scipy as sc
from scipy import integrate
from time import time


class NUM:
    """
    A class to represent a numerical approximation

    Attributes:
        range: int
            max range of iterations
        initial_data: array
            initial conditions
        H: function
            Runge-Kutta instance
        BDF: function
            NDF instance
        tolerance: float
            relative tolerance for both H and BDF
        atol: float
            absolute tolerance for both H and BDF
        t0: float
            initial time
        t: float
            end time
        first_step: float/None
            Initial step size, of None, algorithm chooses itself
        max_step: float
            maximum allowed step size
        ode_name: string
            name of given ODE
        ode: instance of ODE.py class
            used for ODE specific data
    """

    def __init__(self, ode_name, time_interval, rtol, atol, first_step, max_step, ode):
        """
        initialzes numerical approximations
        :param ode_name: string
            ODE name
        :param time_interval: array
            array of shape [t0, tf] with t0 being initial time and tf end time
        :param rtol: float
            relative tolerance
        :param atol: float
            absolute tolerance
        :param first_step: float
            initial step size
        :param max_step: float
            max step size
        :param ode: instance of ODE.py class
            used for ODE specific data and functions
        """
        self.range = None
        self.initial_data = None
        self.H = None
        self.BDF = None
        self.tolerance = rtol
        self.atol = atol
        (self.t0, self.t) = time_interval
        self.first_step = first_step
        self.max_step = max_step
        self.ode_name = ode_name
        self.ode = ode
        self.reset(rtol, atol)

    def set_functions(self):
        """
        initializes functions and data for approximation
        """
        self.f = self.ode.numerical_f
        self.ode.initialize_data()
        self.initial_data = self.ode.initial_data_ode

    def init_model(self):
        """
        initializes approximation functions for Runge-kutta and BDF
        """
        self.H = sc.integrate.RK45(
            fun=self.f,
            t0=self.t0,
            y0=self.initial_data,
            t_bound=self.t,
            first_step=self.first_step,
            max_step=self.max_step,
            rtol=self.tolerance,
            atol=self.atol)

        self.BDF = sc.integrate.BDF(
            fun=self.f,
            t0=self.t0,
            y0=self.initial_data,
            t_bound=self.t,
            first_step=self.first_step,
            max_step=self.max_step,
            rtol=self.tolerance,
            atol=self.atol)

    def approx(self, max_range):
        """
        approximation for the ODE with ode_name
        :param max_range: int
            maximum range
        :return:
            RK_x1_values: array
                array of approximated x1 values for runge kutta
            RK_x2_values: array
                array of approximated x2 values for runge kutta
            RK_t_values: array
                array of time points for runge kutta
            BDF_x1_values: array
                array of approximated x1 values for BDF
            BDF_x2_values: array
                array of approximated x2 values for BDF
            BDF_t_values: array
                array of time points for BDF
        """
        self.range = max_range

        RK_x1_values = [self.initial_data[0]]
        RK_x2_values = [self.initial_data[1]]
        RK_t_values = [self.t0]
        BDF_x1_values = [self.initial_data[0]]
        BDF_x2_values = [self.initial_data[1]]
        BDF_t_values = [self.t0]
        t0 = time()
        for i in range(self.range):  # approximating with runge-kutta
            self.H.step()
            RK_t_values.append(self.H.t)
            RK_x1_values.append(self.H.y[0])
            RK_x2_values.append(self.H.y[1])
            if self.H.status == 'finished':
                break
        print("Runge-Kutta took " + str(time() - t0) + " seconds")
        t0 = time()
        for i in range(self.range):  # approximating with BDF
            self.BDF.step()
            BDF_t_values.append(self.BDF.t)
            BDF_x1_values.append(self.BDF.y[0])
            BDF_x2_values.append(self.BDF.y[1])
            if self.BDF.status == 'finished':
                break
        print("BDF took " + str(time() - t0) + " seconds")

        RK_x1_values = np.array(RK_x1_values, dtype='float32')
        RK_x2_values = np.array(RK_x2_values, dtype='float32')
        RK_t_values = np.array(RK_t_values, dtype='float32')
        BDF_x1_values = np.array(BDF_x1_values, dtype='float32')
        BDF_x2_values = np.array(BDF_x2_values, dtype='float32')
        BDF_t_values = np.array(BDF_t_values, dtype='float32')

        return [RK_x1_values, RK_x2_values, RK_t_values, BDF_x1_values, BDF_x2_values, BDF_t_values]

    def reset(self, rtol, atol):
        """
        resets the instance for another approximation
        :param rtol: float
            relative tolerance
        :param atol: float
            absolute tolerance
        """
        self.range = None
        self.initial_data = None
        self.H = None
        self.BDF = None
        self.tolerance = rtol
        self.atol = atol
