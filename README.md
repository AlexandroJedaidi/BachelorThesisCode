# BachelorThesisCode

## What this is

This repository implements a neural net, runge-kutta and BDF methods to solve a few ordinary differential equations. This code is 
based on [this](https://alexandrojedaidi.github.io/BachelorLatex/main.pdf)
bachelor thesis.

## How it works

This program uses the libraries numpy and tensorflow to create an runge-kutta-method, an BDF-method and an neural net to solve one of three given ODE's. For the runge-kutta and BDF-method, scipy functions found [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html) and [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html) are used. The neural net gets constructed with basic tensorflow functions and the theory is based on the paper found [here](https://arxiv.org/pdf/2006.14372.pdf).

## Get it running

requirements (libraries): tensorflow, numpy, pandas, matplotlib

After installing the required libraries follow these steps:

1. clone this rpository
2. in [main.py] adjust configurations, such as the ode_name and if pretrained data should be used (set preset=true/false)
3. run [main.py]
4. check Output/ode_name folder for solutions/plots

:warning: __if you´re setting the ode_name to "reboundpendulum" and are using pretrained data (preset=true)__: epochs has to be
set to 100.000 (epochs=100000) to fit the preset data
