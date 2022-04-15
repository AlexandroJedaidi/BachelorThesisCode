# BachelorThesisCode

## What this is

This repository implements a neural net, runge-kutta and BDF methods to solve a few ordinary differential equations. This code is 
based on [this]([https://alexandrojedaidi.github.io/BachelorLatex/](https://alexandrojedaidi.github.io/BachelorLatex/main.pdf)) 
bachelor thesis.

## Get it running

requirements (libraries): tensorflow, numpy, pandas, matplotlib

After installing the required libraries follow these steps:

1. clone this rpository
2. in [main.py] adjust configurations, such as the ode_name and if pretrained data should be used (set preset=true/false)
3. run [main.py]
4. check Output/ode_name folder for solutions/plots

:warning: ** if you´re setting the ode_name to "reboundpendulum" and are using pretrained data (preset=true) **: epochs has to be
set to 100.000 (epochs=100000) to fit the preset data
