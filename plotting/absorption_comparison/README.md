This script takes in 6 .dat files form BGW's absorption and outputs their comparisons for each individual direction and the average. These are placed on their own individual plots and one plot.

# BerkeleyGW Absorption Spectra Plotter

## Overview
This Python script generalizes the plotting of optical absorption spectra ($\varepsilon_2$) from **BerkeleyGW** `absorption.x` output files. 

It is designed to compare multiple datasets (e.g., **Pristine vs. Frozen Phonon** comparisons) across anisotropic directions ($b_1, b_2, b_3$). It automatically calculates the average absorption and generates comparison plots.

## Features
* **Generalized Comparisons:** Compare an unlimited number of datasets (Pristine, T-Dep, 300K, etc.) by simply adding them to the configuration list.
* **Automatic Averaging:** Computes the arithmetic mean of the three input directions.
* **Multi-Panel Output:** Generates a combined $2 \times 2$ figure containing plots for $b_1$, $b_2$, $b_3$, and the Average.
* **Individual Outputs:** Automatically saves separate `.png` files for each direction for easy insertion into presentations.
* **Auto-Coloring:** Automatically cycles through distinct colors (Black $\to$ Red $\to$ Blue...) for each dataset added.

## Prerequisites
You need Python 3 installed along with `numpy` and `matplotlib`.