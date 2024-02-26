<<<<<<< HEAD
# PINNs-MPF
These benchmarks in the current git are related to the article: PINNs-MPF: A Physics-Informed Neural Network Framework for Multi-Phase-Field Simulation of Interface Dynamics
## Abstract
We present an application of Physical Informed Neural Networks (PINNs) to handle multi-phase-field (MPF) simulations of microstructure evolution.
This is realized through a multi-variable time-series problem by using full discrete resolution. 
Within each time interval, space, time, and phases/grains were treated separately, constituting discrete subdomains. 
A multi-networking concept is implemented to subdivide the simulation domain into multiple batches, with each batch associated with an independent Neural Network (NN) trained to predict the solution. 
To ensure efficient interaction across different phases/grains and in the spatio-temporal-phasic subdomain, a Master NN handles efficient interaction among the multiple networks, as well as the transfer of learning in different directions. 
A set of systematic simulations with increasing complexity was performed, that benchmarks various critical aspects of MPF simulations, handling different geometries, types of interface dynamics and the evolution of an interfacial triple junction.
A comprehensive approach is adopted to specifically accord attention on the MPF problem to the interfacial regions, facilitating an automatic and dynamic meshing process, significantly simplifying the tuning of  hyper-parameters and serving as a fundamental key for addressing MPF problems using Machine Learning.
The proposed PINNs-MPF framework successfully reproduces benchmark tests with high fidelity and Mean Squared Error (MSE) loss values ranging from 10$^{-4}$ to 10$^{-6}$ compared to ground truth solutions. 

## Technical Remarks
 1. The code requires Python 3.x to run. Ensure that you have the appropriate Python version installed.
 2. The following packages need to be installed:
    - TensorFlow: Deep Learning framework
    - NumPy: Numerical computing library
    - SciPy: Scientific computing library
    - Matplotlib: Plotting library
    - pyDOE: Latin Hypercube Sampling (can be installed using pip)
 3. Ensure that you have the necessary permissions to install packages via pip.

## Code content
Please review the supplementary material of the paper (animated video or PDF) to become better familiarized with the code content. Additionally, it is advisable to look into the Benchmark 2 Python scripts (main.py, PINN.py, and Post_process.py) for 2D simulations in a first way. These scripts contain detailed comments about the code sequence before the complexity increases in subsequent benchmarks.

 For optimal performance, it is recommended to use a system with specifications similar to the AMD Ryzen Threadripper PRO 5975WX 32-Cores, which should have a minimum of 32GB of RAM.

## License Information
 You are free to use, modify, and distribute the code in accordance with the terms of the license of the loaded packages.

![](https://github.com/SFETNI/PINNs_MPF/blob/Main/Supplementary/Intro_Framework.gif)


=======
### Files
- main.py: This file contains the main code implementation.
- The simulation could be dynamically monitored through the log file (output.log), where all details about the execution, such as time stepping, convergence of the different networks, minibatching, and different loss terms, are dynamically printed.

### Save of Weights/Learning
- The weights/model parameters are saved for the triple junction simulations only for the first and last intervals after convergence. Ensure that the necessary directories or storage locations are properly configured for saving the weights.
- The model is already pre-trained (please c.f. the pyramidal training configuration for the training from scratch of the model) ..it is recommended to use the pyramidal training to retrain the model it from scratch.

### Reference Results
- Reference results can be found in the "Ref_results" repository. Please refer to this repository for comparison and evaluation purposes.

### Notes
- The model uses by default a dynamic saving of results for each time interval (please ensure you have enough space).
- For the dynamic post-process, the function process_repository_files_discret_workers_Master requires an additional check step for 
the phase summation plot (ongoing). 


### Variants
This repository contains additional variants of the main simulation: 
- Benchmark_5_triple_junction_16NN_Only_Interfaces : only interfacial regions are considered (ongoing further developement) .. please c.f. the Supplementary Matrerial for more details. 
>>>>>>> 54ff1bf455d7d66ce72d8eaf68b8291411d52a98
