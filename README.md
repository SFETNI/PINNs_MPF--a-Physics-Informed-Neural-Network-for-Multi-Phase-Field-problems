# PINNs-MPF: 
These benchmarks in the current git are related to the article: PINNs-MPF: A Physics-Informed Neural Network Framework for Multi-Phase-Field Simulation of Interface Dynamics
## Abstract
In this project, we present an application of Physical Informed Neural Networks (PINNs) to handle multi-phase-field (MPF) simulations of microstructure evolution. This is realized through a multi-variable time-series problem by using full discrete resolution. 
Within each time interval, space, time, and phases/grains were treated separately, constituting discrete subdomains. A multi-networking concept is implemented to subdivide the simulation domain into multiple batches, with each batch associated with an independent NN trained to predict the solution. A concept of training by blocks is introduced, encompassing hierarchical (pyramidal) spatial decomposition and parallelization. 
To ensure efficient interaction across different phases/grains and in the spatio-temporal-phasic subdomain, a master neural network handles efficient interaction among the multiple networks, as well as the transfer of learning in different directions. A set of systematic simulations with increasing complexity was performed, that benchmarks various critical aspects of MPF simulations, handling different geometries, types of interface dynamics, and the evolution of an interfacial triple junction.
A global approach is taken here to concentrate the MPF problem at the interfacial regions. This enabled an automatic and dynamic meshing process and significantly simplified the tune of the hyper-parameters. The proposed PINNs-MPF framework successfully reproduces benchmark tests with high fidelity and Mean Squared Error (MSE) loss values ranging from 10$^{-4}$ to 10$^{-6}$ compared to the exact solutions.

## Technical Remarks:
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

## License Information:
 You are free to use, modify, and distribute the code in accordance with the terms of the license of the loaded packages.

![Alt Text]([https://ibb.co/dGVVXNy]  
