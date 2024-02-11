\textbf{PINNs-MPF}

These benchmarks in the current git are related to the article: PINNs-MPF: A Physics-Informed Neural Network Framework for Multi-Phase-Field Simulation of Interface Dynamics

In this project, we present an application of Physical Informed Neural Networks (PINNs) to handle multi-phase-field (MPF) simulations of microstructure evolution. This is realized through a multi-variable time-series problem by using full discrete resolution. 

Within each time interval, space, time, and phases/grains were treated separately, constituting discrete subdomains. A multi-networking concept is implemented to subdivide the simulation domain into multiple batches, with each batch associated with an independent NN trained to predict the solution. A concept of training by blocks is introduced, encompassing hierarchical (pyramidal) spatial decomposition and parallelization. 

To ensure efficient interaction across different phases/grains and in the spatio-temporal-phasic subdomain, a master neural network handles efficient interaction among the multiple networks, as well as the transfer of learning in different directions. A set of systematic simulations with increasing complexity was performed, that benchmarks various critical aspects of MPF simulations, handling different geometries, types of interface dynamics, and the evolution of an interfacial triple junction.

A global approach is taken here to concentrate the MPF problem at the interfacial regions. This enabled an automatic and dynamic meshing process and significantly simplified the tune of the hyper-parameters. The proposed PINNs-MPF framework successfully reproduces benchmark tests with high fidelity and Mean Squared Error (MSE) loss values ranging from 10$^{-4}$ to 10$^{-6}$ compared to the exact solutions.
