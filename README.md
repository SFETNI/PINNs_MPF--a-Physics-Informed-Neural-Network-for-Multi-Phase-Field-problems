# PINNs-MPF
These benchmarks in the current git are related to the article: 

PINNs-MPF: A Physics-Informed Neural Network Framework for Multi-Phase-Field Simulation of Interface Dynamics

## Abstract
We present an application of Physics-Informed Neural Networks (PINNs) to handle multi-phase-field (MPF) simulations of microstructure evolution.
This is realized through a multi-variable time-series problem by using full discrete resolution. 
Within each time interval, space, time, and phases/grains were treated separately, constituting discrete subdomains. 
A multi-networking concept is implemented to subdivide the simulation domain into multiple batches, with each batch associated with an independent Neural Network (NN) trained to predict the solution. 
To ensure efficient interaction across different phases/grains and in the spatio-temporal-phasic subdomain, a Master NN handles efficient interaction among the multiple networks, as well as the transfer of learning in different directions. 
A set of systematic simulations with increasing complexity was performed, that benchmarks various critical aspects of MPF simulations, handling different geometries, types of interface dynamics and the evolution of an interfacial triple junction.
A comprehensive approach is adopted to specifically accord attention on the MPF problem to the interfacial regions, facilitating an automatic and dynamic meshing process, significantly simplifying the tuning of  hyper-parameters and serving as a fundamental key for addressing MPF problems using Machine Learning.
The proposed PINNs-MPF framework successfully reproduces benchmark tests with high fidelity and Mean Squared Error (MSE) loss values ranging from 10$^{-4}$ to 10$^{-6}$ compared to ground truth solutions. 



# link to arxiv 
https://arxiv.org/pdf/2407.02230v1

## Citation (if you find this study helpful)
@misc{elfetni2024pinnsmpfphysicsinformedneuralnetwork,
      title={PINNs-MPF: A Physics-Informed Neural Network Framework for Multi-Phase-Field Simulation of Interface Dynamics}, 
      author={Seifallah Elfetni and Reza Darvishi Kamachali},
      year={2024},
      eprint={2407.02230},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2407.02230}, 
}

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
Please review the supplementary material of the paper (animated video or PDF) to become better familiarized with the code content. Additionally, it is advisable to look into the Benchmark 2 Python scripts (main.py, PINN.py, and Post_process.py) for 2D simulations on the first try. These scripts contain detailed comments about the code sequence before the complexity increases in subsequent benchmarks.

 For optimal performance, it is recommended to use a system with specifications similar to the AMD Ryzen Threadripper PRO 5975WX 32-Cores, which should have a minimum of 32GB of RAM.

## Reproducibility with Code Ocean
To ensure reproducibility and ease of running this code, we recommend also refering to Code Ocean.
Code Ocean provides a controlled environment where you can execute the code with all necessary dependencies pre-configured.

### Why Use CodeOcean?
- **Reproducibility**: Code Ocean ensures that the code runs consistently, regardless of changes in Python packages or system configurations.
- **Version Control**: You can access the exact environment used to develop and test this code.
- **Documentation**: Detailed instructions and explanations are provided within the Code Ocean capsule.
- 
## Environment Configuration
For consistent environment configuration, please refer to our [Code Ocean capsule](https://codeocean.com/capsule/9138079/tree/v1) where you can replicate the exact setup used for this project.

## License Information
 You are free to use, modify, and distribute the code in accordance with the terms of the license of the loaded packages.

## Supplementary video:
![](https://github.com/SFETNI/PINNs_MPF/blob/Main/Supplementary/Intro_Framework.gif)


