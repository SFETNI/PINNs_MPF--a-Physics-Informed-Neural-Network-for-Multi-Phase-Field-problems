### Files
main.py: This file contains the main code implementation. The simulation could be dynamically monitored through the log file (output.log), where all details about the execution, such as time stepping, convergence of the different networks, minibatching, and different loss terms, are dynamically printed. The main body PINNs functions are in PINN.py. The MPF resolution and backgrounds are presented in theory.ipynb. pre_post.py includes functions for preparing training data and pre-post processing of results.

### Save of Weights/Learning
Weights are saved at each time interval after convergence. Ensure that the necessary directories or storage locations are properly configured for saving the weights. The model
is already pre-trained .. you can set the option "Transfer_Learning" to False to retrain it from scratch 

### Reference Results
Reference results can be found in the "Ref_results" repository. Please refer to this repository for comparison and evaluation purposes.
