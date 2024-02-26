### Files
main.py: This file contains the main code implementation.
The simulation could be dynamically monitored through the log file (output.log), where all details about the execution, such as time stepping, convergence of the different networks, minibatching, and different loss terms, are dynamically printed.

### Save of Weights/Learning
Weights are saved for the triple junction simulations only for the first and last intervals after convergence. Ensure that the necessary directories or storage locations are properly configured for saving the weights. The model is already pre-trained (please c.f. the pyramidal training configuration for the training from scratch of the model) ..it is recommended to use the pyramidal training to retrain the model it from scratch.

### Reference Results
Reference results can be found in the "Ref_results" repository. Please refer to this repository for comparison and evaluation purposes.

### Notes
The model uses by default a dynamic saving of results for each time interval (please ensure you have enough space).
For the dynamic post-process, the function process_repository_files_discret_workers_Master requires an additional check step for 
the phase summation plot (ongoing). 
