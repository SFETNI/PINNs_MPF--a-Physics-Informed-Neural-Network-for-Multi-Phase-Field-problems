### Files
main_triple_junction_pinns_to_Master.py: This file contains the main code implementation.

### General information  
This is a demonstration example for the feasibilty of the Multiple to Single transfer of the Learning, that could be explored for similar
or different physical problems.

### Details
 Knowledge accumulated over the entire domain can be redistributed to all NNs, enabling the exploration of similar physical problems. The learning is transferred between levels using a pyramidal approach.
In the technical demonstration, the base of the pyramid is set to 64, representing 64 subdivisions. The subdivision is then progressively increased until reaching 4 NNs. At this level, once the 4NNs are trained, one selected NN with the minimum loss is chosen as a candidate to handle the entire domain and becomes a Master \textit{PINN}. This approach allows for continuous coverage of the entire domain.
However, as previously reported in the Discussion section of the related manuscript, our experiments highlighted difficulties in optimizing weights, particularly using the LBFGS optimizer, due to the complexity of the problem for a single NN. Handling the upper level using 4 NNs, interacting at the boundaries, was found to be very efficient.

