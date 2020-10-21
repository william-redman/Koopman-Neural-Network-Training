# Koopman Neural Network Training

Implementation of "A. S. Dogra and W. T. Redman, Optimizing Neural Networks via Koopman Operator Theory, *Advances in Neural Information Processings Systems* **33** (*NeurIPS* 2020)". In particular, this code implements Koopman training based on Node Koopman operators, which was what was used to obtain the results presented in the paper. 

## Requirements

The provided function makes use of only Matlab. 

## Training

As described in the paper, Koopman training requires weight/bias evolution from training iteration t_1 to t_2. With that data, this function can be used to train the network T time steps ahead. T, t_1, and t_2 are all free parameters. 

## Evaluation

As described in the provdied function, weight/bias data for all the weights/biases in a layer is inputed as a tensor D. The assumptions is that the weights/biases are stored in the first and second dimensions, whereas the third dimension is time. Additionally, it assumes that the dimension being "noded" over is the second one (as is standard convention). 

The input argument q determines whether the weights/biases going to each node should be further split up into finer chunks (i.e. whether "Quasi-node" Koopman operators are to be constructed). Setting q = 1 sets them all into their own chunks (and builds Single Weight Koopman operators) and setting q = size(D, 2) sets them all into 
a single chunk (and builds the standard Node Koopman operator).

Finally, the input argument iFlag takes a 1 if the entire predicted trajectory is desired (i.e. all values from t_2 + 1 to t_2 + T). This can be helpful for seeing how Koopman training compares to standard training algorithms (as done in the paper). For time comparisions though, this should be turned off (iFlag = 0).  
