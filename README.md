# Koopman Neural Network Training

This repository is the official implementation of "Optimizing Neural Networks via Koopman Operator Theory" (https://arxiv.org/pdf/2006.02361.pdf). In particular, it implements Koopman training based on Node Koopman operators, which was what was used to obtain all the results presented in the paper. 

## Requirements

The provided function makes use of only Matlab. 

## Training

As described in the paper, Koopman training requires weight/bias evolution from training iteration $t_1$ to $t_2$. With that data, this function can be used to train the network $T$ time steps ahead. $T, t_1,$ and $t_2$ are all free parameters. 

## Evaluation

As described in the provdied function, weight/bias data for all the weights/biases in a layer is inputed 

