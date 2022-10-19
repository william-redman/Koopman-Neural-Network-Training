# Koopman Neural Network Training

From "A. S. Dogra and W. T. Redman, Optimizing Neural Networks via Koopman Operator Theory, *Advances in Neural Information Processings Systems* **33** (*NeurIPS* 2020)". This code implements Koopman training based on Node Koopman operators, which was what was used to obtain the results presented in the paper. 

## Requirements

The provided function makes use of only Matlab. 

## Example data
An example data set of 25 NNs (~720 MB) trained using Ada delta (the data made to use Fig. 1) can be downloaded at https://drive.google.com/file/d/1GTz0osiZttg1VAd0WQaTvaasOHq8btjL/view?usp=sharing. 

Having saved the example data to your computer, you can run HNN_mastere_example.m. This script calls the function NodeKoopmanTraining.m, which builds the Koopman operator and evolves the weights/biases forward. An error vs. weight/bias evolution plot (analogous to Fig. 1e) is produced. Various free parameters can be adjusted to get a feel for how Node Koopman training works. See the comments within the code and the paper for more details. 

## Questions 

If you have any questions regarding the codebase or the associated NeurIPS paper, don't hesitate to email wredman@ucsb.edu or adogra@nyu.edu
