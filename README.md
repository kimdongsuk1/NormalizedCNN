# NormalizedCNN

we propose Normalized Convolutional Neural Network(NCNN). 
NCNN is more adap-tive to a convolutional operator than other nomralizaiton layers. The normalized process is similarto a normalization layers, but NCNN is more adapative to sliced-inputs to convolutional kernel. 
As NCNN is more adpative and sliced-inputs, NCNN can be targetd to micro-batch training. 
NCNNdon’t need a normalization layers. Hence NCNN looks like convolutional version of Self NormalizingNetwork.(SNN).
Among micro-batch trainings, NCNN outperforms other batch-independent normal-ization methods. 
NCNN achives these superiority by standardizing columns of im2col of inputs, which theoretically smooths the gradient of loss. 
The code need to manipulate standard convolution neuralnetworks step by step. 

