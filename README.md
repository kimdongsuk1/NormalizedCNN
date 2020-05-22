# NormalizedCNN

 We propose Normalized Convolutional Neural Network(NCNN). NCNN is more fitted
to a convolutional operator than other nomralizaiton methods. The normalized process is similar
to a normalization methods, but NCNN is more adapative to sliced-inputs and corresponding the
convolutional kernel. Therefor NCNN can be targeted to micro-batch training. Normalizaing of NC is
conducted during convolutional process. In short, NC process is not usual normalization and can not
be realized in deep learning framework optimizing standard convolution process. Hence we named
this method ’Normalized Convolution’. As a result, NC process has universal property which means
NC can be applied to any AI tasks involving convolution neural layer . Since NC don’t need other
normalization layer, NCNN looks like convolutional version of Self Normalizing Network.(SNN).
Among micro-batch trainings, NCNN outperforms other batch-independent normalization methods.
NCNN archives these superiority by standardizing rows of im2col matrix of inputs, which theoretically
smooths the gradient of loss. The code need to manipulate standard convolution neural networks step
by step. The code is available : https://github.com/kimdongsuk1/ NormalizedCNN
