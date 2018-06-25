# Backpropagation & Genetical Algorithm

## The Dataset
The algorithm presented here run on the MNIST dataset download from http://yann.lecun.com/exdb/mnist/.
It should be in the data folder.

## Backpropagation

### Validation on an other dataset
To run on a test dataset (that is not part of the origin MNIST, but is "like" MNIST), put it in the data folder.
Run the backpropagation like this:

python nn.py -images t10k-images-idx3-ubyte -labels t10k-labels-idx1-ubyte

Where:
t10k-images-idx3-ubyte.gz is the image dataset in the data folder.
t10k-labels-idx1-ubyte.gz is the labels dataset in the data folder.

### Pre-Trained model
After training the model, the program save the weights in a pickle file with the name "model_from_nn.model".
To use a pre-trained model, run it like this:

python nn.py -model model_from_nn.model
