# Backpropagation & Genetical Algorithm

## The Dataset
The algorithm presented here run on the MNIST dataset download from http://yann.lecun.com/exdb/mnist/.
It should be in the data folder.

## Learning
For both the backpropagation model and the genetical model, after learning, the scripts save the best model in a file and save the prediction on the test.

### Backpropagation

To run:

    python nn.py

### Genetical algorithm

To run:

    python run_ga.py

#### Test Dataset
The test_nn.pred / test_ga.pred is generated by the test data (in the data folder) of MNIST.
The generate it with another data, first copy it in the data folder with names like:

images.gz

labels.gz

Important: it should be files fo type .gz.

To run:

    python nn.py -images images -labels labels

Or (for GA):
    python run_ga.py -images images -labels labels


#### Pre-Trained model
The weights are saved in the file "model_from_nn.model" / "model_from_ga.model".
To use the pre-trained model (it will only check the test data and create the test_nn.pred / test_ga.pred), run:

    python nn.py -model model_from_nn.model

Or (for GA):

    python run_ga.py -model model_from_nn.model
