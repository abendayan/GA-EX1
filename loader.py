import numpy as np
from mnist import MNIST
import os

def normalize(x):
    return x.astype(np.float)/255.0

def load_data(test_image_path, test_labels_path):
    mndata = MNIST('data/', return_type="numpy")
    mndata.gz = True
    train_image, train_labels = mndata.load_training()
    train_image = normalize(train_image)
    randomize = np.arange(len(train_image))
    np.random.shuffle(randomize)
    train_image = train_image[randomize]
    train_labels = train_labels[randomize]
    valid_image = train_image[:10000]
    valid_labels = train_labels[:10000]
    train_image = train_image[10000-1:-1]
    train_labels = train_labels[10000-1:-1]


    test_image, test_labels = mndata.load(os.path.join('data/', test_image_path), os.path.join('data/', test_labels_path))
    test_image = normalize(np.array(test_image))
    test_labels = np.array(test_labels)

    return train_image, train_labels, valid_image, valid_labels, test_image, test_labels
