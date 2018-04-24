""" Implementation of custom estimator for iris classifier """
from six.moves.urllib.request import urlopen
import os
import logging
import tensorflow as tf


LOG_PATH = './logs'
DOWNLOAD_LOCATION = './datasets1'
FILE_TRAIN_PATH = os.path.join(DOWNLOAD_LOCATION, 'iris_training.csv')
FILE_TEST_PATH = os.path.join(DOWNLOAD_LOCATION, 'iris_test.csv')
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"


# Log setup
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(                                    # filename=os.path.join(LOG_PATH, 'iris.log'),
                    level=logging.DEBUG,
                    format='%(levelname)-10s %(asctime)s %(message)s')


# Download dataset
def download(url, file):
    if not os.path.exists(DOWNLOAD_LOCATION):
        os.mkdir(DOWNLOAD_LOCATION)
        logging.info('Folder created')
    if not os.path.exists(file):
        data = urlopen(url).read()
        with open(file, 'wb') as f:
            f.write(data)
            logging.info('Downloaded file')


download(URL_TRAIN, FILE_TRAIN_PATH)
download(URL_TEST, FILE_TEST_PATH)


# 1. write one or more dataset importing functions
def input_func(file):
    dataset = tf.data.TextLineDataset(filenames=file).skip(1)
    dataset = tf.decode_csv(dataset, [[0.], [0.], [0.], [0.], [0]])
    label = dataset[-1]
    features = dataset
    print(dataset)

# 2. define feature columns
# 3. write model function
# 4. implement training, evaluation and predictions


if __name__ == '__main__':
    input_func(file=FILE_TRAIN_PATH)