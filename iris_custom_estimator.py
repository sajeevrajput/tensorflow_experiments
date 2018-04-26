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
NO_EPOCHS_TRAIN = 10


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

feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth'
]


# 1. write one or more dataset importing functions
def input_func(file, shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]
        del parsed_line[-1]
        features = parsed_line
        features_dict = dict(zip(feature_names, features))
        return features_dict, label

    dataset = tf.data.TextLineDataset(filenames=file).skip(1).map(decode_csv)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()  # return format should be (dict(features), labels)


# 2. define feature columns
feature_columns = [tf.feature_column.numeric_column('SepalLength'),
                   tf.feature_column.numeric_column('SepalWidth'),
                   tf.feature_column.numeric_column('PetalLength'),
                   tf.feature_column.numeric_column('PetalWidth')]


# 3. write model function
def model_fn(features, labels, mode):
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)
    h2 = tf.layers.Dense(10,activation=tf.nn.relu)(h1)
    output_layer = tf.layers.Dense(3)(h2)

    # PREDICTION
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'class_predicted': tf.argmax(output_layer, axis=1)}
        return tf.estimator.EstimatorSpec(mode, predictions)

    labels = tf.squeeze(labels, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, output_layer)


# 4. implement training, evaluation and predictions


if __name__ == '__main__':
    next_batch = input_func(file=FILE_TRAIN_PATH, shuffle=False)
    with tf.Session() as sess:
        for n in range(NO_EPOCHS_TRAIN):
            try:
                print("Run-{0}:\t{1}".format(n, sess.run(model_fn(feature_columns, ))))
            except tf.errors.OutOfRangeError:
                print("End of input")
                break
