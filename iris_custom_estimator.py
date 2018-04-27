""" Implementation of custom estimator for iris classifier """
from six.moves.urllib.request import urlopen
import os
import logging
import tensorflow as tf


LOG_PATH = './logs'
DOWNLOAD_LOCATION = './datasets'
FILE_TRAIN_PATH = os.path.join(DOWNLOAD_LOCATION, 'iris_training.csv')
FILE_TEST_PATH = os.path.join(DOWNLOAD_LOCATION, 'iris_test.csv')
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"
NO_EPOCHS_TRAIN = 500
LEARNING_RATE = 0.05


# Log setup
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
# logging.basicConfig(                                    # filename=os.path.join(LOG_PATH, 'iris.log'),
#                     level=logging.INFO,
#                     format='%(levelname)-10s %(asctime)s %(message)s')
tf.logging.set_verbosity(tf.logging.INFO)


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


# 3. write model function
def model_fn(features, labels, mode):

    # define feature columns
    feature_columns = [tf.feature_column.numeric_column(feature_names[0]),
                       tf.feature_column.numeric_column(feature_names[1]),
                       tf.feature_column.numeric_column(feature_names[2]),
                       tf.feature_column.numeric_column(feature_names[3])]

    # DESIGN NETWORK
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)
    h2 = tf.layers.Dense(10, activation=tf.nn.relu)(h1)
    output_layer = tf.layers.Dense(3)(h2)

    # PREDICTION
    predictions = {'class_predicted': tf.argmax(output_layer, axis=1)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # EVALUATION
    # labels = tf.squeeze(labels, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, output_layer)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['class_predicted'])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'acc': accuracy})

    # TRAINING
    train_op = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss,
                                                                               global_step=tf.train.get_global_step())
    tf.summary.scalar('ACC', accuracy[1])
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    iris_custom_estimator = tf.estimator.Estimator(model_fn=model_fn,
                                                   model_dir='./models'
                                                   # ,config=tf.estimator.RunConfig(model_dir='./models', save_summary_steps=50,save_checkpoints_steps=50)
    )
    # 500 epochs = 500 * 120/32 = 1875 batches
    iris_custom_estimator.train(input_fn=lambda: input_func(FILE_TRAIN_PATH,
                                                            shuffle=True,
                                                            repeat_count=100))
    predictions = iris_custom_estimator.evaluate(input_fn=lambda: input_func(FILE_TEST_PATH,
                                                                             shuffle=False,
                                                                             repeat_count=4),steps=20)
