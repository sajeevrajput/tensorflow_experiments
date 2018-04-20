import os
import six.moves.urllib.request as request
import tensorflow as tf
import pprint


DOWNLOAD_PATH = os.path.join('.', 'datasets')
FILEPATH_TRAIN = os.path.join(DOWNLOAD_PATH, 'iris_training.csv')
FILEPATH_TEST = os.path.join(DOWNLOAD_PATH, 'iris_test.csv')
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"


def download_data(file, url):
    if not os.path.exists(DOWNLOAD_PATH):
        print('Creating directory...')
        os.mkdir(DOWNLOAD_PATH)

    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, 'wb') as f:
            f.write(data)
        print('Downloaded data...{0}'.format(url))


download_data(FILEPATH_TRAIN, URL_TRAIN)
download_data(FILEPATH_TEST, URL_TEST)

feature_name = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth'
]


# define input_function
def input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        # return feature and label from each line
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]
        del parsed_line[-1]
        feature = parsed_line
        featrue_dict = dict(zip(feature_name, feature)), label
        return featrue_dict

    dataset = (tf.data.TextLineDataset(file_path).skip(1).map(decode_csv))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # repeat_count here refers to epochs
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_feature,  batch_labels = iterator.get_next()
    return batch_feature, batch_labels


# define feature_column
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_name]


# define model_function
# def model_func(features, labels, mode):
#     pass

# train, evaluate, predict

if __name__ == '__main__':

    # next_batch = input_fn(file_path=FILEPATH_TRAIN, perform_shuffle=False)
    iris_dnn = tf.estimator.DNNClassifier(hidden_units=[10,10],
                                          feature_columns=feature_columns,
                                          model_dir='./models',
                                          n_classes=3,
                                          optimizer='Adagrad')

    # # Following line throws error. input_fn expects function object and not return value
    # train_inp = input_fn(file_path=FILEPATH_TRAIN, perform_shuffle=True)
    # iris_dnn.train(input_fn=train_inp, steps=10)

    iris_dnn.train(input_fn=lambda: input_fn(FILEPATH_TRAIN,True))
    ev = iris_dnn.evaluate(input_fn=lambda: input_fn(FILEPATH_TRAIN,True))
    print(ev["loss"])