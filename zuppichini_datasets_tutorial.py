"""
Notes from https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
by Francesco Zuppichini
"""
import numpy as np
import tensorflow as tf

# ***********************************

# 1. IMPORT DATA

# *************************************

# from numpy arrays
x1 = np.random.sample((100, 2))
y1 = np.random.sample((100,1))
dataset1 = tf.data.Dataset.from_tensor_slices((x1,y1))

# from placeholders
x2 = tf.placeholder(dtype=tf.float32, shape=[None,2])
dataset2=tf.data.Dataset.from_tensor_slices(x2)
# print(dataset2)

# from tensors
x3 = tf.random_uniform([100,3])
dataset3 = tf.data.Dataset.from_tensors(x3)
# print(dataset3)

# using generators for variable length data
x4 = np.array(([1], [2,3], [4,5,6]))
def _generator():
    for elem in x4:
        yield elem


dataset4 = tf.data.Dataset.from_generator(_generator, output_types=tf.float32)
# print(dataset4)


# ************************************

# 2. CREATE AN ITERATOR

# ************************************

iter1 = dataset1.make_one_shot_iterator()
# features, labels = iter.get_next()
#
# with tf.Session() as sess:
#     for _ in range(100):  # 100 is the no of examples
#         print(sess.run([features, labels]))

iter2 = dataset2.make_initializable_iterator()  # The returned iterator will be in an uninitialized state,
                                                #  and you must run the `iterator.initializer` operation before using it
# elem = iter2.get_next()
#
# with tf.Session() as sess:
#     sess.run(iter2.initializer, feed_dict={x2: np.random.sample((10, 2))})
#     while True:
#         try:
#             print(sess.run(elem))
#         except tf.errors.OutOfRangeError:
#             break

train_data3 = (np.random.sample((50,2)), np.random.sample((50,1)))
test_data3 = (np.random.sample((10,2)), np.random.sample((10,1)))

feat, labels = tf.placeholder(dtype=tf.float32, shape=[None, 2]), tf.placeholder(dtype=tf.float32, shape=[None, 1])
data3 = tf.data.Dataset.from_tensor_slices((feat, labels))
iter3 = data3.make_initializable_iterator()

# _features, _labels = iter3.get_next()

with tf.Session() as sess3:
    print("TRAIN DATA")
    sess3.run(iter3.initializer, feed_dict={feat: train_data3[0], labels: train_data3[1]})
    print(sess3.run(iter3.get_next()))
    print(sess3.run(iter3.get_next()))

    print("TEST DATA")
    sess3.run(iter3.initializer, feed_dict={feat: test_data3[0], labels: test_data3[1]})
    print(sess3.run(iter3.get_next()))
    print(sess3.run(iter3.get_next()))