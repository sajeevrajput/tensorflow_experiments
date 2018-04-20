import tensorflow as tf

ts1 = tf.random_uniform([10, 6])
ts2 = tf.random_uniform([10,2],maxval=100,dtype=tf.int32)
dataset1 = tf.data.Dataset.from_tensor_slices(tensors=ts1)
dataset2 = tf.data.Dataset.from_tensor_slices(tensors=(ts2, ts1))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
# print(dataset1.output_types, dataset1.output_shapes)
# print(dataset2.output_types, dataset2.output_shapes)
# print(dataset3.output_types, dataset3.output_shapes)
# print(dataset2.map(lambda x,y: (x*2,y*2)))
#
# tf.random_uniform()
# tf.read_file()
# tf.image.decode_image()
# tf.image.resize_images()
# tf.py_func()
# tf.data.Dataset.from_tensor_slices()
# tf.data.Dataset.range()
# tf.data.Dataset.zip()
# tf.data.Dataset.batch()
# tf.fill()
# tf.data.Dataset.padded_batch()
# tf.data.Dataset.repeat()
# tf.parse_single_example()



ds1 = tf.data.Dataset.range(100)
ds2 = tf.data.Dataset.range(0,-100, -1)

ds_zip = tf.data.Dataset.zip((ds1, ds2))
ds_batch = ds_zip.batch(5)

iterator = ds_batch.make_one_shot_iterator()
next_elem = iterator.get_next()

ds1_padded = ds1.map(lambda x: tf.fill([tf.cast(x,tf.int32)], x))
ds1_padded = ds1_padded.padded_batch(5, padded_shapes=[None])
iter_pm = ds1_padded.make_one_shot_iterator()
next_elem_pm = iter_pm.get_next()

with tf.Session() as sess:
    # print(sess.run(next_elem))
    # print(sess.run(next_elem))
    while True:
        try:
            print(sess.run(next_elem_pm))
        except tf.errors.OutOfRangeError:
            print('Reached End')
            break
