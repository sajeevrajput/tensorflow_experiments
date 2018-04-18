import tensorflow as tf

ts1 = tf.random_uniform([10, 6])
ts2 = tf.random_uniform([10,2],maxval=100,dtype=tf.int32)
dataset1 = tf.data.Dataset.from_tensor_slices(tensors=ts1)
dataset2 = tf.data.Dataset.from_tensor_slices(tensors=(ts2, ts1))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset1.output_types, dataset1.output_shapes)
print(dataset2.output_types, dataset2.output_shapes)
print(dataset3.output_types, dataset3.output_shapes)
print(dataset2.map(lambda x,y: (x*2,y*2)))
