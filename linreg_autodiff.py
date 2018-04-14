import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


no_epoch=5000
lr=0.001

housing = fetch_california_housing()
x = housing.data
y = housing.target

m, n = x.shape

x_with_bias = np.c_[np.ones((m,1)), x]
scaler = StandardScaler()
scaler.fit(x_with_bias)
X = scaler.transform(x_with_bias)

X_features = tf.constant(X, shape=X.shape, dtype=tf.float32)
y_labels = tf.constant(y.reshape(-1,1), dtype=tf.float32)
theta = tf.Variable(tf.random_uniform([n+1,1], -1, 1), dtype=tf.float32)

y_pred = tf.matmul(X_features,theta)
error = tf.subtract(y_labels,y_pred)
mse = tf.reduce_mean(tf.square(error))

# gradients = tf.gradients(mse,[theta])[0]
# train_op = tf.assign(theta, theta-lr*gradients)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(mse)

_loss=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(no_epoch):
        sess.run(train_op)
        print("Loss at epoch {0}: {1}\t theta: {2}".format(_, mse.eval(), theta.eval()))
        _loss.append(mse.eval())

plt.plot(range(no_epoch),_loss)
plt.show()
