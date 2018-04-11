import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

no_epochs = 5000
lr = 0.001

# build inputs
housing = fetch_california_housing()
X_unscaled = housing.data
scaler = StandardScaler()
scaler.fit(X_unscaled)
x = scaler.transform(X_unscaled)

m, n = housing.data.shape
X = tf.constant(np.c_[np.ones((m, 1)), x], dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

theta = tf.Variable(initial_value=tf.random_uniform([n+1,1],-1,1), dtype=tf.float32, name='theta')

y_pred = tf.matmul(X, theta)
error = tf.subtract(y,y_pred)
mse = tf.reduce_mean(tf.square(error))

gradient = (2/m) * tf.matmul(tf.transpose(X), y_pred)
training_op = tf.assign(theta, tf.subtract(theta, lr * gradient))

_loss = []
_epoch = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(no_epochs):
        if _ % 100 == 0:
            print("loss at epoch-{0}: {1}".format(_, mse.eval()))
        training_op.eval()

        _loss.append(mse.eval())
        _epoch.append(_)

    print('\nbest theta: \n{0}'.format(theta.eval()))

plt.plot(_epoch,_loss)
plt.show()
