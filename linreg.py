
#comment
import tensorflow as tf

#generates a polynomial data set of order 3
def generate_data():
    x = np.linspace(1,8,100)
    y = x**3+4*x**2+5*x+np.random.random(100)*80-60
    print(x)
    return x,y

def display_data(x,y):
    x,y = generate_data()
    plt.figure(1)
    plt.scatter(x,y)
    plt.show()

x=tf.placeholder(tf.float32,shape=(None,),name="x")
y=tf.placeholder(tf.float32, shape=(None,),name="y")


weights = tf.Variable(5.0, name="weights")
biases  = tf.Variable(3.0, name="biases")

loss = tf.square(tf.subtract(y,tf.add(tf.multiply(weights,x),biases)))

optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    for _ in range(20):
        sess.run(optimizer,feed_dict={x:[-1,0,1,2], y:[-1,2,5,8]})
        print(weights.eval(),biases.eval())
    writer.close()
