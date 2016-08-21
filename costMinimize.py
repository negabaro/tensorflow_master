import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random_uniform([1], -10000., 100000.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W,X) - Y), X), ))
update = W.assign(descent)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


for step in xrange(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

print sess.run(hypothesis, feed_dict={X: 5})
print sess.run(hypothesis, feed_dict={X: 2.5})

