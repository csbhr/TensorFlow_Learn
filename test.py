import tensorflow as tf

x_1 = tf.constant(1, shape=[10, 5])
x_2 = tf.constant(2, shape=[10, 5])
x_3 = tf.constant(3, shape=[10, 5])
y = [x_1, x_2, x_3]
z = tf.reduce_sum(y, axis=0)

with tf.Session() as sess:
    print("z", sess.run(z))
