import tensorflow as tf

logits = tf.constant([[1.0, 2.0, 3.0]])
y = tf.nn.softmax(logits)
y_ = tf.constant([[0.0, 0.0, 1.0]])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
cross_entropy_2=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_)
cross_entropy_3=tf.reduce_mean(cross_entropy_2)

with tf.Session() as sess:
    print(sess.run(cross_entropy))
    print(sess.run(cross_entropy_2))
    print(sess.run(cross_entropy_3))
