import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 数据集
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
IMG_NUM, IMG_SIZE, LABEL_SIZE = mnist.train.images.shape[0], mnist.train.images.shape[1], mnist.train.labels.shape[1]

# 占位符
xs = tf.placeholder(tf.float32, [None, IMG_SIZE])
ys = tf.placeholder(tf.float32, [None, LABEL_SIZE])

# 参数
Weight = tf.Variable(tf.zeros([IMG_SIZE, LABEL_SIZE]))
Bias = tf.Variable(tf.zeros([LABEL_SIZE]))

# 预测值
logits = tf.matmul(xs, Weight) + Bias
y_pred = tf.nn.softmax(logits)

# 损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ys)
loss = tf.reduce_mean(cross_entropy)

# 优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 准确度
correct_pred = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练
step_num = 10000
batch_size = 200

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(step_num):
        x_train_data, y_train_data = mnist.train.next_batch(batch_size)
        feed_train_dict = {
            xs: x_train_data,
            ys: y_train_data
        }
        sess.run(train_step, feed_dict=feed_train_dict)
        if i % 10 == 0:
            feed_test_dict = {
                xs: mnist.test.images,
                ys: mnist.test.labels
            }
            now_loss = sess.run(loss, feed_dict=feed_train_dict)
            now_accuracy = sess.run(accuracy, feed_dict=feed_test_dict)
            print("step ", i, " loss:", now_loss, " accuracy:", now_accuracy)
