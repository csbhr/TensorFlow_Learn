import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


'''
Graph：绘制计算图
    ① 在想要特殊标记的变量/操作上可以添加name属性
    ② 利用 with tf.name_scope(name) 可以将一系列变量/操作放在一起作为一个整体
    ③ 必须运行 writer = tf.summary.FileWriter("logs/", sess.graph) 才能写入log文件中
Histogram：利用直方图展示一些变量的变化
    ① 利用函数 tf.summary.histogram(name, values) 可以将变量记录下来
    ② 一般用来记录参数
Scalar：利用折线图战术一些变量的变化
    ① 利用函数 tf.summary.scalar(name, values) 可以将变量记录下来
    ② 一般用来记录 loss/accuracy 之类的变量
合并：
    ① 想要使用 Histogram、Scalar 就必须进行合并
    ② 利用 merged = tf.summary.merge_all() 得到合并操作
    ③ 在迭代过程中：
        利用 merge_result = sess.run(merged, feed_dict) 得到跟踪变量的结果
        利用 writer.add_summary(merge_result, global_step) 将跟踪到的结果写到log文件中
'''


def new_weights(weights_shape, name=None):
    # 为了打破对称性以及0梯度，加入了少量的噪声
    return tf.Variable(tf.truncated_normal(shape=weights_shape, stddev=0.1), name=name)


def new_biases(biases_shape, name=None):
    # 为了避免神经元结点恒为0，使用较小的正数来初始化偏置项
    return tf.Variable(tf.constant(0.05, shape=biases_shape), name=name)


def add_conv_layer(input, kernel_size, num_input_channel, num_output_channel, using_relu=True,
                   layer_name=None):
    filter_shape = [kernel_size, kernel_size, num_input_channel, num_output_channel]
    filter = new_weights(weights_shape=filter_shape, name=layer_name + "_filter")
    biases = new_biases(biases_shape=[num_output_channel], name=layer_name + "_biases")
    tf.summary.histogram(layer_name + "/filter", filter)
    tf.summary.histogram(layer_name + "/biases", biases)
    conv = tf.nn.conv2d(input=input,
                        filter=filter,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    if using_relu:
        conv = tf.nn.relu(conv + biases)
    return conv


def add_pool_layer(input):
    return tf.nn.max_pool(input,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def add_fc_layer(input, num_input, num_output, using_relu=True,
                 layer_name=None):
    weights = new_weights(weights_shape=[num_input, num_output], name=layer_name + "_weights")
    biases = new_biases(biases_shape=[num_output], name=layer_name + "_biases")
    tf.summary.histogram(layer_name + "/weights", weights)
    tf.summary.histogram(layer_name + "/biases", biases)
    fc = tf.add(tf.matmul(input, weights), biases)
    if using_relu:
        fc = tf.nn.relu(fc)
    return fc


# 数据集
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
IMG_NUM, IMG_FLAT_SIZE, LABEL_SIZE \
    = mnist.train.images.shape[0], mnist.train.images.shape[1], mnist.train.labels.shape[1]
IMG_SIZE = int(np.sqrt(IMG_FLAT_SIZE))
CHANNEL_NUM = 1

# 占位符
with tf.name_scope("inputs"):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, IMG_FLAT_SIZE], name="x_input")
    ys = tf.placeholder(dtype=tf.float32, shape=[None, LABEL_SIZE], name="y_input")

# 转换尺寸：2d
with tf.name_scope("transcrope_2d"):
    xs_image = tf.reshape(tensor=xs, shape=[-1, IMG_SIZE, IMG_SIZE, CHANNEL_NUM], name="x_image_2d")

# 第一层卷积和池化
with tf.name_scope('conv_1'):
    conv1 = add_conv_layer(input=xs_image,
                           kernel_size=5,
                           num_input_channel=1,
                           num_output_channel=16,
                           layer_name="conv1")
    pool1 = add_pool_layer(conv1)

# 第二层卷积和池化
with tf.name_scope('conv_2'):
    conv2 = add_conv_layer(input=pool1,
                           kernel_size=5,
                           num_input_channel=16,
                           num_output_channel=36,
                           layer_name="conv2")
    pool2 = add_pool_layer(conv2)

# 转换尺寸：平铺
with tf.name_scope('transcrope_flatten'):
    image_flatten = tf.reshape(tensor=pool2, shape=[-1, 7 * 7 * 36], name="image_flatten")

# 第一个全连接层
with tf.name_scope('fc_1'):
    fc1 = add_fc_layer(input=image_flatten,
                       num_input=7 * 7 * 36,
                       num_output=1024,
                       layer_name="fc1")

# 第二个全连接层
with tf.name_scope('fc_2'):
    fc2 = add_fc_layer(input=fc1,
                       num_input=1024,
                       num_output=10,
                       using_relu=False,
                       layer_name="fc2")

# dropout
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob)

# 输出层
with tf.name_scope('y_pred'):
    y_pred = tf.nn.softmax(fc2_drop)

# loss
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2_drop, labels=ys, name="cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name="loss")
    tf.summary.scalar("loss", loss)

# accuracy
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(ys, axis=1)), dtype=tf.float32),
                              name="accuracy")
    tf.summary.scalar("accuracy", accuracy)

# 优化器
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# 训练
batch_size = 64
step_num = 20000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    for i in range(step_num):
        x_data, y_data = mnist.train.next_batch(batch_size)
        feed_dict = {
            xs: x_data,
            ys: y_data,
            keep_prob: 0.5
        }
        sess.run(train_step, feed_dict=feed_dict)
        if i % 10 == 0:
            feed_train_dict = {
                xs: x_data,
                ys: y_data,
                keep_prob: 1.0
            }
            feed_val_dict = {
                xs: mnist.test.images,
                ys: mnist.test.labels,
                keep_prob: 1.0
            }
            train_loss = sess.run(loss, feed_dict=feed_train_dict)
            val_loss = sess.run(loss, feed_dict=feed_val_dict)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train_dict)
            val_accuracy = sess.run(accuracy, feed_dict=feed_val_dict)
            merge_result = sess.run(merged, feed_dict=feed_val_dict)
            writer.add_summary(merge_result, i)
            print("==== step {0}  train-loss:{1:.3f}\ttrain-accuracy:{2:.3f}\tval-loss:{3:.3f}\tval-accuracy:{4:.3f}"
                  .format(i,
                          train_loss,
                          train_accuracy,
                          val_loss,
                          val_accuracy))
