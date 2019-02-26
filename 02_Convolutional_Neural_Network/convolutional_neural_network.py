import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def new_weights(weights_shape):
    '''
    tf.truncated_normal(shape,
                        mean=0.0,
                        stddev=1.0,)函数：
        ① 产生正太分布随机数，均值和标准差自己设定
        ② 这个函数产生的随机数与均值的差距不会超过两倍的标准差
    tf.random_normal(shape,
                    mean=0.0,
                    stddev=1.0,)函数：
        产生正太分布随机数，均值和标准差自己设定
    '''

    return tf.Variable(tf.truncated_normal(shape=weights_shape))


def new_biases(biases_length):
    # 为了避免神经元结点恒为0，使用较小的正数来初始化偏置项
    return tf.Variable(tf.constant(0.05, shape=[biases_length]))


def add_conv_layer(input,  # 输入
                   input_channel,  # 输入通道
                   output_channel,  # 输出通道
                   filter_size  # 滤波器的宽和高
                   ):
    # 卷积核的shape
    filter_shape = [filter_size, filter_size, input_channel, output_channel]

    # 创建卷积核（权重）、偏置
    weights = new_weights(weights_shape=filter_shape)
    biases = new_biases(output_channel)

    '''
    tf.nn.conv2d(input, filter, strides, padding)函数：
        ① input是卷积层的输入
        ② filter是卷积核，卷积核的shape为 [filter_size, filter_size, input_channel, output_channel]
        ③ strides是步长，卷积核的四个维度的步长。
            四个维度分别为 [image-number, x-axis, y-axis, input-channel]
            其中第一个和最后一个的步长必须为 1，即为 [1, x-axis, y-axis, 1]
        ④ padding是边缘处理方式，可以取值为 "SAME", "VALID"`
            其中取值为"SAME"时输入和输出的图片大小相同
    '''

    # 创建卷积层
    conv_layer = tf.nn.conv2d(input=input,
                              filter=weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
    conv_layer = tf.nn.relu(conv_layer + biases)

    return conv_layer, weights, biases


def add_pool_layer(input):
    '''
    tf.nn.max_pool(value, ksize, strides, padding)函数：
        ① value是池化层的输入
        ② ksize是池化窗口的大小，取一个四维向量
            一般情况为 [1, height, width, 1]，因为一般不会在batch和channels上做池化
        ③ strides是步长，决定了池化后图片尺寸的变化。
            一般形式为 [1, x-axis, y-axis, 1]
            当取值为 [1, 2, 2, 1]时，输出的图片的尺寸在width和height上各减少一半
        ④ padding是边缘处理方式，可以取值为 "SAME", "VALID"`

    '''

    return tf.nn.max_pool(value=input,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def add_fc_layer(input,
                 num_input,
                 num_output):
    weights = new_weights(weights_shape=[num_input, 1])
    biases = new_biases(biases_length=num_output)
    fc = tf.nn.relu(tf.matmul(input, weights) + biases)
    return fc, weights, biases


# 数据集
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
IMG_NUM, IMG_FLAT_SIZE, LABEL_SIZE \
    = mnist.train.images.shape[0], mnist.train.images.shape[1], mnist.train.labels.shape[1]
IMG_SIZE = int(np.sqrt(IMG_FLAT_SIZE))
CHANNEL_NUM = 1

# 占位符
xs = tf.placeholder(dtype=tf.float32,
                    shape=[None, IMG_FLAT_SIZE])
ys = tf.placeholder(dtype=tf.float32,
                    shape=[None, LABEL_SIZE])

'''
tf.reshape(tensor, shape)函数：
    转换尺寸
    如果shape的某一维度的值为 "-1"，则转换后这个维度的尺寸不变
'''

# 转换尺寸：2d
xs_image = tf.reshape(tensor=xs,
                      shape=[-1, IMG_SIZE, IMG_SIZE, CHANNEL_NUM])

# 第一层卷积和池化
conv1, conv1_weights, conv1_biases = add_conv_layer(input=xs_image,
                                                    input_channel=1,
                                                    output_channel=16,
                                                    filter_size=5
                                                    )
pool1 = add_pool_layer(conv1)

# 第二层卷积和池化
conv2, conv2_weights, conv2_biases = add_conv_layer(input=pool1,
                                                    input_channel=16,
                                                    output_channel=36,
                                                    filter_size=5
                                                    )
pool2 = add_pool_layer(conv2)

# 转换尺寸：平铺
image_flatten = tf.reshape(tensor=pool2,
                           shape=[-1, 1764])

# 第一个全连接层
fc1, fc1_weights, fc1_biases = add_fc_layer(input=image_flatten,
                                            num_input=1764,
                                            num_output=128)

# 第二个全连接层
fc2, fc2_weights, fc2_biases = add_fc_layer(input=fc1,
                                            num_input=128,
                                            num_output=10)

# 预测值
y_pred = tf.nn.softmax(fc2)

# loss函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2, labels=ys)
loss = tf.reduce_mean(cross_entropy)

# 优化器
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# 计算准确度
accuracy = tf.reduce_mean(tf.cast(x=tf.equal(tf.argmax(y_pred, 1), tf.argmax(ys, 1)),
                                  dtype=tf.float32))

# 训练
batch_size = 200
epoch_num = 10

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epoch_num):
        for j in range(int(mnist.train.num_examples / batch_size)):
            x_train_data, y_train_data = mnist.train.next_batch(batch_size)
            feed_train_dict = {
                xs: x_train_data,
                ys: y_train_data
            }
            sess.run(train_step, feed_dict=feed_train_dict)
            if j % 50 == 0:
                feed_test_dict = {
                    xs: mnist.test.images,
                    ys: mnist.test.labels
                }
                now_loss = sess.run(loss, feed_dict=feed_train_dict)
                now_accuracy = sess.run(accuracy, feed_dict=feed_test_dict)
                print("step ", i, " loss:", now_loss, " accuracy:", now_accuracy)
