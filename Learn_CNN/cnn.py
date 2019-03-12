import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

### 实现cnn

'''
tf.truncated_normal(shape,
                    mean=0.0,
                    stddev=1.0,
                    dtype=dtypes.float32,
                    seed=None,
                    name=None)
    ① 产生正太分布随机数，均值和标准差自己设定
    ② 这个函数产生的随机数与均值的差距不会超过两倍的标准差

tf.random_normal(shape,
                 mean=0.0,
                 stddev=1.0,
                 dtype=dtypes.float32,
                 seed=None,
                 name=None)
    产生正太分布随机数，均值和标准差自己设定

tf.nn.conv2d(input, 
             filter, 
             strides, 
             padding, 
             use_cudnn_on_gpu=True, 
             data_format="NHWC", 
             dilations=[1, 1, 1, 1], 
             name=None)
    卷积函数
    ① input是卷积层的输入
    ② filter是卷积核，卷积核的shape为 [filter_height, filter_width, in_channels, out_channels]
    ③ strides是步长，卷积核的四个维度的步长。
        四个维度分别为 [image-number, x-axis, y-axis, input-channel]
        其中第一个和最后一个的步长必须为 1，即为 [1, x-axis, y-axis, 1]
    ④ padding是边缘处理方式，可以取值为 "SAME", "VALID"`
        输出尺寸计算：
        当 padding='SAME' 时：
            out_height = ceil(float(in_height) / float(strides[1]))
            out_width = ceil(float(in_width) / float(strides[2]))
        当  padding='VALID' 时：
            out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
            out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

tf.nn.conv2d_transpose(value,
                       filter,
                       output_shape,
                       strides,
                       padding="SAME",
                       data_format="NHWC",
                       name=None)
    反卷积函数
    ① input是反卷积层的输入，shape为[batch, height, width, in_channels]，典型的NHWC格式
    ② filter是卷积核，shape为 [filter_height, filter_width, output_channels, in_channels]
    ③ output_shape是输出的shape，shape为[batch, height, width, output_channels]，必须和filter中的output_channels保持一致
    ④ strides是步长，卷积核的四个维度的步长。
        四个维度分别为 [image-number, x-axis, y-axis, input-channel]
        其中第一个和最后一个的步长必须为 1，即为 [1, x-axis, y-axis, 1]
    ⑤ padding是边缘处理方式，可以取值为 "SAME", "VALID"`
        但输出尺寸是由output_shape控制的

tf.nn.max_pool(value, 
               ksize, 
               strides, 
               padding, 
               data_format="NHWC", 
               name=None)函数：
    ① value是池化层的输入
    ② ksize是池化窗口的大小，取一个四维向量
        一般情况为 [1, height, width, 1]，因为一般不会在batch和channels上做池化
    ③ strides是步长，决定了池化后图片尺寸的变化。
        一般形式为 [1, x-axis, y-axis, 1]
        当取值为 [1, 2, 2, 1]时，输出的图片的尺寸在width和height上各减少一半
    ④ padding是边缘处理方式，可以取值为 "SAME", "VALID"

tf.reshape(tensor, shape, name=None)函数：
    转换尺寸
    如果shape的某一维度的值为 "-1"，则转换后这个维度的尺寸是通过计算得到的

tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)函数：
    dropout处理，为了解决过拟合
'''


def new_weights(weights_shape):
    # 为了打破对称性以及0梯度，加入了少量的噪声
    return tf.Variable(tf.truncated_normal(shape=weights_shape, stddev=0.1))


def new_biases(biases_shape):
    # 为了避免神经元结点恒为0，使用较小的正数来初始化偏置项
    return tf.Variable(tf.constant(0.05, shape=biases_shape))


def add_conv_layer(input, kernel_size, num_input_channel, num_output_channel, using_relu=True):
    filter_shape = [kernel_size, kernel_size, num_input_channel, num_output_channel]
    filter = new_weights(weights_shape=filter_shape)
    biases = new_biases(biases_shape=[num_output_channel])
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


def add_fc_layer(input, num_input, num_output, using_relu=True):
    weights = new_weights(weights_shape=[num_input, num_output])
    biases = new_biases(biases_shape=[num_output])
    fc = tf.matmul(input, weights) + biases
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
xs = tf.placeholder(dtype=tf.float32, shape=[None, IMG_FLAT_SIZE])
ys = tf.placeholder(dtype=tf.float32, shape=[None, LABEL_SIZE])
keep_prob = tf.placeholder(dtype=tf.float32)

# 转换尺寸：2d
xs_image = tf.reshape(tensor=xs, shape=[-1, IMG_SIZE, IMG_SIZE, CHANNEL_NUM])

# 第一层卷积和池化
conv1 = add_conv_layer(input=xs_image,
                       kernel_size=5,
                       num_input_channel=1,
                       num_output_channel=16
                       )
pool1 = add_pool_layer(conv1)

# 第二层卷积和池化
conv2 = add_conv_layer(input=pool1,
                       kernel_size=5,
                       num_input_channel=16,
                       num_output_channel=36
                       )
pool2 = add_pool_layer(conv2)

# 转换尺寸：平铺
image_flatten = tf.reshape(tensor=pool2, shape=[-1, 7 * 7 * 36])

# 第一个全连接层
fc1 = add_fc_layer(input=image_flatten,
                   num_input=7 * 7 * 36,
                   num_output=1024)

# 第二个全连接层
fc2 = add_fc_layer(input=fc1,
                   num_input=1024,
                   num_output=10,
                   using_relu=False)

# dropout
fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob)

# 输出层
y_pred = tf.nn.softmax(fc2_drop)

# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2_drop, labels=ys)
loss = tf.reduce_mean(cross_entropy)

# accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(ys, axis=1)), dtype=tf.float32))

# 优化器
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

# 训练
batch_size = 64
step_num = 20000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
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
            print("==== step {0}  train-loss:{1:.3f}\ttrain-accuracy:{2:.3f}\tval-loss:{3:.3f}\tval-accuracy:{4:.3f}"
                  .format(i,
                          train_loss,
                          train_accuracy,
                          val_loss,
                          val_accuracy))
