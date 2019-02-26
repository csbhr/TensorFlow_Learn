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
    Weights = new_weights(weights_shape=filter_shape)
    Biases = new_biases(output_channel)


    '''
    tf.nn.conv2d(input, filter, strides, padding)函数：
        ① input是卷积层的输入
        ② filter是卷积核，卷积核的shape为 [filter_size, filter_size, input_channel, output_channel]
        ③ strides是步长，卷积核的四个维度的步长。
            四个维度分别为 [image-number, x-axis, y-axis, input-channel]
            其中第一个和最后一个的步长必须为 1
        ④ padding是边缘处理方式，可以取值为 "SAME", "VALID"`
            
    '''

    # 创建卷积层
    conv_layer = tf.nn.conv2d(input=input,
                              filter=Weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
    conv_layer = tf.nn.relu(conv_layer + Biases)

    return conv_layer, Weights, Biases


# 数据集
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
IMG_NUM, IMG_SIZE, LABEL_SIZE = mnist.train.images.shape[0], mnist.train.images.shape[1], mnist.train.labels.shape[1]
