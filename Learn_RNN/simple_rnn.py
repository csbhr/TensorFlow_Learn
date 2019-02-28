import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

### 实现普通的RNN

'''
tf.unstack(value, num=None, axis=0, name="unstack")函数:
    把 value 在 axis 维度分解，分解成多个tensor
    
tf.stack(value=[x1, x2, ...], axis=0, name="stack")函数:
    把 x1, x2, ... 在 axis 维度合并
    
tf.tile(input, multiples, name=None)函数：
    ① 把 input 在 multiples 里指定的在各个维度上复制的次数进行重复
    ② multiples 必须是和 input 维度数相同的列表
'''


def new_weights(weights_shape, name=None):
    # 为了打破对称性以及0梯度，加入了少量的噪声
    return tf.Variable(tf.truncated_normal(shape=weights_shape, stddev=0.1), name=name)


def new_biases(biases_shape, name=None):
    # 为了避免神经元结点恒为0，使用较小的正数来初始化偏置项
    return tf.Variable(tf.constant(0.05, shape=biases_shape), name=name)


def new_state(state_shape, name=None):
    return tf.zeros(shape=state_shape, name=name)


def rnn_cell(rnn_input, upper_state, weights, biases, name=None):
    return tf.tanh(tf.matmul(tf.concat([rnn_input, upper_state], axis=1), weights) + biases, name=name)


def add_output_layer(input, num_input, num_output, layer_name=None):
    weights = new_weights(weights_shape=[num_input, num_output], name=layer_name + "_weights")
    biases = new_biases(biases_shape=[num_output], name=layer_name + "_biases")
    tf.summary.histogram(name=layer_name + "/Weights", values=weights)
    tf.summary.histogram(name=layer_name + "/Biases", values=biases)
    logits = tf.add(tf.matmul(input, weights), biases)
    output = tf.nn.softmax(logits)
    return logits, output


# 数据集
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

sample_num, img_flatten_size, label_size \
    = mnist.train.images.shape[0], mnist.train.images.shape[1], mnist.train.labels.shape[1]
img_size = int(np.sqrt(img_flatten_size))
channel_num = 1
state_size = 128  # cell状态向量的长度，同时也是隐藏层的节点数
step_num = img_size  # 时间点个数，此处设置为图片的行数，每行作为一个时间点

# 输入层
with tf.name_scope("inputs"):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, img_flatten_size], name="x_input")
    ys = tf.placeholder(dtype=tf.float32, shape=[None, label_size], name="y_input")

    # reshape ==> [batch, 时间点, 一个时间点输入]
    xs_image = tf.reshape(xs, shape=[-1, img_size, img_size])
    labels = tf.reshape(ys, shape=[-1, 1, label_size])
    labels = tf.tile(labels, multiples=[1, step_num, 1])

    # 解绑成一个个时间点
    x_inputs = tf.unstack(xs_image, axis=1)
    labels = tf.unstack(labels, axis=1)

# RNN
with tf.name_scope("RNN"):
    rnn_weights = new_weights(weights_shape=[img_size + state_size, state_size], name="RNN_Weights")
    rnn_biases = new_biases(biases_shape=[state_size], name="RNN_Biases")
    tf.summary.histogram(name="RNN/Weights", values=rnn_weights)
    tf.summary.histogram(name="RNN/Biases", values=rnn_biases)

    states_list = []
    logits_list = []
    y_pred_list = []
    # init_state = new_state([xs.shape[0], state_size], name="state_0")
    init_state = tf.placeholder(dtype=tf.float32, shape=[None, state_size], name="state_0")
    init_logits, init_y_pred = add_output_layer(input=init_state, num_input=state_size, num_output=label_size,
                                                layer_name="output_0")
    states_list.append(init_state)
    logits_list.append(init_logits)
    y_pred_list.append(init_y_pred)

    # 遍历时间点
    for i in range(step_num):
        x_input = x_inputs[i]
        now_state = rnn_cell(x_input, states_list[i], rnn_weights, rnn_biases, name="state_{}".format(i + 1))
        now_logits, now_y_pred = add_output_layer(input=now_state, num_input=state_size, num_output=label_size,
                                                  layer_name="output_{}".format(i + 1))
        states_list.append(now_state)
        logits_list.append(now_logits)
        y_pred_list.append(now_y_pred)

# loss
with tf.name_scope("loss"):
    losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label) for logits, label in
              zip(logits_list, labels)]
    loss = tf.reduce_mean(losses, name="loss")
    tf.summary.scalar(name="loss", tensor=loss)

# accuracy
with tf.name_scope("accuracy"):
    final_y_pred = tf.reduce_mean(y_pred_list, axis=0)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_y_pred, 1), tf.argmax(ys, 1)), dtype=tf.float32))
    tf.summary.scalar(name="accuracy", tensor=accuracy)

# 优化器
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# 训练
step_num = 20000
batch_size = 64

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("log_simple_rnn/", sess.graph)
    merged = tf.summary.merge_all()
    for i in range(step_num):
        x_data, y_data = mnist.train.next_batch(batch_size)
        feed_train_dict = {
            xs: x_data,
            ys: y_data,
            init_state: np.zeros(shape=[batch_size, state_size])
        }
        sess.run(train_step, feed_dict=feed_train_dict)
        if i % 10 == 0:
            feed_val_dict = {
                xs: mnist.test.images,
                ys: mnist.test.labels,
                init_state: np.zeros(shape=[mnist.test.images.shape[0], state_size])
            }
            train_loss = sess.run(loss, feed_dict=feed_train_dict)
            val_loss = sess.run(loss, feed_dict=feed_val_dict)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train_dict)
            val_accuracy = sess.run(accuracy, feed_dict=feed_val_dict)
            merged_result = sess.run(merged, feed_dict=feed_val_dict)
            writer.add_summary(merged_result, global_step=i)
            print("==== step {0}  train-loss:{1:.3f}\ttrain-accuracy:{2:.3f}\tval-loss:{3:.3f}\tval-accuracy:{4:.3f}"
                  .format(i,
                          train_loss,
                          train_accuracy,
                          val_loss,
                          val_accuracy))
