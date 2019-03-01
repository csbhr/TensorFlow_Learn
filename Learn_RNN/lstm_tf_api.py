import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

### 使用Tensorflow的API实现lstm


'''
RNN 的使用过程：
    第一步：获取 cell 对象
        如 LSTM cell，使用 cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, 
                                                              forget_bias=1.0,
                                                              state_is_tuple=True, 
                                                              activation=None, 
                                                              reuse=None, 
                                                              name=None)
            ① num_units 是输入的长度
            ② forget_bias 是初始化的forget门的偏置，默认是1.0，即不忘记上一个时间点的任何信息
            ③ state_is_tuple=True 则返回的state是一个2维的tuple，分别为`c_state`和`m_state`
               state_is_tuple=False 则把两个state在column维度合并
    第二步：初始化 state
        利用cell对象来获取初始化的state，初始化为0。
        使用 init_state = cell.zero_state(batch_size, dtype)
    第三步：计算 RNN 输出
        使用 cell_outputs, final_state = tf.nn.dynamic_rnn(cell, 
                                                          inputs, 
                                                          sequence_length=None, 
                                                          initial_state=None,
                                                          dtype=None, 
                                                          parallel_iterations=None, 
                                                          swap_memory=False,
                                                          time_major=False, 
                                                          scope=None)
            ① cell 是第一步获取的cell对象
            ② input 是RNN的输入
            ③ initial_state 是第二步获取的初始化state
'''


def new_weights(weights_shape, name=None):
    # 为了打破对称性以及0梯度，加入了少量的噪声
    return tf.Variable(tf.truncated_normal(shape=weights_shape, stddev=0.1), name=name)


def new_biases(biases_shape, name=None):
    # 为了避免神经元结点恒为0，使用较小的正数来初始化偏置项
    return tf.Variable(tf.constant(0.05, shape=biases_shape), name=name)


# 数据集
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

sample_num, img_flatten_size, label_size \
    = mnist.train.images.shape[0], mnist.train.images.shape[1], mnist.train.labels.shape[1]
img_size = int(np.sqrt(img_flatten_size))
channel_num = 1

step_num = img_size  # 时间点个数，此处设置为图片的行数，每行作为一个时间点
input_num = img_size
hidden_units_num = 128  # 隐藏层的节点数
classes_num = label_size
batch_size = 200

# 输入层
with tf.name_scope("inputs"):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, img_flatten_size], name="x_input")
    ys = tf.placeholder(dtype=tf.float32, shape=[None, classes_num], name="y_input")

    # reshape ==> [batch, step_num, input_num]
    x_inputs = tf.reshape(xs, shape=[-1, step_num, input_num])

# LSTM
with tf.name_scope("LSTM"):
    # 输入输出隐藏层参数
    weights = {
        "in": new_weights(weights_shape=[input_num, hidden_units_num], name="weights_input_hidden"),
        "out": new_weights(weights_shape=[hidden_units_num, classes_num], name="weights_output_hidden")
    }
    biases = {
        "in": new_biases(biases_shape=[hidden_units_num], name="biases_input_hidden"),
        "out": new_biases(biases_shape=[classes_num], name="biases_output_hidden")
    }
    tf.summary.histogram(name="LSTM/in_hidden/weight", values=weights["in"])
    tf.summary.histogram(name="LSTM/in_hidden/biases", values=biases["in"])
    tf.summary.histogram(name="LSTM/out_hidden/weight", values=weights["out"])
    tf.summary.histogram(name="LSTM/out_hidden/biases", values=biases["out"])

    # 输入隐藏层
    with tf.name_scope("input_hidden_layer"):
        # reshape ==> [batch*step_num, input_num]
        x_inputs = tf.reshape(x_inputs, shape=[-1, input_num])
        x_inputs = tf.matmul(x_inputs, weights["in"]) + biases["in"]
        # reshape ==> [batch, step_num, hidden_units_num]
        x_inputs = tf.reshape(x_inputs, shape=[-1, step_num, hidden_units_num])

    # cell
    with tf.name_scope("cell"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units_num, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        cell_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x_inputs, initial_state=init_state,
                                                      time_major=False)

        # 获取cell输出的两种方式：
        # ==> ①
        cell_result = final_state[1]
        # ==> ②
        # cell_outputs = tf.transpose(cell_outputs, [1, 0, 2])
        # cell_result = cell_outputs[-1]

    # 输出隐藏层
    with tf.name_scope("output_hidden_layer"):
        logits = tf.matmul(cell_result, weights["out"]) + biases["out"]
        y_pred = tf.nn.softmax(logits=logits)

# loss
with tf.name_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ys)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar(name="loss", tensor=loss)

# accuracy
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(ys, 1)), dtype=tf.float32))
    tf.summary.scalar(name="accuracy", tensor=accuracy)

# 优化器
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# 训练
step_num = 20000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("log_lstm_tf_api/", sess.graph)
    merged = tf.summary.merge_all()
    for i in range(step_num):
        x_data, y_data = mnist.train.next_batch(batch_size)
        feed_train_dict = {
            xs: x_data,
            ys: y_data,
        }
        sess.run(train_step, feed_dict=feed_train_dict)
        if i % 10 == 0:
            x_val_data, y_val_data = mnist.test.next_batch(batch_size)
            feed_val_dict = {
                xs: x_val_data,
                ys: y_val_data,
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
