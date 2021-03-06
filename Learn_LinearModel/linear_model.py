import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

### 实现简单线性模型

'''
不同交叉熵的区别：
f1：tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,
                                            labels=None, 
                                            logits=None,
                                            dim=-1, 
                                            name=None)
    ① 传入的logits和labels都必须是one-hot编码的
    ② logits是没有经过softmax函数处理过的，函数内部会进行softmax运算
    ③ 梯度在反向传播中只作用于logits，不会作用于lables
    ④ 将来版本将会被弃用，因为有些模型labels也是模型生成，需要梯度的反向传播
f2：tf.nn.softmax_cross_entropy_with_logits_v2(_sentinel=None,
                                               labels=None, 
                                               logits=None,
                                               dim=-1, 
                                               name=None)
    ① 与f1的区别在于，梯度在反向传播中同时会作用于logits和lables
    ② 将来版本，会完全替代f1
f3：tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None,
                                                   labels=None, 
                                                   logits=None,
                                                   dim=-1, 
                                                   name=None)
    ① 与f1的区别在于，lables不需要one-hot编码  
'''

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
    sess.run(tf.global_variables_initializer())
    for i in range(step_num):
        x_data, y_data = mnist.train.next_batch(batch_size)
        feed_dict = {
            xs: x_data,
            ys: y_data
        }
        sess.run(train_step, feed_dict=feed_dict)
        if i % 10 == 0:
            feed_train_dict = {
                xs: x_data,
                ys: y_data
            }
            feed_val_dict = {
                xs: mnist.test.images,
                ys: mnist.test.labels
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
