import tensorflow as tf

'''
常用的loss函数：
    一般分为三类：回归问题、分类问题、自定义loss函数
    参考：https://zhuanlan.zhihu.com/p/44216830

1、回归问题
    1.1 均方根误差（MSE）
        函数：tf.losses.mean_squared_error(labels, 
                                           predictions, 
                                           weights=1.0, 
                                           scope=None,
                                           loss_collection=ops.GraphKeys.LOSSES,
                                           reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
        优点：便于梯度下降，误差大时下降快，误差小时下降慢，有利于函数收敛。
        缺点：受明显偏离正常范围的离群样本的影响较大。
        例子：
            mse = tf.losses.mean_squared_error(y_true, y_pred)
    1.2 平均绝对误差（MAE）
        函数：absolute_difference(labels, 
                                  predictions, 
                                  weights=1.0, 
                                  scope=None,
                                  loss_collection=ops.GraphKeys.LOSSES,
                                  reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
        优点：克服了 MSE 的缺点，受偏离正常范围的离群样本影响较小。
        缺点：收敛速度比 MSE 慢，因为当误差大或小时其都保持同等速度下降，而且在某一点处还不可导，计算机求导比较困难。
        例子：
            maes = tf.losses.absolute_difference(y_true, y_pred)
            maes_loss = tf.reduce_sum(maes)
    1.3 Huber loss
        函数：huber_loss(labels, 
                         predictions, 
                         weights=1.0, 
                         delta=1.0, 
                         scope=None,
                         loss_collection=ops.GraphKeys.LOSSES,
                         reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
        集合 MSE 和 MAE 的优点，但是需要手动调超参数
        核心思想：检测真实值（y_true）和预测值（y_pred）之差的绝对值在超参数 δ 内时，使用 MSE 来计算 loss, 在 δ 外时使用类 MAE 计算 loss。
        例子：
            hubers = tf.losses.huber_loss(y_true, y_pred)
            hubers_loss = tf.reduce_sum(hubers)

2、分类问题：交叉熵
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

3、自定义loss函数
    标准的损失函数并不合适所有场景，有些实际的背景需要采用自己构造的损失函数，Tensorflow 也提供了丰富的基础函数供自行构建。
    TensorFlow 中的判断语句：
        tf.where(condition, x=None, y=None, name=None)
            当 condition 为 true 时返回 x，否则返回 y 
        tf.equal(x, y, name=None)
        tf.not_equal(x, y, name=None)
        tf.greater(x, y, name=None)
        tf.less(x, y, name=None)
        tf.greater_equal(x, y, name=None)
        tf.less_equal(x, y, name=None)
            判断等于、不等于、大于、小于、大于等于、小于等于，返回 bool 类型的 tensor
            当维度不一致时广播后比较
    例子：当预测值（y_pred）比真实值（y_true）大时，使用 (y_pred-y_true)*loss_more 作为 loss；
          反之，使用 (y_true-y_pred)*loss_less。
        loss = tf.reduce_sum(tf.where(tf.greater(y_pred, y_true), (y_pred-y_true)*loss_more,(y_true-y_pred)*loss_less))
'''
