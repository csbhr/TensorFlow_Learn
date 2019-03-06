import tensorflow as tf

'''
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
'''
x = tf.constant(2.0, dtype=tf.float32)
y = tf.constant(3.0, dtype=tf.float32)
z1 = tf.where(tf.equal(x, y), x, y)
z2 = tf.where(tf.not_equal(x, y), x, y)
z3 = tf.where(tf.greater(x, y), x, y)
z4 = tf.where(tf.less(x, y), x, y)
z5 = tf.where(tf.greater_equal(x, y), x, y)
z6 = tf.where(tf.less_equal(x, y), x, y)
with tf.Session() as sess:
    print(sess.run(z1))
    print(sess.run(z2))
    print(sess.run(z3))
    print(sess.run(z4))
    print(sess.run(z5))
    print(sess.run(z6))

'''
tf.unstack(value, num=None, axis=0, name="unstack")函数:
    把 value 在 axis 维度分解，分解成多个tensor

tf.stack(value=[x1, x2, ...], axis=0, name="stack")函数:
    把 x1, x2, ... 在 axis 维度合并

tf.tile(input, multiples, name=None)函数：
    ① 把 input 在 multiples 里指定的在各个维度上复制的次数进行重复
    ② multiples 必须是和 input 维度数相同的列表
'''
