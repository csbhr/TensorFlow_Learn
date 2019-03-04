import tensorflow as tf

'''

常用的优化方法：
参考：http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants

1. 梯度下降（GD）的变种：
    1.1 Batch gradient descent
        可以保证：凸优化可以下降到全局最优点，非凸优化可以下降到局部最优点
        每次更新都需要将整个数据集放入进行计算梯度，速度很慢
        不能随时在线添加新样本
    1.2 Stochastic gradient descent
        每次跟新参数只需要一个样本来计算梯度，速度快
        可以在线添加新样本
        在更新参数过程中会出现巨大的波动，这种波动从另一个角度来说，可以跳到更优的局部最优点
    1.3 Mini-batch gradient descent
        兼备了前两者的优点
        减少了更新参数过程中的波动
        可以被深度学习计算库优化进行高效的运算
    1.4 GD面临的挑战
        ① 较难选取合适的学习率
        ② 对模型中所有的参数使用了相同的学习率
        ③ 如何避免落入非凸优化的众多局部最优点
2. 梯度下降的优化：
    2.1 Momentum
        
        
    
        
        


TensorFlow 中比较常用的优化器类：
参考：https://blog.csdn.net/xierhacker/article/details/53174558

1. class tf.train.Optimizer
    ① 优化器（optimizers）类的基类。这个类定义了在训练模型的时候添加一个操作的API。
    ② 基本上不会直接使用这个类，但是会用到他的子类比如GradientDescentOptimizer、AdagradOptimizer、MomentumOptimizer 等等
    
2. class tf.train.GradientDescentOptimizer
    ① 初始化：optimizer = tf.train.GradientDescentOptimizer(learning_rate, 
                                                             use_locking=False, 
                                                             name="GradientDescent")
        learning_rate: A Tensor or a floating point value. 要使用的学习率 
        use_locking: 要是True的话，就对于更新操作（update operations.）使用锁 
        name: 名字，可选，默认是”GradientDescent”
    ② 这个类实现的是优化算法是：梯度下降算法（GD）

'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate, use_locking=False, name="GradientDescent")