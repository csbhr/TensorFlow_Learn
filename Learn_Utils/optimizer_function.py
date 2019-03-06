import tensorflow as tf

'''

常用的优化方法：
参考：http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants
参考：https://blog.csdn.net/aliceyangxi1987/article/details/73210204

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
        加入动量项，可以加速在梯度不变的维度的参数变化，阻尼在梯度变化维度的参数变化
        可以加速SGD的收敛，阻尼震荡
        一般动量项因子取值为0.9左右
    2.2 Nesterov accelerated gradient
        与Momentum不同的地方在于：Momentum计算梯度时使用的是当前位置，计算当前位置的梯度后再加上动量项；
        而Nesterov accelerated gradient是先使用前一步的梯度走到未来一步，计算未来位置的梯度，再加上动量项进行修正
        这种算法可以避免更新走的过快
    2.3 Adagrad
        这个算法就可以对低频的参数做较大的更新，对高频的做较小的更新
        Adagrad可以自己调节学习率，减少学习率的手动调节
        缺点：随着分母的不断累积，学习率会越来越小，无线接近于0
    2.4 Adadelta
        这个算法是对Adagrad的改进，Adagrad的分母会累加之前所有的梯度平方；
        而Adadelta的分母只累加固定大小的项，仅仅是计算对应的平均值，学习率不会越来越小
        学习率的更新法则与学习率初始值没有关系，不需要设置学习率默认值
    2.5 Adam
        Adam除了像Adadelta一样存储了过去梯度的平方的指数衰减平均值，也像 momentum 一样保持了过去梯度的指数衰减平均值
        实践表明，Adam 比其他适应性学习方法效果要好

TensorFlow 中比较常用的优化器类：
参考：https://blog.csdn.net/xierhacker/article/details/53174558

1. class tf.train.Optimizer
    ① 优化器（optimizers）类的基类。这个类定义了在训练模型的时候添加一个操作的API。
    ② 基本上不会直接使用这个类，但是会用到他的子类比如GradientDescentOptimizer、AdagradOptimizer、MomentumOptimizer 等等
    
2. class tf.train.GradientDescentOptimizer
    ① 初始化：optimizer = tf.train.GradientDescentOptimizer(learning_rate, 
                                                             use_locking=False, 
                                                             name="GradientDescent")
        learning_rate: 学习率 
        use_locking: 要是True的话，就对于更新操作（update operations）使用锁 
        name: 名字，可选，默认是”GradientDescent”
    ② 这个类实现的是优化算法是：梯度下降算法（GD）

3. class tf.train.AdadeltaOptimizer
    ① 初始化：optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, 
                                                      rho=0.95, 
                                                      epsilon=1e-8,
                                                      use_locking=False, 
                                                      name="Adadelta")
        learning_rate: 学习率 
        rho: 过去梯度的平方的指数衰减率
        epsilon: 梯度修正率
        use_locking: 要是True的话，就对于更新操作（update operations）使用锁 
        name: 名字，可选，默认是”Adadelta”
    ② 这个类实现的是优化算法是：Adadelta
    
4. class tf.train.AdagradOptimizer
    ① 初始化：optimizer = tf.train.AdagradOptimizer(learning_rate, 
                                                     initial_accumulator_value=0.1,
                                                     use_locking=False, 
                                                     name="Adagrad")
        learning_rate: 学习率 
        initial_accumulator_value: 分母累加值初始值，必须是正数
        use_locking: 要是True的话，就对于更新操作（update operations）使用锁 
        name: 名字，可选，默认是”Adagrad”
    ② 这个类实现的是优化算法是：Adagrad

5. class tf.train.MomentumOptimizer
    ① 初始化：optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                                      momentum,
                                                      use_locking=False, 
                                                      name="Momentum", 
                                                      use_nesterov=False)
        learning_rate: 学习率 
        momentum: 动量
        use_locking: 要是True的话，就对于更新操作（update operations）使用锁 
        name: 名字，可选，默认是”Momentum”
        use_nesterov: 要是True的话，就使用Nesterov Momentum
    ② 这个类实现的是优化算法是：Momentum

6. class tf.train.AdamOptimizer
    ① 初始化：optimizer = tf.train.AdamOptimizer(learning_rate=0.001, 
                                                  beta1=0.9, 
                                                  beta2=0.999, 
                                                  epsilon=1e-8,
                                                  use_locking=False, 
                                                  name="Adam")
        learning_rate: 学习率 
        beta1: The exponential decay rate for the 1st moment estimates.
        beta2: The exponential decay rate for the 2st moment estimates.
        epsilon: epsilon hat
        use_locking: 要是True的话，就对于更新操作（update operations）使用锁 
        name: 名字，可选，默认是”Adam”
    ② 这个类实现的是优化算法是：Adam

'''
