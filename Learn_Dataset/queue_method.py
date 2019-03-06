import tensorflow as tf

### 使用queue读硬盘中的数据

'''
参考：https://zhuanlan.zhihu.com/p/27238630

在tensorflow中，为了方便管理，在内存队列前又添加了一层所谓的“文件名队列”。
在这个队列中可以对数据集进行epoch重复、打乱顺序等操作。
具体使用如下步骤：
1、构建文件名队列（此时队列只是处于构建阶段，还未填充）
    filename_queue = tf.train.string_input_producer(string_tensor,
                                                    num_epochs=None,
                                                    shuffle=True,
                                                    seed=None,
                                                    capacity=32,
                                                    shared_name=None,
                                                    name=None,
                                                    cancel_op=None)
        ① string_tensor：文件名的 list
        ② num_epochs：epoch数
        ③ shuffle：指在一个epoch内文件的顺序是否被打乱，如果设置shuffle=True，那么在一个epoch内，数据的前后顺序就会被打乱
2、创建队列读取对象（当遍历结束是，再次读取会抛出异常）
    reader = tf.WholeFileReader(name=None)
    key, value = reader.read(queue, name=None)  # 此函数每次从queue中读取一个元素
    key, value = reader.read_up_to(queue, 
                                   num_records, 
                                   name=None)   # 此函数每次从queue中读取num_records个元素
3、填充队列
    threads = tf.train.start_queue_runners(sess=None, 
                                           coord=None, 
                                           daemon=True, 
                                           start=True, 
                                           collection=ops.GraphKeys.QUEUE_RUNNERS)
        ① sess 是会话
        ② 只有执行了这步，队列才可以被填充和使用
        ③ 运行此步之前必须运行 sess.run(tf.local_variables_initializer())，因为在 tf.train.string_input_producer 中定义了局部变量（epoch）
'''

# 例子
with tf.Session() as sess:
    # 要读三幅图片A.jpg, B.jpg, C.jpg
    filename = ['A.jpg', 'B.jpg', 'C.jpg']

    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)

    # reader从文件名队列中读数据，对应的方法是reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
    sess.run(tf.local_variables_initializer())

    # 使用start_queue_runners之后，才会开始填充队列
    threads = tf.train.start_queue_runners(sess=sess)

    i = 0
    while True:
        i += 1
        # 获取图片数据并保存
        image_data = sess.run([key, value])
        print(key, value)
