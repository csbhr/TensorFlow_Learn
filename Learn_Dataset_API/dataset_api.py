import tensorflow as tf

### 使用 Dataset API 读取数据


'''
参考：https://zhuanlan.zhihu.com/p/30751039
参考：https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

在TensorFlow中读取数据一般有三种方法：
    ① 使用placeholder读内存中的数据
    ② 使用queue读硬盘中的数据：在 queue_method.py 文件中介绍
    ③ 使用Dataset API读取：在本文件中介绍
    
使用Dataset API读取数据，一般情况下是三个步骤：
    ① 生成Dataset对象
    ② 对Dataset对象进行变换操作
    ③ 由Dataset对象实例化一个Iterator，读取Dataset对象中的数据
    
1. 生成Dataset对象
    1.1 tf.data.Dataset.from_tensor_slices(tensors) ：可以生成一个Dataset对象
        ① 简单方式，dataset中存储若干个元素，如：
            dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        ② 此函数的真正作用是切分传入Tensor的第一个维度，生成相应的dataset。
        一般第一个维度是batch_size，所以此函数可以把数据切分成一个一个的元素。
        当我们还需要存储label时，可以以字典的形式创建Dataset对象，此函数会自动按第一个维度切分为对用的<image,label>的元素对。
        如：
            dataset = tf.data.Dataset.from_tensor_slices((image_tensor, label_tensor))
        或：
            dataset = tf.data.Dataset.from_tensor_slices(
                {
                    "image": image_tensor,                                       
                    "label": label_tensor
                }
            )
        ③ 此函数的输入，可以是数组、张量、placeholder。使用placeholder作为输入可以动态的改变Dataset中的数据。
    1.2 tf.data.TextLineDataset()：
        这个函数的输入是一个文件的列表，输出是一个dataset。dataset中的每一个元素就对应了文件中的一行。
        可以使用这个函数来读入CSV文件。
    1.3 tf.data.FixedLengthRecordDataset()：
        这个函数的输入是一个文件的列表和一个record_bytes，之后dataset的每一个元素就是文件中固定字节数record_bytes的内容。
        通常用来读取以二进制形式保存的文件，如CIFAR10数据集就是这种形式。
    1.4 tf.data.TFRecordDataset()：
        这个函数是用来读TFRecord文件的，dataset中的每一个元素就是一个TFExample。

2. 对Dataset对象进行变换操作（map、batch、shuffle、repeat）
    2.1 map：
        map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset。
        如对dataset中每个元素的值加1：
            dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
            dataset = dataset.map(map_func=lambda x: x + 1)  # 2.0, 3.0, 4.0, 5.0, 6.0
    2.2 batch：
        batch就是将多个元素组合成batch。
        如将dataset中的每个元素组成了大小为32的batch：
            dataset = dataset.batch(batch_size=32)
    2.3 shuffle：
        shuffle的功能为打乱dataset中的元素。
        它有一个参数buffersize，表示打乱时使用的buffer的大小。
        如使用缓存为10000来打乱数据集：
            dataset = dataset.shuffle(buffer_size=10000)
    2.4 repeat：
        repeat的功能就是将整个序列重复多次，主要用来处理机器学习中的epoch。
        它有一个参数count，表示重复的次数，即epoch的数量，如果不指定count值，则无限重复下去，没有结束。
        如原先的数据是一个epoch，将之变成5个epoch：
            dataset = dataset.repeat(count=5)

3. 由Dataset对象实例化一个Iterator，读取Dataset对象中的数据
    3.1 One shot Iterator（最简单的Iterator）
        创建方式：iter = dataset.make_one_shot_iterator()
        一般用于从数组、张量创建的Dataset
        例子：
            x = np.random.sample((100,2))
            dataset = tf.data.Dataset.from_tensor_slices(x)  # make a dataset from a numpy array
            iter = dataset.make_one_shot_iterator()  # create the iterator
            el = iter.get_next()  # get data from iterator
            with tf.Session() as sess:
                print(sess.run(el))  # output: [ 0.42116176  0.40666069]
    3.2 initializable iterator（可初始化的迭代器）
        创建方式：iter = dataset.make_initializable_iterator()
        一般用于从placeholder创建的Dateset，可以利用feed-dict机制变换数据集
        例子：
            EPOCHS = 10
            x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
            dataset = tf.data.Dataset.from_tensor_slices((x, y))  # make a dataset from placeholder
            iter = dataset.make_initializable_iterator()  # create the iterator
            features, labels = iter.get_next()  # get data from iterator
            train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
            test_data = (np.array([[1,2]]), np.array([[0]]))
            with tf.Session() as sess:
                sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1]})  # initialise iterator with train data
                for _ in range(EPOCHS):
                    sess.run([features, labels])
                sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1]})  # switch to test data
              print(sess.run([features, labels]))
'''
