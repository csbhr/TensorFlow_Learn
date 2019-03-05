import tensorflow as tf

# 使用 Dataset API 读取数据

'''
参考：https://zhuanlan.zhihu.com/p/30751039

在TensorFlow中读取数据一般有三种方法：
    ① 使用placeholder读内存中的数据
    ② 使用queue读硬盘中的数据：在 queue_method.py 文件中介绍
    ③ 使用Dataset API读取：在本文件中介绍
    
使用Dataset API读取数据，一般情况下是三个步骤：
    ① 生成Dataset对象
    ② 对Dataset对象进行变换操作
    ③ 由Dataset对象实例化一个Iterator，遍历Dataset对象中的数据
    
1. 生成Dataset对象
    1.1 tf.data.Dataset.from_tensor_slices(tensors) ：可以生成一个Dataset对象
        ① 简单方式，dataset中存储若干个元素，如：
            dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        ② 此函数的真正作用是切分传入Tensor的第一个维度，生成相应的dataset。
        一般第一个维度是batch_size，所以此函数可以把数据切分成一个一个的元素。
        当我们还需要存储label时，可以以字典的形式创建Dataset对象，此函数会自动按第一个维度切分为对用的<image,label>的元素对。
        如：
            dataset = tf.data.Dataset.from_tensor_slices(
                {
                    "image": image_tensor,                                       
                    "label": label_tensor
                }
            )
    1.2 tf.data.TextLineDataset()：
        这个函数的输入是一个文件的列表，输出是一个dataset。dataset中的每一个元素就对应了文件中的一行。
        可以使用这个函数来读入CSV文件。

'''

dataset = tf.data.Dataset.from_tensor_slices(tensors)
