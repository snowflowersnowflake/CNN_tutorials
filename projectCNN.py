## mnist_lenet5_forward.py
# coding:utf-8
import tensorflow as tf


# 设定神经网络的超参数
# 定义神经网络可以接收的图片的尺寸和通道数
IMAGE_SIZE = 28
NUM_CHANNELS = 1
# 定义第一层卷积核的大小和个数
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
# 定义第二层卷积核的大小和个数
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
# 定义第三层全连接层的神经元个数
FC_SIZE = 512
# 定义第四层全连接层的神经元个数
OUTPUT_NODE = 10


# 定义初始化网络权重函数
def get_weight(shape, regularizer):
    '''
    args:
    shape：生成张量的维度
    regularizer: 正则化项的权重
    '''
    # tf.truncated_normal 生成去掉过大偏离点的正态分布随机数的张量，stddev 是指定标准差
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # 为权重加入 L2 正则化，通过限制权重的大小，使模型不会随意拟合训练数据中的随机噪音
    if regularizer is not None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 定义初始化偏置项函数
def get_bias(shape):
    '''
    args:
    shape：生成张量的维度
    '''
    b = tf.Variable(tf.zeros(shape)) # 统一将 bias 初始化为 0
    return


# 定义卷积计算函数
def conv2d(x, w):
    '''
    args:
    x: 一个输入 batch
    w: 卷积层的权重
    '''
    # strides 表示卷积核在不同维度上的移动步长为 1，第一维和第四维一定是 1，
    # 这是因为卷积层的步长只对矩阵的长和宽有效:
    # padding='SAME'表示使用全 0 填充，而'VALID'表示不填充
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义最大池化操作函数
def max_pool_2x2(x):
    '''
    args:
    x: 一个输入 batch
    '''
    # ksize 表示池化过滤器的边长为 2，strides 表示过滤器移动步长是 2，'SAME'提供使用全 0 填充
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义前向传播的过程
def forward(x, train, regularizer):
    '''
    args:
    x: 一个输入 batch
    train: 用于区分训练过程 True，测试过程 False
    regularizer：正则化项的权重
    '''
    # 实现第一层卷积层的前向传播过程
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)  # 初始化卷积核
    conv1_b = get_bias([CONV1_KERNEL_NUM]) # 初始化偏置项
    conv1 = conv2d(x, conv1_w) # 实现卷积运算
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) # 对卷积后的输出添加偏置，并过 relu 非线性激活函数
    pool1 = max_pool_2x2(relu1) # 将激活后的输出进行最大池化
    # 实现第二层卷积层的前向传播过程，并初始化卷积层的对应变量
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer) # 该层每个卷积核的通道数要与上一层卷积核的个数一致
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)  # 该层的输入就是上一层的输出 pool1
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)
    # 将上一池化层的输出 pool2（矩阵）转化为下一层全连接层的输入格式（向量）
    pool_shape = pool2.get_shape().as_list()  # 得到pool2输出矩阵的维度，并存入list中，注意pool_shape[0]是一个 batch 的值
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  # 从 list 中依次取出矩阵的长宽及深度，并求三者的乘积就得到矩阵被拉长后的长度
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])  # 将 pool2 转换为一个 batch 的向量再传入后续的全连接
    # 实现第三层全连接层的前向传播过程
    fc1_w = get_weight([nodes, FC_SIZE], regularizer) # 初始化全连接层的权重，并加入正则化
    fc1_b = get_bias([FC_SIZE])  # 初始化全连接层的偏置项
    # 将转换后的 reshaped 向量与权重 fc1_w 做矩阵乘法运算，然后再加上偏置，最后再使用 relu 进行激活
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层输出使用 dropout，也就是随机的将该层输出中的一半神经元置为无效，是为了避免过拟合而设置的，一般只在全连接层中使用
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    # 实现第四层全连接层的前向传播过程，并初始化全连接层对应的变量
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y


## mnist_lenet5_backward.py
# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np
# 定义训练过程中的超参数
BATCH_SIZE = 100 # 一个 batch 的数量
LEARNING_RATE_BASE = 0.005 # 初始学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZER = 0.0001 # 正则化项的权重
STEPS = 50000 # 最大迭代次数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均的衰减率
MODEL_SAVE_PATH="./model/" # 保存模型的路径
MODEL_NAME="mnist_model" # 模型命名


# 训练过程
def backward(mnist):
    # x, y_是定义的占位符，需要指定参数的类型，维度（要和网络的输入与输出维度一致），类似于函数的形参，运行时必须传入值
    x = tf.placeholder(tf.float32,[BATCH_SIZE,
                                   mnist_lenet5_forward.IMAGE_SIZE,
                                   mnist_lenet5_forward.IMAGE_SIZE,
                                   mnist_lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    y = mnist_lenet5_forward.forward(x,True, REGULARIZER) # 调用前向传播网络得到维度为10 的 tensor
    global_step = tf.Variable(0, trainable=False) # 声明一个全局计数器，并输出化为 0
    # 先是对网络最后一层的输出 y 做 softmax，通常是求取输出属于某一类的概率，其实就是一个num_classes 大小的向量，再将此向量和实际标签值做交叉熵，需要说明的是该函数返回的是一个向量
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce) # 再对得到的向量求均值就得到 loss
    loss = cem + tf.add_n(tf.get_collection('losses')) # 添加正则化中的 losses
    # 实现指数级的减小学习率，可以让模型在训练的前期快速接近较优解，又可以保证模型在训练后期不会有太大波动
    # 计算公式：decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True) # 当 staircase=True 时，（global_step/decay_steps）则被转化为整数，以此来选择不同的衰减方式

    # 传入学习率，构造一个实现梯度下降算法的优化器，再通过使用 minimize 更新存储要训练的变量的列表来减小 loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # 实现滑动平均模型，参数 MOVING_AVERAGE_DECAY 用于控制模型更新的速度。训练过程中会对每一个变量维护一个影子变量，这个影子变量的初始值
    # 就是相应变量的初始值，每次变量更新时，影子变量就会随之更新
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]): # 将 train_step 和 ema_op 两个训练操作绑定到 train_op 上
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver() # 实例化一个保存和恢复变量的 saver
    with tf.Session() as sess: # 创建一个会话，并通过 python 中的上下文管理器来管理这个会话
        init_op = tf.global_variables_initializer() # 初始化计算图中的变量
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) # 通过 checkpoint 文件定位到最新保存的模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) # 加载最新的模型

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE) # 读取一个 batch 的数据
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_lenet5_forward.IMAGE_SIZE,
                                          mnist_lenet5_forward.IMAGE_SIZE,
                                          mnist_lenet5_forward.NUM_CHANNELS))  # 将输入数据 xs 转换成与网络输入相同形状的矩阵
            # 喂入训练图像和标签，开始训练
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs,y_: ys})
            if i % 100 == 0: # 每迭代 100 次打印 loss 信息，并保存最新的模型
                print("After %d training step(s), loss on training batch is %g." % (step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True) # 读入 mnist 数据
    backward(mnist)

if __name__ == '__main__':
    main()



## mnist_lenet5_test.py
# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np
TEST_INTERVAL_SECS = 5


def test(mnist):
    # 创建一个默认图，在该图中执行以下操作（多数操作和 train 中一样，就不再重复解释，大家对照学习即可）
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[
            mnist.test.num_examples,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
        y = mnist_lenet5_forward.forward(x,False,None)

        ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # 判断预测值和实际值是否相同
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 求平均得到准确率

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 根据读入的模型名字切分出该模型是属于迭代了多少次保存的
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(mnist.test.images, (mnist.test.num_examples,
                                                                mnist_lenet5_forward.IMAGE_SIZE,
                                                                mnist_lenet5_forward.IMAGE_SIZE,
                                                                mnist_lenet5_forward.NUM_CHANNELS))
                    accuracy_score = sess.run(accuracy,feed_dict={x:reshaped_x, y_: mnist.test.labels})  # 计算出测试集上准确率
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS) # 每隔 5 秒寻找一次是否有最新的模型


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()