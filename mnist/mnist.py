# coding=utf-8
__author__ = 'zhangxiaozi'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入层节点数
OUTPUT_NODE = 10  # 输出层节点数

'''配置神经网络的参数'''
LAYER1_NODE = 500  # 隐藏层神经元个数
BATCH_SIZE = 100  # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARAZTION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

'''定义一个接口函数，用于计算神经网络的前向结果，
其中参数avg_classs是用于计算参数平均值的类
这样方便在测试时使用滑动平均模型'''


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    '''
    :param input_tensor: 输入
    :param avg_class: 用于计算参数平均值的类
    :param weights1: 第一层权重
    :param biases1: 第一层偏置
    :param weights2: 第二层权重
    :param biases2: 第二层偏置
    :return: 返回神经网络的前向结果
    '''
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类计算参数的滑动平均值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


'''训练模型的过程'''


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')  # 维度可以自动算出，也就是样本数
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))  # 一种正态的随机数
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)  # 一般训练轮数的变量指定为不可训练的参数

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络的参数的变量上使用滑动平均，其他辅助变量就不需要了
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用滑动平均的前向结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1),
                                                                   logits=y)  # labels=tf.argmax(y_, 1), logits=y
    # 这里tf.argmax(y_,1)表示在“行”这个维度上张量最大元素的索引号
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)  # 正则化损失函数
    regularaztion = regularizer(weights1) + regularizer(weights2)  # 模型的正则化损失
    loss = cross_entropy_mean + regularaztion  # 总损失函数=交叉熵损失和正则化损失的和

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率
        global_step,  # 迭代轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAY,  # 学习率衰减速率
        staircase=True)

    # 优化损失函数，用梯度下降法来优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))


def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    print('x')
    main()
    print('x')
