#coding=utf-8
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
if __name__ == '__main__':
    rng = numpy.random

    # 学习速率 迭代次数 50次迭代输出
    learning_rate = 0.01
    training_epochs = 2000
    display_step = 50

    # 训练数据
    train_X = numpy.asarray(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    train_Y = numpy.asarray(
        [2.94, 4.53, 5.96, 7.88, 9.02, 10.94,  12.14, 13.96, 14.74, 16.68, 17.79, 19.67, 21.20, 22.07, 23.75,  25.22, 27.17,  28.84, 29.84, 31.78])
    n_samples = train_X.shape[0]

    # tf Graph Input
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # 创建模型

    # 变量权重和偏置值
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # 构建线性模型
    activation = tf.add(tf.multiply(X, W), b)

    # 最小平方误差
    cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)  # L2 loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 随机梯度下降

    # 初始化变量
    init = tf.global_variables_initializer()

    # 启动模型
    with tf.Session() as sess:
        sess.run(init)
        plt.ion()

        # 训练
        n = 0
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # 每display_step次输出查看
            if epoch % display_step == 0:
                print("step:", '%04d' % (epoch + 1), "cost=","{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), "W=", sess.run(W), "b=", sess.run(b))
            plt.plot(train_X, train_Y, 'ro', label='Original data')
            plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
            n = n + 1
            plt.title(str(n)+ ':  y='+str(sess.run(W))+ 'x+'+str(sess.run(b)))
            plt.pause(0.001)
            plt.show()
            plt.cla()

        print("Optimization Finished!")
        print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}),"W=", sess.run(W), "b=", sess.run(b))

