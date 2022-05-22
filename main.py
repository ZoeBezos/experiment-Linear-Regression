import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False


# 数据集归一化
def min_max_normalize(x, test):
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0)), \
           (test - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))


# 添加偏置项
def add_bias(x, test):
    return np.append(np.ones([x.shape[0], 1], dtype=float), np.array(x, dtype=float), axis=1), \
           np.append(np.ones([test.shape[0], 1], dtype=float), np.array(test, dtype=float), axis=1)


class Linear_Regression:
    def __init__(self, input_x, input_y, input_test):
        self.x = input_x
        self.y = input_y
        self.test = input_test
        self.closed_theta = np.linalg.pinv(self.x.T.dot(self.x)).dot(self.x.T).dot(self.y)
        self.theta = np.array([[0], [0]], dtype=float)
        self.lr = 0.1
        self.epochs = 100
        self.closed_form_sol()
        self.gradient_descent_sol()

    def closed_form_sol(self):
        y_predict = self.test.dot(self.closed_theta)
        print("---------------------------解析解法--------------------------",
              "\n由解析解法预测南京2014年的房价为：", y_predict[0][0], "千/平方米",
              "\n此时θ值为", self.closed_theta.T, ".T",
              "\n回归曲线为函数解析式为f(x)=", self.closed_theta[0][0], "+", self.closed_theta[1][0], "x",
              "\n---------------------------------------------------------"
              )

    # 假设模型
    def h_x(self, isTraining=True):
        if isTraining:
            return self.x.dot(self.theta)
        else:
            return self.test.dot(self.theta)

    # 损失函数：最小二乘法
    def LSM_func(self, pred):
        return 0.5 * np.sum((pred - self.y).T.dot(pred - self.y))

    # 优化函数：梯度下降法
    def gradient_descent_sol(self):
        epochs_list = []
        loss_list = []
        lr_init = self.lr
        theta_init0, theta_init1 = self.theta[0][0], self.theta[1][0]
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("解析解法和梯度下降法实现线性回归预测南京2014房价", fontsize=16, fontweight='bold')
        plt.subplots_adjust(top=0.85)
        # 可视化的静态部分
        axes[0].set_xlabel("年份")
        axes[0].set_ylabel("房价(千/平方米)")
        axes[0].set_title("房价变化图")
        axes[0].scatter(self.x[:, 1], self.y, color=(0, 0, 0))
        axes[0].plot(self.x[:, 1], self.x.dot(self.closed_theta), label='解析解法', color=(1, 0, 0))
        axes[1].set_xlabel("迭代数")
        axes[1].set_ylabel("误差")
        axes[1].set_title("误差变化图")
        plt.ion()
        for i in range(self.epochs):
            gradient = self.x.T.dot(self.h_x() - self.y)
            self.theta -= self.lr * gradient
            epochs_list.append(i + 1)
            loss_list.append(self.LSM_func(self.h_x()))
            self.lr *= 0.99
            # 可视化的动态部分
            try:
                axes[0].lines.remove(line_0[0])
                axes[1].lines.remove(line_1[0])
            except Exception:
                pass
            line_0 = axes[0].plot(self.x[:, 1], self.h_x(), label='梯度下降法', color=(0, 1, 0))  #
            axes[0].legend()
            line_1 = axes[1].plot(epochs_list, loss_list, color=(0, 0, 0))
            plt.pause(0.1)
        y_predict = self.h_x(False)
        print("---------------------------梯度下降法------------------------",
              "\n当设置学习率为：", lr_init,
              "\nθ的初始值为：[[", theta_init0, theta_init1, "]].T",
              "\n由梯度下降法经历", self.epochs, "次迭代预测南京2014年的房价为：", y_predict[0][0], "千/平方米",
              "\n此时θ值为", self.theta.T, ".T",
              "\n回归曲线为函数解析式为f(x)=", self.theta[0][0], "+", self.theta[1][0], "x",
              "\n---------------------------------------------------------"
              )


if __name__ == '__main__':
    X = np.loadtxt('data\\x.txt', dtype=float, ndmin=2)
    Y = np.loadtxt('data\\y.txt', dtype=float, ndmin=2)
    Test = np.array(2014, dtype=float, ndmin=2)
    X, Test = add_bias(min_max_normalize(X, Test)[0], min_max_normalize(X, Test)[1])

    model = Linear_Regression(X, Y, Test)


