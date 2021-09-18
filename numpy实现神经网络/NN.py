"""
 Env: /anaconda3/python3.7
 Time: 2021/8/10 14:46
 Author: karlieswfit
 File: NN.py
 Describe: 通过numpy 实现一个简单的神经网络 并通过sklearn鸢尾花数据集进行分类预测

"""

import numpy as np

class Activation_Function:
    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, X):
        return X * (1 - X)


# 定义一个单层隐藏的神经网络层
class SimpleNeuralNetwork:
    def __init__(self, input_size, output_size, activationFuction):
        self.input_size = input_size
        self.output_size = output_size
        self.activationFuction = activationFuction
        self.W = np.random.uniform(low=-0.5, high=0.5, size=(output_size, input_size))
        self.b = np.zeros(shape=(output_size, 1))

    def forward(self, X):
        self.input = X
        Z = np.dot(self.W, X) + self.b
        self.output = self.activationFuction.forward(Z)
        return self.output

    def backward(self, delta):
        # 计算上一层的delta
        self.delta = self.activationFuction.backward(self.input) * np.dot(self.W.T, delta)
        self.W_grad = np.dot(delta, self.input.T)
        self.b_grad = delta


class NeuralNetworks:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(SimpleNeuralNetwork(layers[i], layers[i + 1], Activation_Function()))

    def train(self, input, target, lr):
        self.predict(input)
        self.caculated_gradient(target=target)
        self.update_W_b(lr=lr)

    def predict(self, input):
        for i in range(len(self.layers)):
            output = self.layers[i].forward(input)
            input = output

        return output

    def caculated_gradient(self, target):
        delta = (self.layers[-1].output - target) * self.layers[-1].activationFuction.backward(self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta

    def update_W_b(self, lr):
        for layer in self.layers:
            layer.W -= layer.W_grad * lr
            layer.b -= layer.b_grad * lr

    def loss(self,y_pre,targer):
        return ((y_pre-targer)**2).sum()/2



def one_hot(target,n):
    list=[]
    for i in target:
        inner_list = []
        for j in range(n):
            if i==j:
                inner_list.append(1)
            else:inner_list.append(0)
        list.append(inner_list)
    return np.array(list).reshape(len(target),-1)



def train():
    from sklearn.datasets import load_iris
    epoch=400
    data = load_iris().data
    targets = load_iris().target
    targets=one_hot(target=targets,n=3)
    model=NeuralNetworks([4,6,3])
    for index in range(epoch):
        loss=0
        for (input,target) in zip(data,targets):
            model.train(input.reshape(-1,1),target.reshape(-1,1),lr=0.05)
            loss+=model.loss(model.predict(input.reshape(-1,1)),target.reshape(-1,1))
        if index%10==0:
            print('第{index}次迭代的loss:{loss}'.format(index=index,loss=loss))

    return model


def test(model):
    from sklearn.datasets import load_iris
    data = load_iris().data
    targets = load_iris().target
    sum=0
    for i in range(len(data)):
        y_pre=model.predict(data[i].reshape(-1,1)).argmax()
        sum+=(y_pre==targets[i])

    return sum/(len(data))


if __name__ == '__main__':
    model=train()
    print(test(model))


"""
结果：
第360次迭代的loss:0.4684289119392209
第370次迭代的loss:0.47037902629258793
第380次迭代的loss:0.4748604631962376
第390次迭代的loss:0.4822365144828569
0.96

"""


