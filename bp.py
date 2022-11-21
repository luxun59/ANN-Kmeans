import numpy as np

''' 
产生从 a 到 b随机数
'''
def rand(a, b): 
    return (b - a) * np.random.random() + a

''' 
激活函数
'''
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

'''
激活函数的导数
'''
def sigmoid_derivative(x):
    return x * (1 - x)



class BP:
    def __init__(self, layer, iter, max_error):
        self.m_nodesnum = layer  # 层的节点个数 
        self.m_weights = []   # 权值矩阵
        self.m_nodeOut = []
        self.m_iterNum = iter          # 最大迭代次数
        self.max_error = max_error  # 停止的误差范围
        self.m_learningRate = 0.1
        self.m_error = []
        # 初始化一个(d+1) * q的矩阵，多加的1是将隐藏层的阀值加入到矩阵运算中
        # 第1个权重为空
        self.m_weights.append(np.ones(self.m_nodesnum[0]))
        for i in range(1,len(self.m_nodesnum)):
            self.m_weights.append ( np.random.random(( self.m_nodesnum[i-1]+ 1, self.m_nodesnum[i])) )
        for i in range(0,len(self.m_nodesnum)):    
            self.m_nodeOut.append ( np.zeros(self.m_nodesnum[i]))
        
    def forword(self,input):
        inputLayer = np.array(input)
        self.m_nodeOut[0] = inputLayer
        for i in range(1,len(self.m_nodesnum)): 
            #第i层的输出
            # print(self.m_nodesnum[i])
            ah= self.m_nodeOut[i-1].dot(self.m_weights[i][1:,:]) + self.m_weights[i][0,:]
            self.m_nodeOut[i] = sigmoid(ah)
        return self.m_nodeOut[-1]

    def backword(self,X,Y):
        output = self.forword(X)
        print("back")
        print(X)
        print('out')
        print(output)
        errorNow= error = np.array(Y) - np.array( output )#.reshape(-1,1)
        # errorNow = errorPrev*sigmoid_derivative(self.m_nodeOut[-1])
        errorNow= errorNow.reshape(-1,1)
        # print(errorNow.shape)
        # print(errorNow)
        

        for i in range(len(self.m_nodesnum)-1,0,-1):
            errorPrev = errorNow
            errorWj = self.m_weights[i][1:,:].dot(errorPrev)
            # errorWj = self.m_weights[i][1:,:] * errorPrev.sum(axis=0) 
            # errorWj = np.squeeze(errorWj.reshape(-1,1)   )
            # print("&&&&&&&&&&&&&&")
            # print(errorWj.shape) 
            # errorNow = sigmoid_derivative(self.m_nodeOut[i]).T
            errorNow = np.squeeze(errorWj) * sigmoid_derivative(self.m_nodeOut[i-1])
            errorNow = errorNow.reshape(-1,1)
            # print("errorNow")
            # print(errorNow.shape)
            # deltaWeight = self.m_learningRate* self.m_nodeOut[i].reshape(-1,1).T.dot(errorPrev.T)
            deltaWeight = self.m_nodeOut[i-1].reshape(-1,1).dot(errorPrev.T)
            deltaWeight =  self.m_learningRate*deltaWeight
            # print('daltaWeight')
            # print(deltaWeight.shape)     
            self.m_weights[i][1:,:] += deltaWeight
            # deltaWeight0 = -self.m_learningRate*errorPrev
            deltaWeight0 = errorPrev
            deltaWeight0 = -1*self.m_learningRate*deltaWeight0
            # print('deltaWeight0')
            # print(deltaWeight0.shape)
            self.m_weights[i][0,:] += np.squeeze(deltaWeight0) 

        print(error)
        print("end")
        return error


    def fit(self, X, y):

        for i in range(self.m_iterNum):
            error = 0.0
            for j in range(len(X)):
                error += self.backword(X[j], y[j])
            error = error.sum()
            if abs(error) <= self.max_error:
                break


    def predict(self, x_test):
        output = self.forword(x_test)
        return output




    # #  正向传播与反向传播
    # def forword_backword(self, xj, y, learning_rate=0.1):
    #     xj = np.array(xj)
    #     y = np.array(y)
    #     input = np.ones((1, xj.shape[0] + 1))
    #     input[:, :-1] = xj
    #     x = input
    #     # ah = np.dot(x, self.input_weights)
    #     ah = x.dot(self.input_weights)
    #     bh = sigmoid(ah)

    #     input = np.ones((1, self.hidden_n + 1))
    #     input[:, :-1] = bh
    #     bh = input

    #     bj = np.dot(bh, self.output_weights)
    #     yj = sigmoid(bj)

    #     error = yj - y
    #     self.gj = error * sigmoid_derivative(yj)

    #     # wg = np.dot(self.output_weights, self.gj)

    #     wg = np.dot(self.gj, self.output_weights.T)
    #     wg1 = 0.0
    #     for i in range(len(wg[0]) - 1):
    #         wg1 += wg[0][i]
    #     self.eh = bh * (1 - bh) * wg1
    #     self.eh = self.eh[:, :-1]

    #     #  更新输出层权值w，因为权值矩阵的最后一行表示的是阀值多以循环只到倒数第二行
    #     for i in range(self.output_weights.shape[0] - 1):
    #         for j in range(self.output_weights.shape[1]):
    #             self.output_weights[i][j] -= learning_rate * self.gj[0][j] * bh[0][i]

    #     #  更新输出层阀值b，权值矩阵的最后一行表示的是阀值
    #     for j in range(self.output_weights.shape[1]):
    #         self.output_weights[-1][j] -= learning_rate * self.gj[0][j]

    #     #  更新输入层权值w
    #     for i in range(self.input_weights.shape[0] - 1):
    #         for j in range(self.input_weights.shape[1]):
    #             self.input_weights[i][j] -= learning_rate * self.eh[0][j] * xj[i]

    #     # 更新输入层阀值b
    #     for j in range(self.input_weights.shape[1]):
    #         self.input_weights[-1][j] -= learning_rate * self.eh[0][j]
    #     return error

    # def fit(self, X, y):

    #     for i in range(self.iter):
    #         error = 0.0
    #         for j in range(len(X)):
    #             error += self.forword_backword(X[j], y[j])
    #         error = error.sum()
    #         if abs(error) <= self.max_error:
    #             break

    # def predict(self, x_test):
    #     x_test = np.array(x_test)
    #     tmp = np.ones((x_test.shape[0], self.input_n + 1))
    #     tmp[:, :-1] = x_test
    #     x_test = tmp
    #     an = np.dot(x_test, self.input_weights)
    #     bh = sigmoid(an)
    #     #  多加的1用来与阀值相乘
    #     tmp = np.ones((bh.shape[0], bh.shape[1] + 1))
    #     tmp[:, : -1] = bh
    #     bh = tmp
    #     bj = np.dot(bh, self.output_weights)
    #     yj = sigmoid(bj)
    #     return yj




if __name__ == '__main__':
    #  指定神经网络输入层，隐藏层，输出层的元素个数
    layer = [2,4,1]
    X = [
            [1, 1],
            [2, 2],
            [1, 2],
            [1, -1],
            [2, 0],
            [2, -1]
        ]
    y = [[0], [0], [0], [1], [1], [1]]

    x_test = [[2, 1],
              [2, 2]]
    #  设置最大的迭代次数，以及最大误差值
    bp = BP(layer, 10000, 0.01)
    print(bp.m_weights)
    print(bp.m_weights[0].shape)
    print(bp.m_weights[1].shape)
    # print(bp.m_weights[2].shape)
    # print(bp.m_weights[3].shape)
    print('forword')
    print(bp.forword([1,1]))


    # A = np.random.random(( 4, 2))
    # b = [2,3]
    # B = np.array(b)
    # print(type(A))
    # print(type(A) == np.ndarray)
    # print(A)
    # print(A.sum(axis=0))
    # print(B)
    # print(B.dot(A.T))
    # print(sigmoid_derivative(B))
    bp.fit(X, y)
    print(bp.predict(X))