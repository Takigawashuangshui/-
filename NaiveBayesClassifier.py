import numpy as np
# 只考虑离散值
class NaiveBayesClassifier:
    def __init__(self,n_classes=2):
        #n_classes 为属性数，分类数。
        self.n_classes=n_classes
        self.priori_P={}
        self.conditional_P={}
        self.N={}
        pass

    def fit(self,X,y):#y最终存放分类结果1/0。
        for i in range(self.n_classes):
            # 公式 7.19
            self.priori_P[i]=(len(y[y==i])+1)/(len(y)+self.n_classes)
        for col in range(X.shape[1]):#x.shape[1] 为矩阵x第一维长度,col是属性数（根据x第一维长度个数）
            self.N[col]=len(np.unique(X[:,col]))#N样本空间
            self.conditional_P[col]={}
            for row in range(X.shape[0]):#遍历每个样本x
                val=X[row,col]#此处x的取值
                if val not in self.conditional_P[col].keys():#对属性col下每个取值val计算条件概率
                    self.conditional_P[col][val]={}
                    for i in range(self.n_classes):
                        D_xi=np.where(X[:,col]==val)#在属性col上取值为val的集合
                        D_c=np.where(y==i)#在分类c上取值为i的集合
                        D_cxi=len(np.intersect1d(D_xi,D_c))#在分类c属性col上取值为val的集合
                        # 公式 7.20
                        self.conditional_P[col][val][i]=(D_cxi+1)/(len(y[y==i])+self.N[col])
                else:
                    continue

    def predict(self,X):#输入Xtest
        pred_y=[]#记录每一个x对应的分类结果
        for i in range(len(X)):
            p=np.ones((self.n_classes,))
            for j in range(self.n_classes):
                p[j]=self.priori_P[j]#P(c)
            for col in range(X.shape[1]):
                val=X[i,col]
                for j in range(self.n_classes):
                    p[j]*=self.conditional_P[col][val][j]
            pred_y.append(np.argmax(p))  #argmax返回使p值最大的分类结果的索引
        return np.array(pred_y)
# 连续值
class NaiveBayesClassifierContinuous:
    def __init__(self,n_classes=2):
        self.n_classes=n_classes
        self.priori_P={}

    def fit(self,X,y):
        self.mus=np.zeros((self.n_classes,X.shape[1]))
        self.sigmas=np.zeros((self.n_classes,X.shape[1]))

        for c in range(self.n_classes):
            # 公式 7.19
            self.priori_P[c]=(len(y[y==c]))/(len(y))
            X_c=X[np.where(y==c)]

            self.mus[c]=np.mean(X_c,axis=0)  #axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
            self.sigmas[c]=np.std(X_c,axis=0)#标准差

    def predict(self,X):
        pred_y=[]
        for i in range(len(X)):
            p=np.ones((self.n_classes,))
            for c in range(self.n_classes):
                p[c]=self.priori_P[c]
                for col in range(X.shape[1]):
                    x=X[i,col]
                    #公式7.18
                    p[c]*=1./(np.sqrt(2*np.pi)*self.sigmas[c,col])*np.exp(-(x-self.mus[c,col])**2/(2*self.sigmas[c,col]**2))
            pred_y.append(np.argmax(p))
        return np.array(pred_y)

if __name__=='__main__':
    X = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                [2, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1],
                                [1, 1, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0],
                                [1, 1, 1, 1, 1, 0], [0, 2, 2, 0, 2, 1],
                                [2, 2, 2, 2, 2, 0], [2, 0, 0, 2, 2, 1],
                                [0, 1, 0, 1, 0, 0], [2, 1, 1, 1, 0, 0],
                                [1, 1, 0, 0, 1, 1], [2, 0, 0, 2, 2, 0],
                                [0, 0, 1, 1, 1, 0]])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    X_test=np.array([[0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                    [1, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 0],
                     [1, 1, 0, 0, 1, 1], [2, 0, 0, 2, 2, 0],
                     [0, 0, 1, 1, 1, 0],
                     [2, 0, 0, 2, 2, 0],
                     [0, 0, 1, 1, 1, 0]
                     ])

    naive_bayes=NaiveBayesClassifier(n_classes=2)
    naive_bayes.fit(X,y)
    pred_y=naive_bayes.predict(X_test)
    print('pred_y:',pred_y)

    naive_bayes=NaiveBayesClassifierContinuous(n_classes=2)
    naive_bayes.fit(X,y)
    pred_y=naive_bayes.predict(X_test)
    print('Continuous pred_y:',pred_y)