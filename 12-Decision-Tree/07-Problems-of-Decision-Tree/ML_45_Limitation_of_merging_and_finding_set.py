#coding:utf-8
"""
------------------------------------------------
@File Name    : ML_45_Limitation_of_merging_and_finding_set
@Function     : 
@Author       : Minux
@Date         : 2018/10/7
@Revised Date : 2018/10/7
------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:,2:]
y = iris.target

'''
数据基本分布信息可视化
'''
def Plot_Info_IRIS():
    plt.scatter(X[y==0,0],X[y==0,1])
    plt.scatter(X[y==1,0],X[y==1,1])
    plt.scatter(X[y==2,0],X[y==2,1])
    plt.show()


'''
可视化决策边界函数
'''
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    X_new = np.c_[x0.ravel(),x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])

    plt.contourf(x0,x1,zz,cmap=custom_cmap)

def Decision_Tree_Function():
    dt_clf = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=729)
    dt_clf.fit(X, y)
    plot_decision_boundary(dt_clf,axis=[0.5, 7.5, 0, 3])
    Plot_Info_IRIS()

'''
说明决策树算法对样本点敏感,但是不稳定
'''
def Show_Limitation_of_DT():
    X_new = np.delete(X, 138, axis=0)
    y_new = np.delete(y, 138)
    # print(X_new.shape)
    # print(y_new.shape)
    dt_clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
    dt_clf.fit(X_new, y_new)
    plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
    Plot_Info_IRIS()




if __name__ == '__main__':
    # Decision_Tree_Function()
    Show_Limitation_of_DT()
