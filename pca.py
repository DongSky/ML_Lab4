#encoding:utf-8
import numpy as np
import argparse
from skimage.io import imread
from skimage.io import imsave
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def pca(data, n):
    # first, normalize the data into mu == 0
    mean = np.mean(data)
    meanData = data - mean
    # then, calculate the eigen vector and eigen value
    cov = np.mat(np.cov(meanData, rowvar=0))
    eigValue, eigVector = np.linalg.eig(cov)
    # sort the eigen vector by sorting eigen value, and then choose top n
    eigValueIndice = np.argsort(eigValue)
    targetEigValue = eigValueIndice[-1:-(n+1):-1]
    targetEigVector = eigVector[:,targetEigValue]
    # calculate new data
    newData = meanData * targetEigVector
    # retransform the data into standard
    newData = newData * targetEigVector.T + mean
    return newData 
if __name__ == "__main__":
    # some parameters
    num = 100
    mean = [0, 0, 0]
    cov = [[3, 6, 4], [1, 1, 1], [4, 6, 3]]
    # generate data
    x, y, z = np.random.multivariate_normal(mean, cov, num ).T
    initMat = np.mat([np.array([x[i], y[i], z[i]]) for i in range(len(x))])
    # calculate new data
    outputMat = np.mat(pca(initMat,2))
    print(outputMat)
    # draw pics
    ax = plt.figure().add_subplot(111,projection='3d')
    ax.scatter(x,y,z,c='g')
    ax.scatter(outputMat[:,0],outputMat[:,1],outputMat[:,2],c='r') 
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
