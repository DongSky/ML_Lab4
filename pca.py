#encoding:utf-8
import numpy as np
import argparse
from skimage.io import imread
from skimage.io import imsave
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def pca(data, n):
    mean = np.mean(data)
    meanData = data - mean
    cov = np.mat(np.cov(meanData, rowvar=0))
    eigValue, eigVector = np.linalg.eig(cov)
    eigValueIndice = np.argsort(eigValue)
    targetEigValue = eigValueIndice[-1:-(n+1):-1]
    targetEigVector = eigVector[:,targetEigValue]
    newData = meanData * targetEigVector
    newData = newData * targetEigVector.T + mean
    return newData 
def psnr(img1, img2):
    diff = np.abs(img1 - img2)
    rmse = np.sqrt((np.array(diff)**2).sum())
    psnr = 20 * np.log10(255/rmse)
    return np.abs(psnr)
if __name__ == "__main__":
    num = 100
    mean = [0, 0, 0]
    cov = [[10, 8, 5], [2, 1, 1], [4, 6, 3]]
    x, y, z = np.random.multivariate_normal(mean, cov, num ).T
    initMat = np.mat([np.array([x[i], y[i], z[i]]) for i in range(len(x))])
    outputMat = np.mat(pca(initMat,2))
    print(outputMat)
    ax = plt.figure().add_subplot(111,projection='3d')
    ax.scatter(x,y,z,c='g')
    ax.scatter(outputMat[:,0],outputMat[:,1],outputMat[:,2],c='r') 
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
