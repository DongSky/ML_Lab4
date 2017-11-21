#encoding:utf-8
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
if __name__ == "__main__":
    num = 100
    mean = [0, 0, 0]
    cov = [[10, 8, 5], [2, 1, 1], [4, 6, 3]]
    x, y, z = np.random.multivariate_normal ( mean , cov , num ).T
    output = np.mat([np.array([x[i], y[i], z[i]]) for i in range(len(x))])
    print(output)
    ax = plt.figure().add_subplot(111,projection='3d')
    ax.scatter(x,y,z,c='g')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()