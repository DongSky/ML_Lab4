#encoding:utf-8
import numpy as np
import argparse
import os
from skimage.io import imread
from skimage.io import imsave
from skimage import transform
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
    resize = 0.3
    parse = argparse.ArgumentParser()
    parse.add_argument("-n",action="store",dest="arg_n",type=int,default=10)
    result = parse.parse_args()
    lfw_path = "./lfw"
    img_path = []
    img_vector = []
    for dirPath, dirNames, fileNames in os.walk(lfw_path):
        for filename in fileNames:
            if filename.split(".")[1] == "jpg":
                img_path.append(os.path.join(dirPath, filename))
    img_path = img_path[:200]
    for i in img_path:
        img_vector.append(np.array(transform.rescale(imread(i,as_grey=True),[resize,resize])).ravel())
    img_vector = np.mat(img_vector)
    print(img_vector.shape)
    new_img_vector = np.mat(pca(img_vector,min(result.arg_n,img_vector.shape[1]))).astype(np.float64)
    new_img_vector = new_img_vector / np.max(np.abs(new_img_vector))
    tot_psnr = 0.0
    for i in range(img_vector.shape[0]):
        tot_psnr += psnr(img_vector[i],new_img_vector[i])
    tot_psnr /= img_vector.shape[0]
    print("PSNR: ",tot_psnr)
    initPic = img_vector[0].reshape((int(250*resize),int(250*resize)))
    newPic = new_img_vector[0].reshape((int(250*resize),int(250*resize)))
    imsave(arr=initPic,fname="init.jpg")
    imsave(str(result.arg_n)+"_pca.jpg",newPic)
    # initPic = imread("test.jpg",as_grey=True)
    # imsave(arr=initPic,fname="init.jpg")
    # newPic = np.mat(pca(np.mat(initPic),result.arg_n))
    # newPic = newPic/np.max(np.abs(newPic))
    # #print(newPic)
    # imsave("pca.jpg",newPic)
    # print("PSNR: ",psnr(newPic,initPic))
