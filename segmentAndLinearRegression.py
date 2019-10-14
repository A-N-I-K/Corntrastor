# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:08:18 2019

@author: Jen

# Segment image by the number of acturall row
# Do Linear regression on each segment
# Not much differences by checking on the plots
# Need to evaluation the MSPE between the lodged and non-lodged images.
# Or only fit the vertical lines, since the regression model also change its slop to minimize MSE, but we do not need them to do prediction.
"""
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image
import math
from sklearn.linear_model import LinearRegression
import driver as dr


# convert the image to coordinates array
def toNpArray(image):
    arr = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] == 255:  # if there is a dot, save its coordinates               
                arr.append((i, j))
    return arr;


# equally segment the coordinates, return 2d array
def segment(label, coordinates, width, col):
    array2d = [[] for i in range(label)]  # segment points
    for coor in coordinates:
        index = math.floor(coor[1] / width)
        array2d[index].append(coor);
    return array2d


# do linear_regression(y to x) by the given data([x1,y1],[x2,y2]...)
# return a pair of predicted x, and actural Y for graph
def linear_regression(arr):
    np_coor = np.array(arr)
    x = np_coor[:, 1].reshape(-1, 1)
    Y = np_coor[:, 0].reshape(-1, 1)
    reg = LinearRegression().fit(Y, x)  # Since it is vertical line, fit y to x(reverse x and y in normal regression)
    x_pred = reg.predict(Y);
    return (x_pred, Y)


def drawImg(coor, arr2d):
    np_coor = np.array(coor)  # convert the list to nparray
    plt.scatter(np_coor[:, 1], np_coor[:, 0])  # x y
    for segment in arr2d:  # draw predicted line
        res = linear_regression(segment)
        plt.plot(res[0], res[1], color='red', linewidth=2)
        plt.show()


def main2():
    img1 = imread("040.png")  # non-lodged
    label1 = 4;  # "040.png" has four rows of corns
    # np array
    # row1 = len(img1)
    col1 = len(img1[0])
    width1 = col1 / label1
    filteredImage1 = dr.filterClusters(img1, 10)
    coordinates1 = toNpArray(filteredImage1)
    arr1 = segment(label1, coordinates1, width1, col1);
    plt.figure(1);
    drawImg(coordinates1, arr1)
    
    img2 = imread("000.png")  # Looks like lodged. I took it from cybox
    label2 = 4;  # "000.png" has four rows of corns
    col2 = len(img2[0])
    width2 = col2 / label2
    filteredImage2 = dr.filterClusters(img2, 10)
    coordinates2 = toNpArray(filteredImage2)
    arr2 = segment(label2, coordinates2, width2, col2);
    plt.figure(2)
    drawImg(coordinates2, arr2)
    
    
if __name__ == '__main__':
    
    main2()
    pass
