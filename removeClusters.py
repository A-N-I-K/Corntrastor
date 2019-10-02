# -*- coding: utf-8 -*-
'''
Created on Oct 1, 2019

@author: Jieyun Hu
'''
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image


def removeClusters(image,row,col):#pick the top-left point    
    for i in range(row):
        for j in range(col):
            if image[i][j]==255: #if it is a white dot, do dfs
                image[i][j]=1 #set the top-left point to 1
                dfs(i,j,image,row,col)
                image[i][j]=255 #set it back to 255


def dfs(i,j,image,row,col):
    if(i<0 or i>=row or j<0 or j>=col):
        return
    if(image[i][j]==0): return
    if(image[i][j]==255):
        image[i][j]=0        
    dfs(i+1,j,image,row,col)
    dfs(i,j+1,image,row,col)
    dfs(i-1,j,image,row,col)
    dfs(i,j-1,image,row,col)

def showPic(image):
    img = Image.fromarray( image , 'L')
    img.show()   
    
def main():
    FileName = "040.png"
    image = imread(FileName) #np array
    showPic(image)
    row = len(image)
    col = len(image[0])
    removeClusters(image,row,col)
    showPic(image)
    
if __name__ == '__main__':
    
    main()
    
    