# -*- coding: utf-8 -*-
'''
Created on Oct 1, 2019

@author: Jen
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
    
def findSegmentBoarders(image,label): #find the right boarder for the first label-1 segments.
    col = len(image[0])
    arr = []
    boarder = 0;
    width = col/label
    arr.append(boarder) #add 0
    for i in range(label-1):
        boarder = width+boarder
        arr.append(boarder)
    arr.append(col)
    return arr

def dfs2(i,j,row,col,image,centerPoint,closedPoint,minDist):#closedPoint is an array to store the most closed point with x,y coordinates 
    if(i<0 or i>=row or j<0 or j>=col):#minDist is to store the minimum distance from all points in cluster
        return
    if(image[i][j]==0 or image[i][j]==1): return
    if(image[i][j]==255):
        image[i][j]=0
        distY = abs(j-centerPoint) 
        if(distY<minDist):           
            closedPoint[0]=i
            closedPoint[1]=j
            #print(closedPoint)
            minDist = distY  
    #print(closedPoint)
    dfs2(i+1,j,row,col,image,centerPoint,closedPoint,minDist)
    dfs2(i,j+1,row,col,image,centerPoint,closedPoint,minDist)
    dfs2(i-1,j,row,col,image,centerPoint,closedPoint,minDist)
    dfs2(i,j-1,row,col,image,centerPoint,closedPoint,minDist)
    return closedPoint   

def reduceCluster2(image,boarders):# pick the points most closed to the center
    row = len(image)
    col = len(image[0]) 
    n = len(boarders)
    for i in range(row):
        for j in range(col):
            for k in range(n-1):
                if(j>boarders[k] and j<boarders[k+1]):
                    centerPoint = (boarders[k+1]+boarders[k])/2
                    closedPoint = [i,j]
                    closedPoint = dfs2(i,j,row,col,image,centerPoint,closedPoint,col)#tried to pass closedPoint by reference, but it didn't work like that.So add return the closedpoint in dfs instead.
                    if(closedPoint is not None):
                        image[closedPoint[0]][closedPoint[1]]=1
    

def showPic(image):
    img = Image.fromarray( image , 'L')
    img.show()  
    
def changeto225(image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            if(image[i][j]==1):
                image[i][j]=255
    
def main3():
    FileName = "040.png"
    image = imread(FileName) #np array
    showPic(image)
    row = len(image)
    col = len(image[0])
    removeClusters(image,row,col)
    showPic(image)
    
    image2 = imread("040.png")
    boarders=findSegmentBoarders(image2,4)
    reduceCluster2(image2,boarders)
    changeto225(image2)
    showPic(image2)
    
if __name__ == '__main__':
    
    main3()
    
    