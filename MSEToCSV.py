# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:58:36 2019

@author: Jieyun Hu
"""

import driver
import os
import pandas as pd
from skimage.io import imread
import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
import statistics

#lower the threthold for cluter fileter filterClusters(img, 3) 

def imgProcess(toBulkProcess):
    
    if(toBulkProcess == False):
        return

    handlerProcess = driver.File(driver.INPUTFOLDERNAME, driver.INTERMEDFOLDERNAME)
        
    # Get images
    imageList = handlerProcess.getImages()
        
    # Process images
    processedImageList = driver.bulkProcess(imageList)
        
    # Save images
    handlerProcess.setImages(processedImageList)
        
    # Initialize filter handler
    handlerFilter = driver.File(driver.INTERMEDFOLDERNAME, driver.OUTPUTFOLDERNAME)
        
    # Get images
    imageList = handlerFilter.getSKImages()
        
    # Cluster filter images
    filteredImageList = driver.bulkFilter(imageList)
        
    # Save images
    handlerFilter.setSKImages(filteredImageList)
    

def writeTofile(array,csvFileName,toCSV):
    if toCSV == False:
        return
    #filewriter = csv.writer(open(csvFileName, 'w'), delimiter=' ', lineterminator='\n')
    #filewriter.writerow(array)
    with open(csvFileName, 'w', newline ='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(array)
    csvFile.close()


def strictFit(line,points,rowNum,width,drawable):
    
    strictBounds = line.getStrictFit3(points, rowNum, width)  # MSE Minimization Variation
        
    firstLine = strictBounds[0]
    lastLine = strictBounds[1]

    bandwidth = (lastLine - firstLine)/(rowNum-1)
    
    drawImg2(points, firstLine,rowNum,bandwidth,drawable)
     
    return [strictBounds[2],strictBounds[3]]
    
def verticalFit(line,points,rowNum,width,drawable):
     
    verticalFitModel = line.getVerticalFit(points,rowNum,width)
    
    interceptArr = verticalFitModel[0]
    
    drawImg3(points, interceptArr,drawable)
    
    return [verticalFitModel[1],verticalFitModel[2]]
      

def densityFit(img,rows,drawable):
   
    densityArr = pointsPerCol(copy.copy(img))
     #print(densityArr)
    peaks = findPeaks(rows,densityArr,9)
        
    #print(peaks)
        
    res = findMSE(img,peaks,rows)
        
    if(drawable == True):
    
        points = coordinatesArray(img)
        
        drawImg3(points,peaks,drawable)
    
    return res
        
def drawImg2(points, firstLine,row,bandwidth,drawable):#based on strict fitting
    if(drawable == False):
        return
    np_points = np.array(points)  # convert the list to nparray
    plt.scatter(np_points[:, 1], np_points[:, 0])  # x y
    index = 0
    while(index<row):
        line = index*bandwidth+firstLine
        plt.axvline(x=line,color='red',linewidth=2)
        plt.show()
        index = index+1    
  
def drawImg3(points,interceptArr,drawable):    #based on vertical fitting  
    if(drawable == False):
        return
    np_points = np.array(points)  # convert the list to nparray
    plt.scatter(np_points[:, 1], np_points[:, 0])  # x y
    for intercept in interceptArr:
        plt.axvline(x=intercept,color='red',linewidth=2)
        plt.show()
        
def pointsPerCol(img):
    width = len(img[0])
    lst = [0]*width
    for i in range(len(img)):
        for j in range(width):
            if img[i][j]==255:
                lst[j] += 1
    return lst
                
def findPeaks(rows,arr,maskN):
    width = len(arr)
    peaks = []
    boarderMask = 5
    for index in range(width):
        if(index<=boarderMask or index >= width-boarderMask):
            arr[index]=0
    for row in range(rows):
        maxIndex = 0
        maxIndex = arr.index(max(arr))
        peaks.append(maxIndex)
        leftborder = (maxIndex-maskN)>0 and (maxIndex-maskN) or 0
        rightborder = (maxIndex+maskN)>width and width or maxIndex+maskN
        for i in range(leftborder,rightborder):
            arr[i] = 0
    return peaks

#convert the image to coordinates array
#arr = []
def coordinatesArray(image):
    arr = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j]==255: #if there is a dot, save its coordinates               
                arr.append((i,j))
    return arr;

#Find MSE,Mean,sd for given lines
def findMSE(img,lstOfLines,rows):
    
     height = len(img)
     width = len(img[0])
     lstOfLines.sort()
     partitionLines = []
     #partiton the points by mid point of lstOfLines
     
     for i in range(rows-1):
         mid = (lstOfLines[i+1]+lstOfLines[i])/2
         partitionLines.append(mid)
     
     partitionLines.append(width)
     index = 0
     totalSS = 0
     totalPoints = 0

     # store distances from points to given lines in its partition to caculate dev and mean
     distArrBySeg = [[] for i in range(rows)]
     for col in range(width):
         seg_ss = 0
         
         for row in range(height):
             #print([row,col])
             if (col<=partitionLines[index]):
                 if (img[row][col]==255):
                     totalPoints +=1
                     seg_ss += (col-lstOfLines[index])**2
                     distArrBySeg[index].append(abs(col-lstOfLines[index]))
             else:
                 index += 1;
                 if (img[row][col]==255):
                     totalPoints +=1
                     seg_ss += (col-lstOfLines[index])**2
                     distArrBySeg[index].append(abs(col-lstOfLines[index]))
         totalSS += seg_ss
     
     
     distDevArr = []
     distMeanArr = []
     for seg in distArrBySeg:
         distDevArr.append(statistics.stdev(seg))
         distMeanArr.append(statistics.mean(seg))
     totalMean = statistics.mean(distMeanArr)  
     
     #index 0 : totalMSE, index 1: totalmean; index2: standard deviation for each segment; index 3: mean for each segement
     return [totalSS/totalPoints,totalMean,distDevArr,distMeanArr]      
                 

def main2():
    
    toCSV = True #convert to csv
    
    drawable = False # draw the picture
    
    toBulkProcess = False # bulk processing the picture
    
    strict = False
    
    vertical = False
    
    density = False
    
    best = True
    
    imgProcess(toBulkProcess)
    
    path,dirs,files = next(os.walk("raw_images"))
    # number of files in folder
    file_count = len(files)
    #read labels file
    readLabels= pd.read_csv('labels.csv')
    #convert labels to list
    rowNumList = readLabels["rowNum"].tolist()
    #no sideTrim
    sideTrim = 0
    
    MSEArr = []
    MSE = []
    Mean = []
    DevArr = []
    MeanArr = []
    n = file_count #can be set to a different value for iterating
       
    for i in range(n):
        filename = ""
        if (vertical or strict or best):
            filename = "filtered_images/%03d.png" % i
        elif (density):
            filename = "processed_images/%03d.png" % i
        
        img = None
              
        line = driver.Line(sideTrim)
                
         # Open a single image
        try:
                    
            img = imread(filename)

        except FileNotFoundError:
                    
            print ("Invalid filename")
                
         # Image properties
        width = len(img[0])
        height = len(img)
                
        res = []
        if (strict == True): 
            points = line.getPoints(img) 
            res = strictFit(line,points,rowNumList[i],width,drawable)
            if len(res)>0:
                res[1].sort()
                MSEArr.append(res[1])
                MSE.append([res[0]])                
        elif (vertical == True):
            points = line.getPoints(img) 
            res = verticalFit(line,points,rowNumList[i],width,drawable)
            if len(res)>0:  
                res[1].sort()
                MSEArr.append(res[1])
                MSE.append([res[0]]) 
        elif(density == True):
            res = densityFit(img,rowNumList[i],drawable)
            if len(res)>0:
                res[2].sort()
                res[3].sort()
                MSE.append([res[0]]) 
                Mean.append([res[1]])
                DevArr.append(res[2]) # it is stand deviation
                MeanArr.append(res[3])
        elif(best == True):
            #code from Anik
            segments = []
            totalMSE = 0
            stripWidth = round(width / rowNumList[i])
            points = line.getPoints(img)
            for j in range(rowNumList[i]):
                
                subPoints = line.getSubPoints(points, (stripWidth * (j + sideTrim)), (stripWidth * (j + 1 - sideTrim)))
                
                # Get the segment using the best fitting model AND the deviation/MSE
                segment = line.getBestFit(subPoints, j * stripWidth, (j + 1) * stripWidth, height)
                
                # Append ONLY the line segment to the list of line segments
                segments.append((segment[0], segment[1]))
                
                # Print current deviation/MSE of the line segment
                # print(segment[2], " ", end='')
                
                # Update deviation/MSE
                totalMSE += segment[2]
                    
                    # Print average deviation/MSE
                avgMSE = totalMSE / rowNumList[i]
            MSE.append([avgMSE])
    #print(MSE)
    #print(MSEArr)    

    if(density == True):
        writeTofile(DevArr,"Deviation array.csv",toCSV) 
        writeTofile(Mean,"totalMean.csv",toCSV) 
        writeTofile(MeanArr,"Mean array.csv",toCSV) 
    elif(vertical == True or strict == True):
        writeTofile(MSEArr,"mse_arr.csv",toCSV)
    writeTofile(MSE,"totalMSE.csv",toCSV) 

if __name__ == '__main__':   
    main2()
    pass