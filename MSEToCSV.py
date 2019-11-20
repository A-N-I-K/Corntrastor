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
    
def main2():
    
    toCSV = False #convert to csv
    
    drawable = False # draw the picture
    
    toBulkProcess = False # bulk processing the picture 
    
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
    
    n = file_count #can be set to a different value for iterating
    
    for i in range(n):
        
        filename = "filtered_images/%03d.png" % i
        
        img = None
        
        line = driver.Line(sideTrim)
                
         # Open a single image
        try:
                    
            img = imread(filename)
                    
        except FileNotFoundError:
                    
            print ("Invalid filename")
                
         # Image properties
         
        #height = len(img)
         
        width = len(img[0])
                
        points = line.getPoints(img)

        strictBounds = line.getStrictFit3(points, rowNumList[i], width)  # MSE Minimization Variation
        
        firstLine = strictBounds[0]
        lastLine = strictBounds[1]

        bandwidth = (lastLine - firstLine)/(rowNumList[i]-1)
        drawImg2(points, firstLine,rowNumList[i],bandwidth,drawable)
        
        MSEArr.append(strictBounds[3])
        MSE.append([strictBounds[2]])
    
    #print(MSE)
    #print(MSEArr)
    writeTofile(MSEArr,"mse_arr.csv",toCSV)
    writeTofile(MSE,"totalMSE.csv",toCSV)
    
    

if __name__ == '__main__':   
    main2()
    pass