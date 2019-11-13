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
#lower the threthold for cluter fileter filterClusters(img, 3) 

def imgProcess():

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
    
def main2():
    toCSV = True
    #imgProcess()
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
    
    for i in range(file_count):
        
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
         
                
        #stripWidth = round(width / rowNumList[i])
                
        points = line.getPoints(img)
        #print(rowNumList[i])
        # strictBounds = line.getStrictFit(points, rows, height, width)
        strictBounds = line.getStrictFit3(points, rowNumList[i], width)  # MSE Minimization Variation
        
        #print(strictBounds[2])
        #print(strictBounds[3])
        MSEArr.append(strictBounds[3])
        MSE.append([strictBounds[2]])
    
    
    writeTofile(MSEArr,"mse_arr.csv",toCSV)
    writeTofile(MSE,"totalMSE.csv",toCSV)
    
    

if __name__ == '__main__':   
    main2()
    pass