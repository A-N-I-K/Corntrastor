'''
Created on Sep 18, 2019

@author: Anik
'''

from os import listdir
from os.path import isfile, join
from PIL import Image, ImageEnhance
import colorsys, os

INPUTFOLDERNAME = "raw_images"
OUTPUTFOLDERNAME = "processed_images"


class Level(object):
    
    def __init__(self, minv, maxv, gamma):
        
        self.minv = minv / 255.0
        self.maxv = maxv / 255.0
        self._interval = self.maxv - self.minv
        self._invgamma = 1.0 / gamma

    def newLevel(self, value):
        
        if value <= self.minv: return 0.0
        if value >= self.maxv: return 1.0
        
        return ((value - self.minv) / self._interval) ** self._invgamma

    def convertAndLevel(self, band_values):
        
        h, s, v = colorsys.rgb_to_hsv(*(i / 255.0 for i in band_values))
        new_v = self.newLevel(v)
        return tuple(int(255 * i)
                for i
                in colorsys.hsv_to_rgb(h, s, new_v))
        

class File(object):
    
    def __init__(self, inF, outF):
        
        self.inF = inF
        self.outF = outF
    
    def getFilenames(self):
        
        if not os.path.isdir(self.inF):
            os.makedirs(self.inF)
            
        return [f for f in listdir(self.inF) if isfile(join(self.inF, f))]
    
    def getImages(self):
        
        filenames = self.getFilenames()
        imageList = []
        
        for file in filenames:
            imageList.append(self.openImg(self.inF + "/" + file))
        
        return imageList
    
    def setImages(self, imageList):
        
        if not os.path.isdir(self.outF):
            os.makedirs(self.outF)
        
        for i, img in enumerate(imageList):
            img.save(self.outF + "/{:03d}.png".format(i))
    
    def openImg(self, fileName):
        
        img = None
        
        try:
            img = Image.open(fileName)
        except FileNotFoundError:
            print ("Invalid filename")
        
        return img
    
    
class Trim(object):
    
    def __init__(self, maxWhiteThresh, rowHeight):
        
        self.maxWhiteThresh = maxWhiteThresh
        self.rowHeight = rowHeight
    
    def naiveTrim(self, img, topTrim, bottomTrim):
    
        width, height = img.size
        
        left = 0
        right = width
        top = int(height * topTrim)
        bottom = int(height * bottomTrim)
        
        return img.crop((left, top, right, bottom))
    
    def smartTrim(self, img):
        
        width = img.size[0]
        
        left = 0
        right = width
        top = self.getTop(img)
        bottom = self.getBottom(img)
        
        return img.crop((left, top, right, bottom))
    
    def getTop(self, img):
        
        width, height = img.size
        
        left = 0
        right = width
        
        for i in range (int(height / (2 * self.rowHeight)) - 1):
            rowImg = img.crop((left, i * self.rowHeight, right, (i + 1) * self.rowHeight))
            pixels = rowImg.getdata()
            
            whiteThresh = 50
            count = 0
            
            for pixel in pixels:
                if pixel > whiteThresh:
                    count += 1
                    
            n = len(pixels)
            
            if (count / float(n)) < self.maxWhiteThresh:
                return ((i + 1) * self.rowHeight)
            
        return  (int(height / 2) - 1)
    
    def getBottom(self, img):
        
        width, height = img.size
        
        left = 0
        right = width
        
        for i in range (int(height / (2 * self.rowHeight)) - 1):
            rowImg = img.crop((left, height - ((i + 1) * self.rowHeight), right, height - (i * self.rowHeight)))
            pixels = rowImg.getdata()
            
            whiteThresh = 50
            count = 0
            
            for pixel in pixels:
                if pixel > whiteThresh:
                    count += 1
                    
            n = len(pixels)
            
            if (count / float(n)) < self.maxWhiteThresh:
                return (height - ((i + 1) * self.rowHeight))
            
        return  (int(height / 2) + 1)


def convertToRGB(img):
    
    img.load()
    
    rgb = Image.new("RGB", img.size, (255, 255, 255))
    rgb.paste(img, mask=img.split()[3])
    
    return rgb

    
def adjustLevel(img, minv=0, maxv=255, gamma=1.0):

    if img.mode != "RGB":
        raise ValueError("Image not in RGB mode")

    newImg = img.copy()

    leveller = Level(minv, maxv, gamma)
    levelled_data = [
        leveller.convertAndLevel(data)
        for data in img.getdata()]
    newImg.putdata(levelled_data)
    
    return newImg


def convertToGreyscale(img):
    
    return ImageEnhance.Contrast(img).enhance(50.0)


def binarizeImg(img):
    
    return img.convert('1')


def bulkProcess(imageList):
    
    processedImageList = []
    
    for i, img in enumerate(imageList):
        
        # Convert image to RGB
        rgb = convertToRGB(img)
        
        # Adjust image level
        levelledImg = adjustLevel(rgb, 100, 255, 9.99)
        
        # Convert to greyscale
        grayImg = convertToGreyscale(levelledImg)
        
        # Binarize image
        binImg = binarizeImg(grayImg)
        
        # Initialize trimmer
        trimmer = Trim(0.1, 1)
        
        # Trim image (Naive)
        trimmedImg = trimmer.smartTrim(binImg)
        
        # Update processed image list
        processedImageList.append(trimmedImg)
        
        # Display status
        print("Image {:03d}.png processing completed".format(i))
    
    return processedImageList


def main():
    
    # Initialize handler
    handler = File(INPUTFOLDERNAME, OUTPUTFOLDERNAME)
    
    # Get images
    imageList = handler.getImages()
    
    # Process images
    processedImageList = bulkProcess(imageList)
    
    # Save images
    handler.setImages(processedImageList)
    
    print("Program successfully terminated")
    return


if __name__ == '__main__':
    
    main()
    pass
