'''
Created on Sep 18, 2019

@author: Anik
'''

from PIL import Image, ImageEnhance
import colorsys

# Specify the file name here
FILENAME = "image01.png"


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


def openImg(fileName):
    
    img = None
    
    try:
        img = Image.open(fileName)
    except FileNotFoundError:
        print ("Invalid filename")
    
    return img


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


def main():
    
    # Open image
    img = openImg(FILENAME)
    
    # Convert image to RGB
    rgb = convertToRGB(img)
    
    # Adjust image level
    levelledImg = adjustLevel(rgb, 100, 255, 9.99)
    
    # Convert to grayscale and binarize the image
    grayImg = convertToGreyscale(levelledImg)
    binImg = binarizeImg(grayImg)
    
    # Save processed image(s)
    rgb.save("rgb.png")
    levelledImg.save("levelledImg.png")
    grayImg.save("grayImg.png")
    binImg.save("binImg.png")
    
    print("Program successfully terminated.")
    return


if __name__ == '__main__':
    
    main()
    pass
