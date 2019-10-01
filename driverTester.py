'''
Created on Sep 22, 2019

@author: Anik
'''

from PIL import Image
import unittest, driver, os


class TestDriver(unittest.TestCase):
        
    def test_openImg_01(self):
        
        handler = driver.File("test","test")
        
        result = handler.openImg("invalidImg.png")
        self.assertEqual(result, None, "Invalid file test error")
    
    def test_openImg_02(self):
        
        handler = driver.File("test","test")
        
        img = Image.new('RGB', (60, 30), color='red')
        img.save('testImg.png')
        
        result = handler.openImg("testImg.png")
        self.assertEqual(isinstance(result, Image.Image), True, "Valid file test error")
        
        result.close()
        os.remove("testImg.png")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
