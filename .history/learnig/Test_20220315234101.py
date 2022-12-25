import os
import shutil

imagePath = ''
xmlsPath = ''
imagePath_Move=''

imageList = os.listdir(imagePath)
xmlList = os.listdir(xmlsPath)

for image in imageList:
    name  = image.split()[0] + '.xml'
    if name in xmlList:
        shutil.move(os.path.join(imagePath,image),imagePath_Move)
        
    