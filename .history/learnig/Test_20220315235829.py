import os
import shutil

imagePath = '/data2/guobo/01_SHIPRSDET/ShipDetv2/Voc/train_set/images/'
xmlsPath = '/data2/guobo/01_SHIPRSDET/ShipDetv2/Voc/train_set/xmls-No-object/'
imagePath_Move='/data2/guobo/01_SHIPRSDET/ShipDetv2/Voc/train_set/image-No-object/'

imageList = os.listdir(imagePath)
xmlList = os.listdir(xmlsPath)

for image in imageList:
    name  = image.split('.')[0] + '.xml'
    if name in xmlList:
        print(name)
        # shutil.move(os.path.join(imagePath,image),imagePath_Move)
        
    