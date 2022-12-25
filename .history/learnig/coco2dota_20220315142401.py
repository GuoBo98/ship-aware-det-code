from unicodedata import category
from pycocotools.coco import COCO 
import os
from tqdm import trange

if __name__ == '__main__':
    
    annFile = '/data2/guobo/01_SHIPRSDET/new-COCO-Format/annotations/shipRS_trainset.json'
    disPath = '/data2/guobo/01_SHIPRSDET/dota/trainset'
    
    cocoGT = COCO(annFile)
    
    for i in trange(len(cocoGT.imgs)):
        pointobbs = []
        img_name = cocoGT.imgs[i+1]['file_name']
        for j in range(len(cocoGT.anns)):
            pointobb = []
            ann = cocoGT.anns[j+1]
            if ann['image_id'] == i+1:
                pointobb = [float(int(point)) for point in ann['pointobb']]
                '''convert name A B to A-B'''
                category_name = cocoGT.cats[ann['category_id']]['name']
                category_name_split = category_name.split(' ')
                if len(category_name_split) == 2 :
                    category_name = category_name_split[0] + '-' + category_name_split[1]
                elif len(category_name_split) == 3 :
                    category_name = category_name_split[0] + '-' + category_name_split[1] + '-' + category_name_split[2]
                elif len(category_name_split) == 1 :
                    pass
                else:
                    print('error')
                        
                pointobb.append(category_name)
                pointobb.append(0)
                pointobbs.append(pointobb)
        with open(os.path.join(disPath,img_name[0:-4] + '.txt'),'w') as f:
            for pointobb in pointobbs:                
                for item in pointobb:
                    f.write(str(item) + ' ')
                f.write('\n')
            
                

        
                
        
    
    