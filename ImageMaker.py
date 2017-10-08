import cv2  
import numpy as np
from random import randint
import json
# empty image
import time
start_time = time.time()

lister={}
Dir="/Volumes/wd/Data/Test2/"
train=False

subdir=0
jsonName=0
recordNum=0
if train:
	subdir='Rec/'		#train
	jsonName='tag.json'
	recordNum=10000
else:
	subdir='TestRec/' 	#test
	jsonName='TestTag.json'
	recordNum=300





def generateImg(i):
	raniX=randint(20, 80)
	raniY=randint(20, 80)
	raniR=randint(5, 18)
	img = np.zeros((100, 100,3), np.uint8)
	cv2.circle(img, (raniX, raniY), raniR, (101,67,254),-1)
	cv2.imwrite(Dir+subdir+`i`+'.jpg',img)
	lister[`i`+'_jpg']={
	"filename":`i`+'.jpg',
	"xmins":(raniX-raniR-1),
	"xmaxs":(raniX+raniR+1),
	"ymins":(raniY-raniR-1),
	"ymaxs":(raniY+raniR+1)
	}
	# cv2.imshow("shape", img)

for i in range(recordNum):
	generateImg(i)

file = open(Dir+jsonName,"w")
file.truncate()
file.write(json.dumps(lister))
file.close()  
print("--- %s seconds ---" % (time.time() - start_time))

# cv2.destroyAllWindows()