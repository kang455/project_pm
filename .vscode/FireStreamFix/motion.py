import cv2,numpy as np
a=cv2.imread('FireStreamFix/flame.jpg')[232:317,454:498]
b=cv2.imread('FireStreamFix/flame2.jpg')[232:317,454:498]
d=np.abs(a.astype(float)-b).mean(2)
mask=cv2.inRange(cv2.cvtColor(a,cv2.COLOR_BGR2HSV),(0,75,100),(40,255,255))>0
print('mean',d.mean(),'warm >8',float((d[mask]>8).mean()),'warm >25',float((d[mask]>25).mean()))
