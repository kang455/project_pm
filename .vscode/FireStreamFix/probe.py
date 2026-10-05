import json,sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
model=YOLO('../Tools/YoloBridge/fire.pt')
im=Image.open('FireStreamFix/flame.jpg')
rows=[]
for size in [640,960,1280]:
 r=model.predict(im,imgsz=size,conf=.05,device='cpu',verbose=False)[0]
 rows.append(dict(size=size,boxes=[dict(name=r.names[int(c)],score=float(s),box=b) for b,c,s in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist(),r.boxes.conf.tolist())]))
print(json.dumps(rows));Path('FireStreamFix/probe.json').write_text(json.dumps(rows))
