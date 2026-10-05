import os,json
from pathlib import Path
os.environ['YOLO_CONFIG_DIR']=str(Path('DetectionImprove/.cache').resolve())
from ultralytics import YOLO
from PIL import Image
import torch
torch.set_num_threads(4)
m=YOLO('../Tools/YoloBridge/fire.pt');out=[]
for p in [Path('FireStreamFix/flame.jpg')]+list(Path('FireAudit').glob('view_*.jpg')):
 im=Image.open(p);r=m.predict(im,imgsz=640,conf=.25,verbose=False)[0]
 for box,c in zip(r.boxes.xyxy.tolist(),r.boxes.cls.tolist()):
  if r.names[int(c)]!='fire':continue
  x,y,z,w=box;dx=(z-x)*.5;dy=(w-y)*.5
  crop=im.crop((max(0,int(x-dx)),max(0,int(y-dy)),min(im.width,int(z+dx)),min(im.height,int(w+dy))))
  q=m.predict(crop,imgsz=640,conf=.1,verbose=False)[0]
  out.append(dict(image=str(p),box=box,crop=[dict(name=q.names[int(c)],score=float(s)) for c,s in zip(q.boxes.cls,q.boxes.conf)]))
print(json.dumps(out));Path('FireStreamFix/crops.json').write_text(json.dumps(out,indent=2))
