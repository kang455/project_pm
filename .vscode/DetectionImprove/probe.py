import os,json
from pathlib import Path
os.environ['YOLO_CONFIG_DIR']=str(Path(__file__).parent.resolve())
from PIL import Image
import torch
from ultralytics import YOLO
torch.set_num_threads(4)
root=Path(__file__).parent
models=Path('../Tools/YoloBridge')
person=YOLO(str(models/'forklift.pt'))
nohelmet=YOLO(str(models/'nohelmet.pt'))
images=[Path('ModelAdditionValidation/YellowHelmetPair.jpg'),Path('ModelAdditionValidation/BareheadFrontal.jpg')]
shots=Path('C:/Users/a0105/OneDrive/사진/스크린샷')
for i,name in enumerate(['스크린샷 2026-10-05 143642.png','스크린샷 2026-10-05 143633.png']):
 im=Image.open(shots/name).convert('RGB')
 im=im.crop((5,590,810,978))
 p=root/f'user_game_{i}.jpg';im.save(p);images.append(p)
rows=[]
for path in images:
 im=Image.open(path).convert('RGB')
 full=nohelmet.predict(im,conf=.25,verbose=False)[0]
 row={'image':str(path),'full':full.boxes.data.tolist(),'crops':[]}
 people=person.predict(im,conf=.2,classes=[3],verbose=False)[0]
 for box in people.boxes.xyxy.tolist()[:4]:
  x1,y1,x2,y2=box;w=x2-x1;h=y2-y1
  rect=(max(0,int(x1-.2*w)),max(0,int(y1-.18*h)),min(im.width,int(x2+.2*w)),min(im.height,int(y1+.65*h)))
  crop=im.crop(rect)
  crop.save(root/f'{path.stem}_{len(row["crops"])}.jpg')
  r=nohelmet.predict(crop,imgsz=640,conf=.25,verbose=False)[0]
  row['crops'].append({'person':box,'rect':rect,'boxes':r.boxes.data.tolist()})
 rows.append(row)
(root/'roi_probe.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows))
