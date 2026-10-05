import os,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
import torch
from ultralytics import YOLO
from PIL import Image

def iou(a,b):
 overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
 return overlap/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)

def evaluate(path,files):
 m=YOLO(str(path));rows=[]
 for p in files:
  im=Image.open(p);label=p.parents[2]/'labels'/p.parent.name/(p.stem+'.txt')
  targets=[]
  if label.exists():
   for line in label.read_text().splitlines():
    c,x,y,w,h=map(float,line.split());targets.append([(x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height])
  result=m.predict(im,imgsz=640,conf=.05,device='cpu',verbose=False)[0]
  candidates=[dict(score=float(s),box=b) for b,c,s in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist()) if result.names[int(c)]=='fire']
  rows.append(dict(image=str(p),targets=targets,candidates=candidates,ms=result.speed['inference']))
 return rows

def metrics(rows,threshold):
 positives=[r for r in rows if r['targets']];negatives=[r for r in rows if not r['targets']]
 hits=sum(any(c['score']>=threshold and any(iou(c['box'],t)>=.3 for t in r['targets']) for c in r['candidates']) for r in positives)
 false=sum(any(c['score']>=threshold for c in r['candidates']) for r in negatives)
 return dict(threshold=threshold,positive_images=len(positives),detected_positive_images=hits,negative_images=len(negatives),false_positive_images=false,mean_ms=sum(r['ms'] for r in rows)/len(rows))

if __name__=='__main__':
 torch.set_num_threads(4)
 test=sorted((ROOT/'dataset_v3/images/test').glob('*.jpg'))
 test+=sorted((ROOT.parent/'FireAudit').glob('view_*.jpg'))
 test+=sorted((ROOT/'live_holdout/images/train').glob('*.jpg'))
 # Original false positives from user screenshots are kept outside training.
 test+=sorted((ROOT/'external_negatives').glob('*.jpg'))
 outputs={}
 for key,path in [('original',ROOT.parent.parent/'Tools/YoloBridge/fire.pt'),('trained',ROOT/'fire_warehouse_candidate.pt')]:
  rows=evaluate(path,test)
  outputs[key]={'metrics':[metrics(rows,t) for t in [.25,.35,.45,.55,.65]],'rows':rows}
  (ROOT/'evaluation.json').write_text(json.dumps(outputs,indent=2))
  print(key,outputs[key]['metrics'],flush=True)


