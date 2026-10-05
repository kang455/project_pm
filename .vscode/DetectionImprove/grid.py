exec(open('DetectionImprove/probe.py',encoding='utf-8').read().split('rows=[]')[0])
for path in images[:1]+images[2:3]:
 im=Image.open(path).convert('RGB'); people=person.predict(im,conf=.2,classes=[3],verbose=False)[0]
 for box in people.boxes.xyxy.tolist()[:2]:
  x1,y1,x2,y2=box;w=x2-x1;h=y2-y1
  for top in [.35,.45,.55]:
   rect=(max(0,int(x1-.25*w)),max(0,int(y1-.25*h)),min(im.width,int(x2+.25*w)),min(im.height,int(y1+top*h)))
   r=nohelmet.predict(im.crop(rect),imgsz=640,conf=.25,verbose=False)[0]
   print(path.name,box,top,rect,r.boxes.data.tolist())
