exec(open('DetectionImprove/probe.py',encoding='utf-8').read().split('rows=[]')[0])
fire=YOLO(str(models/'fire.pt'))
out=[]
for p in sorted(Path('DetectionImprove/ForcedFlame').glob('*.jpg')):
 r=fire.predict(Image.open(p).convert('RGB'),imgsz=960,conf=.25,verbose=False)[0]
 out.append({'image':p.name,'boxes':r.boxes.data.tolist()})
print(json.dumps(out));Path('DetectionImprove/size_probe.json').write_text(json.dumps(out,indent=2))
