import cv2,json
from pathlib import Path
def evidence(image, box):
 x1,y1,x2,y2=map(int,box);crop=image[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
 if crop.size==0:return {}
 hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
 mask=cv2.inRange(hsv,(0,75,100),(40,255,255))
 n,labels,stats,cents=cv2.connectedComponentsWithStats(mask)
 areas=stats[1:,cv2.CC_STAT_AREA]
 return {'fill':round(float((mask>0).mean()),4),'components':int((areas>=3).sum()),'largest_fraction':round(float(areas.max()/crop.shape[0]/crop.shape[1]),4) if len(areas) else 0}
rows=[]
for f in list(Path('FireAudit').glob('view_*.jpg.before.json'))+list(Path('SceneSafetyValidation').glob('Comparison_*.jpg.json')):
 r=json.loads(f.read_text(encoding='utf-8-sig'));im=cv2.imread(str(f).replace('.before.json','').replace('.json',''))
 for d in r.get('raw_detections',r['detections']):
  if d['model_id']=='fire' and d['class_name']=='fire':rows.append({'image':str(f),'confidence':d['confidence'],'box':d['xyxy'],**evidence(im,d['xyxy'])})
print(json.dumps(rows,indent=2));Path('DetectionImprove/fire_features.json').write_text(json.dumps(rows,indent=2))
