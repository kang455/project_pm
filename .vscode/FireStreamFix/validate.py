import os,json,time
from pathlib import Path
os.environ['YOLO_CONFIG_DIR']=str(Path('DetectionImprove/.cache').resolve())
from server import Bridge
import cv2,numpy as np
c=json.loads(Path('FireStreamFix/models.json').read_text(encoding='utf-8-sig'))
for m in c['models']:m['weights']=str((Path('../Tools/YoloBridge')/m['weights']).resolve())
Path('FireStreamFix/test_models.json').write_text(json.dumps(c))
b=Bridge(Path('FireStreamFix/test_models.json'),'cpu')
settings=[dict(model_id=m['model_id'],enabled=m['model_id']=='fire') for m in c['models']]
summary=[]
for name,files in [('current',[Path('FireStreamFix/flame.jpg'),Path('FireStreamFix/flame2.jpg')]),('old_positive',sorted(Path('DetectionImprove/ForcedFlame').glob('*.jpg')))]:
 for i,p in enumerate(files):
  r=b.predict(p.read_bytes(),i+1,settings,name)
  summary.append(dict(image=str(p),fire=len(r['detections']),raw=r['raw_detections']))
for p in Path('FireAudit').glob('view_*.jpg'):
 for i in range(2):r=b.predict(p.read_bytes(),i+1,settings,str(p))
 summary.append(dict(image=str(p),fire=len(r['detections'])))
Path('FireStreamFix/validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
