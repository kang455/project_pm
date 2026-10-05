import os,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
import torch
from ultralytics import YOLO
from evaluate import predictions,metric
torch.set_num_threads(1)
report={}
for kind,cid in [('helmet',1),('nohelmet',0),('forklift',2)]:
    run='forklift_v2' if kind=='forklift' and (ROOT/'runs/forklift_v2/weights/best.pt').exists() else kind
    path=ROOT/'runs'/run/'weights/best.pt'
    if not path.exists() or time.time()-path.stat().st_mtime<2:continue
    snapshot=ROOT/(kind+'_snapshot.pt');shutil.copy2(path,snapshot)
    model=YOLO(str(snapshot))
    dataset=ROOT/'dataset_v2' if run=='forklift_v2' else ROOT/'dataset'
    rows=predictions(model,kind,'val',dataset)
    options=[(metric(rows,cid,t),t) for t in [.25,.35,.45,.55,.65]]
    passing=[z for z in options if z[0]['precision']>=.95 and z[0]['recall']>=.85]
    measured,threshold=max(passing or options,key=lambda z:(2*z[0]['precision']*z[0]['recall']/max(.001,z[0]['precision']+z[0]['recall']),z[0]['precision'],-z[1]))
    report[kind]=dict(metric=measured,threshold=threshold,epoch=model.ckpt.get('epoch'),
        ready=measured['precision']>=.95 and measured['recall']>=.85,
        person=metric(rows,3,threshold) if kind=='forklift' else None)
    if report[kind]['ready']:
        shutil.copy2(snapshot,ROOT/(kind+'_warehouse_candidate.pt'))
(ROOT/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
