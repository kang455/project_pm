import os, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
os.environ['WANDB_MODE']='disabled'
import torch
from ultralytics import YOLO
torch.set_num_threads(4)
data=ROOT/'dataset'
for split in ('train','val','test'):
    sources=[(ROOT.parent/'SafetyTraining/dataset_v2/forklift',split)] if split!='test' else [(ROOT.parent/'SafetyTraining/final_holdout/forklift',s) for s in ('train','val','test')]
    for source, part in sources:
        for image in (source/'images'/part).glob('*.jpg'):
            name=part+'_'+image.stem
            (data/'images'/split).mkdir(parents=True,exist_ok=True)
            (data/'labels'/split).mkdir(parents=True,exist_ok=True)
            shutil.copy2(image,data/'images'/split/(name+'.jpg'))
            lines=(source/'labels'/part/(image.stem+'.txt')).read_text().splitlines()
            (data/'labels'/split/(name+'.txt')).write_text('\n'.join('0 '+line.split(' ',1)[1] for line in lines if line.split()[0]=='3'))
(data/'data.yaml').write_text('path: '+data.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n',encoding='utf-8')
model=YOLO(str(ROOT.parent/'SafetyTraining/forklift_warehouse_candidate.pt'))
model.train(data=str(data/'data.yaml'),epochs=12,imgsz=640,batch=4,device='cpu',workers=0,optimizer='AdamW',lr0=.0005,lrf=.1,project=str(ROOT/'runs'),name='person',exist_ok=True,plots=False,amp=False,patience=4,seed=61527,mosaic=0,mixup=0,degrees=4,translate=.15,scale=.25,fliplr=.5,hsv_h=.008,hsv_s=.15,hsv_v=.2,verbose=False)
shutil.copy2(model.trainer.best,ROOT/'person_warehouse.pt')
trained=YOLO(str(ROOT/'person_warehouse.pt'))
metrics=trained.val(data=str(data/'data.yaml'),split='test',imgsz=640,batch=4,device='cpu',workers=0,plots=False,conf=.25)
(ROOT/'person_report.json').write_text(json.dumps({'task':trained.task,'names':trained.names,'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'map50':float(metrics.box.map50),'test_images':188},indent=2))
