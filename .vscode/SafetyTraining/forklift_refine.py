import os,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache');os.environ['WANDB_MODE']='disabled'
import torch
from ultralytics import YOLO
torch.set_num_threads(4)
initial=ROOT/'forklift_initial.pt'
if not initial.exists():shutil.copy2(ROOT/'forklift_warehouse_candidate.pt',initial)
model=YOLO(str(initial))
model.train(data=str(ROOT/'dataset_v2/forklift/data.yaml'),epochs=10,imgsz=640,batch=4,
    device='cpu',workers=0,optimizer='AdamW',lr0=.0004,lrf=.1,project=str(ROOT/'runs'),
    name='forklift_v2',exist_ok=True,pretrained=True,cache=False,plots=False,amp=False,
    patience=4,seed=31527,deterministic=True,mosaic=0,mixup=0,copy_paste=0,
    degrees=4,translate=.18,scale=.3,fliplr=.5,flipud=0,hsv_h=.008,hsv_s=.15,
    hsv_v=.2,warmup_bias_lr=.0004,verbose=False)
shutil.copy2(model.trainer.best,ROOT/'forklift_warehouse_candidate.pt')
(ROOT/'forklift_refined.json').write_text(json.dumps({'best':str(model.trainer.best),'epochs':model.trainer.epoch+1}))
print('Forklift refinement complete',flush=True)
