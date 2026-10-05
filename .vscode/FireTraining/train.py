import os,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
os.environ['WANDB_MODE']='disabled'
import torch
from ultralytics import YOLO

def main():
 torch.set_num_threads(4)
 model=YOLO(str(ROOT.parent.parent/'Tools/YoloBridge/fire.pt'))
 print('Original model',model.task,model.names,'layers',len(model.model.model),flush=True)
 # Preserve original class IDs; fine-tune only the detector head on CPU.
 model.train(data=str(ROOT/'dataset_v3/data.yaml'),epochs=8,imgsz=416,batch=4,device='cpu',workers=0,
     freeze=len(model.model.model)-1,optimizer='AdamW',lr0=.0003,lrf=.2,
     project=str(ROOT/'runs'),name='warehouse_fire_v3',exist_ok=True,pretrained=True,
     cache=False,plots=False,amp=False,patience=6,seed=10526,deterministic=True,
     mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=.05,scale=.15,
     fliplr=.5,flipud=0,hsv_h=.015,hsv_s=.2,hsv_v=.2,verbose=False)
 best=Path(model.trainer.best)
 shutil.copy2(best,ROOT/'fire_warehouse_candidate.pt')
 print('Training finished',best,flush=True)

if __name__=='__main__':main()


