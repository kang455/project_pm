import os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
os.environ['WANDB_MODE']='disabled'
import torch
from ultralytics import YOLO

def main():
 torch.set_num_threads(4)
 model=YOLO(str(ROOT/'yolo26n.pt'))
 model.train(data=str(ROOT/'dataset_v3/data.yaml'),epochs=24,imgsz=640,batch=4,device='cpu',workers=0,
     optimizer='AdamW',lr0=.001,lrf=.1,project=str(ROOT/'runs'),name='warehouse_fire_small',exist_ok=True,
     pretrained=True,cache=False,plots=False,amp=False,patience=7,seed=10526,
     deterministic=True,mosaic=0,mixup=0,copy_paste=0,degrees=4,translate=.2,scale=.25,
     fliplr=.5,flipud=0,hsv_h=.015,hsv_s=.2,hsv_v=.2,warmup_bias_lr=.001,verbose=False)
 shutil.copy2(model.trainer.best,ROOT/'fire_warehouse_candidate.pt')
 print('Full fine-tuning finished:',model.trainer.best,flush=True)

if __name__=='__main__':main()
