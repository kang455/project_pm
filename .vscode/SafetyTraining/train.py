import os, json, shutil, argparse, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
os.environ['WANDB_MODE']='disabled'
import torch
from ultralytics import YOLO

def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['helmet','nohelmet','forklift']);args=parser.parse_args()
    completed=ROOT/(args.model+'_completed.json')
    if completed.exists():
        print('Already completed '+args.model,flush=True);return
    lock=ROOT/(args.model+'.lock')
    try:
        fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    except FileExistsError:
        while lock.exists() and not completed.exists():time.sleep(5)
        if completed.exists():return
        raise RuntimeError('Parallel training failed '+args.model)
    torch.set_num_threads(4)
    model=YOLO(str(ROOT.parent/'FireTraining/yolo26n.pt'))
    model.train(data=str(ROOT/'dataset'/args.model/'data.yaml'),epochs=18,imgsz=640,batch=4,
        device='cpu',workers=0,optimizer='AdamW',lr0=.001,lrf=.1,
        project=str(ROOT/'runs'),name=args.model,exist_ok=True,pretrained=True,cache=False,
        plots=False,amp=False,patience=5,seed=10527,deterministic=True,
        mosaic=0,mixup=0,copy_paste=0,degrees=4,translate=.15,scale=.25,fliplr=.5,
        flipud=0,hsv_h=.008,hsv_s=.15,hsv_v=.2,warmup_bias_lr=.001,verbose=False)
    shutil.copy2(model.trainer.best,ROOT/(args.model+'_warehouse_candidate.pt'))
    completed.write_text(json.dumps({'model':args.model,'best':str(model.trainer.best),'finished':time.time()}))
    lock.unlink()
    print('COMPLETED '+args.model,flush=True)

if __name__=='__main__':main()
