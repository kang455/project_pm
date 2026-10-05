import os,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
import torch
from ultralytics import YOLO

if __name__=='__main__':
 source=ROOT/'runs/warehouse_fire_small/weights/best.pt'
 if not source.exists():raise RuntimeError('Checkpoint not available')
 if time.time()-source.stat().st_mtime<2:raise RuntimeError('Checkpoint is still being saved')
 copied=ROOT/'checkpoint_snapshot.pt';shutil.copy2(source,copied)
 model=YOLO(str(copied))
 if model.names!={0:'fire',1:'smoke'}:raise RuntimeError('Unexpected classes')
 model.save(str(ROOT/'fire_warehouse_candidate.pt'))
 print('Candidate snapshot saved:',model.names,flush=True)
