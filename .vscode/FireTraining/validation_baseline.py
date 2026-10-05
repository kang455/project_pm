import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
import torch
from evaluate import evaluate,metrics
if __name__=='__main__':
 torch.set_num_threads(4)
 rows=evaluate(ROOT.parent.parent/'Tools/YoloBridge/fire.pt',sorted((ROOT/'dataset_v3/images/val').glob('*.jpg')))
 result=metrics(rows,.25)
 (ROOT/'validation_baseline.json').write_text(json.dumps(dict(metrics=result,rows=rows),indent=2))
 print(result,flush=True)
