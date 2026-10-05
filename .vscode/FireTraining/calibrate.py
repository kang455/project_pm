import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
import torch
from evaluate import evaluate,metrics

if __name__=='__main__':
 torch.set_num_threads(4)
 rows=evaluate(ROOT/'fire_warehouse_candidate.pt',sorted((ROOT/'dataset_v3/images/val').glob('*.jpg')))
 options=[metrics(rows,t) for t in [.25,.35,.45,.55,.65,.75]]
 # Select using validation only; test is held out until after this decision.
 baseline=json.loads((ROOT/'validation_baseline.json').read_text())['metrics']
 acceptable=[r for r in options if r['false_positive_images']==0 and r['detected_positive_images']>=max(baseline['detected_positive_images'],.6*r['positive_images'])]
 print(options,flush=True)
 (ROOT/'calibration_attempt.json').write_text(json.dumps(dict(options=options,rows=rows),indent=2))
 if not acceptable:raise RuntimeError('No threshold met validation acceptance; candidate must not replace production')
 chosen=acceptable[0]
 (ROOT/'calibration.json').write_text(json.dumps(dict(chosen=chosen,options=options,rows=rows),indent=2))
 print('Validation-selected threshold',chosen,flush=True)



