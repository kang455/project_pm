import csv,subprocess,time,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
seen=0
while True:
 try:
  rows=list(csv.DictReader((ROOT/'runs/warehouse_fire_small/results.csv').open()))
  last=rows[-1];epoch=int(last['epoch'])
 except (OSError,IndexError,ValueError):time.sleep(3);continue
 if epoch==seen:time.sleep(3);continue
 seen=epoch
 print('Completed epoch',epoch,'validation recall',last['metrics/recall(B)'],'mAP50',last['metrics/mAP50(B)'],flush=True)
 finished='Full fine-tuning finished:' in (ROOT/'train_small.log').read_text(errors='replace')
 if not finished and (epoch<6 or float(last['metrics/recall(B)'])<.8 or float(last['metrics/mAP50(B)'])<.85):
  time.sleep(3);continue
 time.sleep(3)
 result=subprocess.run([sys.executable,str(ROOT/'candidate_from_checkpoint.py')])
 if result.returncode:continue
 result=subprocess.run([sys.executable,str(ROOT/'calibrate.py')])
 if result.returncode:
  if finished:raise RuntimeError('Full model failed validation')
  continue
 for script in ['evaluate.py','prepare_deployment.py']:
  subprocess.run([sys.executable,str(ROOT/script)],check=True)
 (ROOT/'ready.json').write_text(json.dumps({'selected_after_epoch':epoch,'validation_selected':True}))
 print('Validated candidate ready for deployment',flush=True)
 break
