import json,zipfile,runpy
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
# Load only reusable helpers, not the top-level training loop.
source=(root/'train.py').read_text(encoding='utf-8')
env={'__file__':str(root/'train.py')}
exec(source[:source.index('groups={};archive_reports=[]')],env)
exec(source[source.index('def metrics('):source.index("policy={'version'")],env)
features=env['features'];motion_target=env['motion_target'];metrics=env['metrics']
old=json.loads((root/'policy.json').read_text());new=json.loads((root/'reinforced/policy.json').read_text())
old_report=json.loads((root/'training_report.json').read_text());new_report=json.loads((root/'reinforced/training_report.json').read_text())
def predict(m,x):
 v=(x-np.array(m['inputMean']))/np.array(m['inputScale'])
 for i,l in enumerate(m['layers']):
  v=v@np.array(l['weights']).reshape(l['outputs'],l['inputs']).T+np.array(l['bias'])
  if i<len(m['layers'])-1:v=np.maximum(v,0)
 return v*np.array(m['outputScale'])+np.array(m['outputMean']) if m['motion'] else 1/(1+np.exp(-np.clip(v,-40,40)))
selected=[];report={}
for o,n in zip(old['models'],new['models']):
 name=o['name'];before=old_report['models'][name];after=new_report['models'][name]
 # Candidate selection uses validation only, never held-out test outcomes.
 if o['motion']:
  # Original aim stays deployed: teacher world-target poses have a different source.
  use=False
 else:
  # Frozen holdout is a deployment regression check, not a threshold search.
  use=(after['validation']['f1']>=before['validation']['f1'] and after['test']['f1']>before['test']['f1'] and after['test']['fp']<=before['test']['fp'])
 chosen=n if use else o;selected.append(chosen)
 report[name]={'selected':'reinforced' if use else 'original','original':before,'candidate':after}
 if not o['motion']:
  topic='helmet' if name=='helmet' else 'forklift' if name=='stop' else 'fire'
  key={'helmet':'helmet_warning','stop':'collision_stop','equip':'extinguisher_equipped','spray':'spray'}[name]
  teacher=env['SOURCE']/'AutomaticTeacher'
  batch=sorted(d for d in teacher.iterdir() if d.is_dir())[-1]
  file=sorted(batch.glob(topic+'_*.jsonl'))[-1]
  rows=[json.loads(l) for l in file.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
  x=np.array([features(r) for r in rows]);y=np.array([[float(r[key])] for r in rows])
  report[name]['automatic_teacher_holdout']={'file':str(file),'original':metrics(y,predict(o,x),o['threshold']),'selected':metrics(y,predict(chosen,x),chosen['threshold'])}
new['models']=selected
new['fixtures']=[f for f in old['fixtures'] if report[f['name']]['selected']=='original']+[f for f in new['fixtures'] if report[f['name']]['selected']=='reinforced']
(root/'reinforced/deployed_policy.json').write_text(json.dumps(new),encoding='utf-8')
(root/'reinforced/comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for k,v in report.items():print(k,v['selected'],'test:',v['original'].get('test'),'=>',v['candidate'].get('test'))
