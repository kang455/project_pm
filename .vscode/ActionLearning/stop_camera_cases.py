import json,zipfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
policy=json.loads((root/'policy.json').read_text())
model=next(x for x in policy['models'] if x['name']=='stop')
cases=[]
with zipfile.ZipFile(Path(r'C:\Users\a0105\AppData\LocalLow\DefaultCompany\project_pm\SafetyDemonstrations\지게차.zip')) as z:
 for name in sorted(z.namelist())[:5]:
  rows=[json.loads(line) for line in z.read(name).decode('utf-8-sig').splitlines() if line.strip()]
  stops=[r['time'] for r in rows if r.get('action')=='human_forklift_stop']
  if not stops:continue
  candidates=[r for r in rows if 'head_position' in r and r.get('collision_stop') and r.get('perception')]
  if not candidates:continue
  r=min(candidates,key=lambda r:abs(r['time']-stops[0]))
  cases.append({'file':name,'time':stops[0],'position':r['head_position'],'rotation':r['head_rotation']})
(root/'stop_camera_cases.json').write_text(json.dumps({'cases':cases}),encoding='utf-8')
print(json.dumps(cases))
