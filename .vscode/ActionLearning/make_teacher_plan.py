import zipfile,json
from pathlib import Path
p=Path.home()/'AppData/LocalLow/DefaultCompany/project_pm/SafetyDemonstrations'
plans=[]
for topic,f,key in [('helmet','안전모 경고 학습.zip','helmet_warning'),('forklift','지게차.zip','collision_stop'),('fire','파이어 학습.zip','spray')]:
 with zipfile.ZipFile(p/f) as z:
  files=sorted(n for n in z.namelist() if n.endswith('.jsonl'))[:-2]
  for n in files:
   rows=[json.loads(l) for l in z.read(n).decode('utf-8-sig').splitlines() if l.strip()]
   r=next((r for r in rows if r.get(key) and r.get('head_position') and r.get('perception')),None)
   if r:plans.append(dict(topic=topic,position=r['head_position'],rotation=r['head_rotation']))
Path(__file__).with_name('teacher_plan.json').write_text(json.dumps(dict(cases=plans)),encoding='utf-8')
print(json.dumps(dict(cases=plans)))
