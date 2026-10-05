import zipfile,json,collections
from pathlib import Path
root=Path(r'C:\Users\a0105\AppData\LocalLow\DefaultCompany\project_pm\SafetyDemonstrations')
out=Path(__file__).resolve().parent
for filename in ['안전모 경고 학습.zip','지게차.zip','파이어 학습.zip']:
 records=[];errors=0;files=[]
 with zipfile.ZipFile(root/filename) as z:
  for info in z.infolist():
   if not info.filename.endswith('.jsonl'):continue
   rows=[]
   for line in z.read(info).decode('utf-8-sig').splitlines():
    try: rows.append(json.loads(line))
    except ValueError:errors+=1
   records+=rows
   samples=[x for x in rows if 'observations' in x]
   files.append({'file':info.filename,'samples':len(samples),'duration':samples[-1]['time']-samples[0]['time'] if samples else 0})
 samples=[x for x in records if 'observations' in x];events=[x for x in records if x.get('kind')=='human_action']
 report={'archive':filename,'files':files,'errors':errors,'schemas':dict(collections.Counter(x.get('schema_version',0) for x in samples)),'events':dict(collections.Counter(x.get('action') for x in events)),'states':{k:sum(bool(x.get(k)) for x in samples) for k in ['spray','collision_stop','helmet_warning','collision_risk','extinguisher_equipped']},'vision':{k:sum(any(d.get('class_name')==k for d in (x.get('perception') or {}).get('detections',[])) for x in samples) for k in ['person','forklift','fire','nohelmet']},'extinguished_max':max((x.get('extinguished',0) for x in samples),default=0),'first_sample':samples[0] if samples else None,'event_times':events}
 (out/(filename+'.report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k not in ['first_sample','event_times']},ensure_ascii=True))
