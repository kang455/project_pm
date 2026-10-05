from pathlib import Path
root=Path(__file__).resolve().parent
code=(root/'train.py').read_text(encoding='utf-8')
code=code.replace('ROOT=Path(__file__).resolve().parent','BASE=Path(__file__).resolve().parent\nROOT=BASE/"reinforced"\nROOT.mkdir(exist_ok=True)')
marker='def dataset(topic,split,label,motion=False):'
insertion='''# Only first four automated sessions enter fitting; fifth remains a live holdout.
teacher_root=SOURCE/'AutomaticTeacher'
teacher_sessions=[]
if teacher_root.exists():
 batches=sorted(d for d in teacher_root.iterdir() if d.is_dir())
 if batches:
  for topic in groups:
   files=sorted(batches[-1].glob(topic+'_*.jsonl'))
   for file in files[:-1]:
    rows=[json.loads(l) for l in file.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    rows=[r for r in rows if r.get('control_mode')=='scripted_teacher' and r.get('perception')]
    groups[topic]['train'].append((str(file),rows))
    teacher_sessions.append(str(file))
'''
code=code.replace(marker,insertion+'\n'+marker)
code=code.replace('mean=x.mean(0);std=', '''# Small feature noise only on fitting data; never add duplicates to evaluation.
 if not is_motion:
  copies=[x];targets=[y]
  for repeat in range(4):
   augmented=x.copy()
   for offset in [0,6,12,18]:
    present=x[:,offset]>0
    augmented[present,offset+1]=np.clip(x[present,offset+1]+np.random.normal(0,.025,present.sum()),0,1)
    augmented[present,offset+2:offset+6]=np.clip(x[present,offset+2:offset+6]+np.random.normal(0,.008,(present.sum(),4)),0,1)
   copies.append(augmented);targets.append(y)
  x=np.concatenate(copies);y=np.concatenate(targets)
 mean=x.mean(0);std=''')
code=code.replace("report={'archives':archive_reports,'models':{}}", "report={'archives':archive_reports,'teacher_training_files':teacher_sessions,'models':{}}")
(root/'train_reinforced.py').write_text(code,encoding='utf-8')
print('Prepared augmented behavioral cloning; original training outputs preserved.')
