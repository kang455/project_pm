import json,zipfile,math,hashlib,random,collections
from pathlib import Path
import numpy as np
import torch
from torch import nn
BASE=Path(__file__).resolve().parent
ROOT=BASE/"reinforced"
ROOT.mkdir(exist_ok=True)
SOURCE=Path(r'C:\Users\a0105\AppData\LocalLow\DefaultCompany\project_pm\SafetyDemonstrations')
torch.set_num_threads(4);torch.manual_seed(105);np.random.seed(105)
names=['person','forklift','fire','nohelmet']
feature_names=[n+'_'+v for n in names for v in ['present','confidence','cx','cy','width','height']]+['image_risk','image_gap']
def features(sample):
 p=sample.get('perception') or {};ds=p.get('detections',[]);w=p.get('width',960);h=p.get('height',540);f=[]
 for name in names:
  candidates=[d for d in ds if d.get('class_name')==name and len(d.get('xyxy',[]))==4 and (name!='nohelmet' or len(d.get('person_xyxy',[]))==4)]
  if not candidates:f += [0]*6;continue
  d=max(candidates,key=lambda d:d.get('confidence',0));x1,y1,x2,y2=d['xyxy']
  f += [1,d['confidence'],(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h]
 obs=sample.get('observations',[]);gap=obs[5] if len(obs)>5 else -1
 f += [float(sample.get('collision_risk',False)),max(-1,min(1,gap))]
 return np.asarray(f,dtype=np.float32)
def qrotate(q,v):
 q=np.array([q['x'],q['y'],q['z'],q['w']],float);q/=max(1e-9,np.linalg.norm(q));return v+2*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v)
def motion_target(sample,x):
 head=sample['head_rotation'];inverse={'x':-head['x'],'y':-head['y'],'z':-head['z'],'w':head['w']}
 pos=np.array([sample['right_position'][k]-sample['head_position'][k] for k in ['x','y','z']]);local=qrotate(inverse,pos)
 direction=qrotate(inverse,qrotate(sample['right_rotation'],np.array([0.,0.,1.])))
 ray=np.array([(2*x[14]-1)*16/9,1-2*x[15],1.]);ray/=np.linalg.norm(ray)
 return np.r_[local,direction-ray].astype(np.float32)
groups={};archive_reports=[]
for topic,filename in [('helmet','안전모 경고 학습.zip'),('forklift','지게차.zip'),('fire','파이어 학습.zip')]:
 files=[]
 with zipfile.ZipFile(SOURCE/filename) as z:
  for name in sorted(z.namelist()):
   if not name.endswith('.jsonl'):continue
   rows=[json.loads(line) for line in z.read(name).decode('utf-8-sig').splitlines() if line.strip()]
   samples=[r for r in rows if 'observations' in r and r.get('schema_version')==2 and r.get('control_mode')=='human' and r.get('perception')]
   if not samples:continue
   files.append((name,samples))
 groups[topic]={'train':files[:-2],'val':files[-2:-1],'test':files[-1:]}
 archive_reports.append({'topic':topic,'sha256':hashlib.sha256((SOURCE/filename).read_bytes()).hexdigest(),'splits':{k:[n for n,s in v] for k,v in groups[topic].items()}})
# Only first four automated sessions enter fitting; fifth remains a live holdout.
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

def dataset(topic,split,label,motion=False):
 xs=[];ys=[]
 for name,rows in groups[topic][split]:
  for row in rows:
   x=features(row)
   if motion and not(row.get('spray') and row.get('right_controller_active') and x[12]):continue
   if not row.get('right_controller_active',True):continue
   y=motion_target(row,x) if motion else [float(row.get(label,False))]
   if np.isfinite(x).all() and np.isfinite(y).all():xs.append(x);ys.append(y)
 return np.array(xs,np.float32),np.array(ys,np.float32)
def export(model,mean,std,ym=None,ys=None):
 return {'inputMean':mean.tolist(),'inputScale':std.tolist(),'outputMean':[] if ym is None else ym.tolist(),'outputScale':[] if ys is None else ys.tolist(),'layers':[{'inputs':layer.in_features,'outputs':layer.out_features,'weights':layer.weight.detach().numpy().reshape(-1).tolist(),'bias':layer.bias.detach().numpy().tolist()} for layer in model if isinstance(layer,nn.Linear)]}
def metrics(y,p,threshold):
 a=y[:,0]>.5;b=p[:,0]>=threshold;tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum());return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}
policy={'version':1,'featureNames':feature_names,'models':[]};report={'archives':archive_reports,'teacher_training_files':teacher_sessions,'models':{}}
for name,topic,label,is_motion in [('helmet','helmet','helmet_warning',False),('stop','forklift','collision_stop',False),('equip','fire','extinguisher_equipped',False),('spray','fire','spray',False),('aim','fire','spray',True)]:
 splits={s:dataset(topic,s,label,is_motion) for s in ['train','val','test']}
 x,y=splits['train'];vx,vy=splits['val'];tx,ty=splits['test']
 if len(x)<10 or len(vx)==0 or len(tx)==0:raise RuntimeError('Insufficient independent samples: '+name)
 # Small feature noise only on fitting data; never add duplicates to evaluation.
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
 mean=x.mean(0);std=np.maximum(x.std(0),.1);ym=y.mean(0) if is_motion else None;ys=np.maximum(y.std(0),.05) if is_motion else None
 net=nn.Sequential(nn.Linear(26,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,y.shape[1]))
 optim=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.02)
 xt=torch.tensor((x-mean)/std);yt=torch.tensor((y-ym)/ys if is_motion else y)
 vt=torch.tensor((vx-mean)/std);vyt=torch.tensor((vy-ym)/ys if is_motion else vy)
 lossfn=nn.SmoothL1Loss() if is_motion else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-y.sum())/max(1,float(y.sum()))]).clamp(1,8))
 best=math.inf;wait=0;best_state=None
 for epoch in range(500):
  net.train();optim.zero_grad();loss=lossfn(net(xt),yt);loss.backward();optim.step();net.eval()
  with torch.no_grad():validation=float(lossfn(net(vt),vyt))
  if validation<best-1e-5:best=validation;wait=0;best_state={k:v.clone() for k,v in net.state_dict().items()}
  else:wait+=1
  if wait>=45:break
 net.load_state_dict(best_state);net.eval()
 with torch.no_grad():vp=net(vt).numpy();tp=net(torch.tensor((tx-mean)/std)).numpy()
 exported=export(net,mean,std,ym,ys);exported.update(name=name,motion=is_motion,threshold=.5)
 if is_motion:
  predictions=tp*ys+ym;details={'test_local_position_mae_m':float(np.abs(predictions[:,:3]-ty[:,:3]).mean()),'test_direction_residual_mae':float(np.abs(predictions[:,3:]-ty[:,3:]).mean())}
 else:
  vp=1/(1+np.exp(-np.clip(vp,-40,40)));tp=1/(1+np.exp(-np.clip(tp,-40,40)))
  threshold=max([.35,.45,.55,.65,.75],key=lambda t:(metrics(vy,vp,t)['f1'],metrics(vy,vp,t)['precision']))
  exported['threshold']=threshold;details={'validation':metrics(vy,vp,threshold),'test':metrics(ty,tp,threshold),'threshold_selected_on_validation':threshold}
  if name=='helmet':details['limited_warning_events']=5
 details.update(train_samples=len(x),validation_samples=len(vx),test_samples=len(tx),epochs=epoch+1)
 report['models'][name]=details;policy['models'].append(exported);torch.save(net.state_dict(),ROOT/(name+'_policy.pt'))
 print(name,json.dumps(details),flush=True)
 # Cross-runtime inference fixtures are withheld samples, not fitting data.
 for i in range(min(4,len(tx))):
  if 'fixtures' not in policy:policy['fixtures']=[]
  policy['fixtures'].append({'name':name,'features':tx[i].tolist(),'outputs':((tp[i]*ys+ym) if is_motion else tp[i]).tolist()})
(ROOT/'policy.json').write_text(json.dumps(policy),encoding='utf-8')
(ROOT/'training_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('TRAINING COMPLETE',flush=True)
