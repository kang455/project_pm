import os,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache')
import torch
from ultralytics import YOLO

def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)
def predictions(model,kind,split,base=None):
    rows=[]
    base=base or ROOT/'dataset'
    files=sorted((base/kind/'images'/split).glob('*.jpg'))
    for result,p in zip(model.predict(files,imgsz=640,conf=.05,device='cpu',verbose=False,stream=True),files):
        h,w=result.orig_shape;labels=[]
        for line in (base/kind/'labels'/split/(p.stem+'.txt')).read_text().splitlines():
            cid,x,y,bw,bh=map(float,line.split());labels.append((int(cid),[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
        boxes=[(int(cid),float(score),box) for cid,score,box in zip(result.boxes.cls.tolist(),result.boxes.conf.tolist(),result.boxes.xyxy.tolist())]
        rows.append({'image':p.name,'split':split,'labels':labels,'predictions':boxes})
    return rows
def metric(rows,cid,threshold):
    tp=fp=fn=positive=negative=negative_fp=0
    for row in rows:
        truth=[b for c,b in row['labels'] if c==cid];used=set();found=0
        detections=sorted([(s,b) for c,s,b in row['predictions'] if c==cid and s>=threshold],reverse=True)
        # Mirror server acceptance: suppress duplicate boxes only within this model/class.
        unique=[]
        for item in detections:
            if not any(iou(item[1],existing[1])>.4 for existing in unique):unique.append(item)
        detections=unique
        for score,box in detections:
            matches=[(iou(box,gt),i) for i,gt in enumerate(truth) if i not in used]
            best=max(matches,default=(0,-1))
            if best[0]>=.3:tp+=1;used.add(best[1]);found+=1
            else:fp+=1
        fn+=len(truth)-len(used)
        if truth:positive+=1
        else:negative+=1;negative_fp+=bool(detections)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                positive_images=positive,negative_images=negative,negative_false_positive_images=negative_fp)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-only',action='store_true');parser.add_argument('--model',choices=['helmet','nohelmet','forklift']);args=parser.parse_args()
    torch.set_num_threads(4);report={}
    config=json.loads((ROOT.parent.parent/'Tools/YoloBridge/models.json').read_text())
    for kind,cid,default in [('helmet',1,.25),('nohelmet',0,.35),('forklift',2,.25)]:
        if args.model and kind!=args.model:continue
        old=YOLO(str(ROOT.parent.parent/'Tools/YoloBridge'/(kind+'.pt')))
        entry={'original_names':old.names,'original_architecture':old.model.yaml.get('yaml_file'),'class_id':cid}
        dataset=ROOT/'dataset_v2' if kind=='forklift' and (ROOT/'dataset_v2/forklift/data.yaml').exists() else ROOT/'dataset'
        baseline={split:predictions(old,kind,split,dataset) for split in ('val','test')}
        entry['baseline']={split:metric(rows,cid,default) for split,rows in baseline.items()}
        if args.baseline_only:report[kind]=entry;continue
        new=YOLO(str(ROOT/(kind+'_warehouse_candidate.pt')))
        if new.names!=old.names:raise RuntimeError('Class map changed '+kind)
        candidate={split:predictions(new,kind,split,dataset) for split in ('val','test')}
        options=[(metric(candidate['val'],cid,t),t) for t in [.25,.35,.45,.55,.65]]
        # Calibrate on validation only, never choose thresholds using the test results.
        valid=[(m,t) for m,t in options if m['recall']>=.6 and m['precision']>=.85]
        if not valid:valid=options
        chosen,threshold=max(valid,key=lambda z:(2*z[0]['precision']*z[0]['recall']/max(.001,z[0]['precision']+z[0]['recall']),z[0]['precision'],-z[1]))
        entry.update(candidate_names=new.names,threshold=threshold,candidate={split:metric(rows,cid,threshold) for split,rows in candidate.items()})
        b=entry['baseline']['test'];n=entry['candidate']['test']
        holdout=ROOT/'final_holdout' if (ROOT/'final_holdout'/kind/'data.yaml').exists() else ROOT/'holdout'
        if (holdout/kind/'data.yaml').exists():
            old_holdout=[];new_holdout=[]
            for split in ('train','val','test'):
                old_holdout.extend(predictions(old,kind,split,holdout))
                new_holdout.extend(predictions(new,kind,split,holdout))
            entry['independent_holdout']={'baseline':metric(old_holdout,cid,default),'candidate':metric(new_holdout,cid,threshold)}
            (ROOT/(kind+'_holdout_predictions.json')).write_text(json.dumps(new_holdout))
            b=entry['independent_holdout']['baseline'];n=entry['independent_holdout']['candidate']
            entry['deployment_test']=holdout.name
        entry['deployable']=n['precision']>=.85 and n['recall']>=.65 and n['recall']>=b['recall'] and n['fp']<=b['fp']
        if kind=='forklift':entry['person']=metric(candidate['test'],3,threshold)
        report[kind]=entry
        (ROOT/(kind+'_predictions.json')).write_text(json.dumps(candidate))
    filename='baseline.json' if args.baseline_only else (args.model+'_evaluation.json' if args.model else 'evaluation.json')
    (ROOT/filename).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
