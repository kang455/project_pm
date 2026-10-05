import json
from pathlib import Path
from server import Bridge
root=Path(__file__).parent
c=json.loads((root/'models.json').read_text())
for m in c['models']:m['weights']=str((Path('../Tools/YoloBridge')/m['weights']).resolve())
(root/'test_models.json').write_text(json.dumps(c),encoding='utf-8')
b=Bridge(root/'test_models.json','cpu')
fire_only=[dict(model_id=m['model_id'],enabled=m['model_id']=='fire') for m in c['models']]
negatives=[]
for f in Path('FireAudit').glob('view_*.jpg'):
 for n in range(2):r=b.predict(f.read_bytes(),n+1,fire_only,f.name)
 negatives.append(dict(image=f.name,fire=sum(d['model_id']=='fire' for d in r['detections'])))
flames=[]
for i,f in enumerate(sorted(Path('DetectionImprove/ForcedFlame').glob('*.jpg'))):
 r=b.predict(f.read_bytes(),i+1,fire_only,'flame-live')
 (root/(f.stem+'_final_flame.json')).write_text(json.dumps(r,indent=2))
 flames.append(dict(image=f.name,fire=sum(d['model_id']=='fire' for d in r['detections'])))
people=[]
for f in [Path('ModelAdditionValidation/BareheadFrontal.jpg'),Path('ModelAdditionValidation/YellowHelmetPair.jpg'),root/'user_game_0.jpg',root/'user_game_1.jpg']:
 r=b.predict(f.read_bytes(),1,[],f.name+'-final')
 (root/(f.stem+'_final.json')).write_text(json.dumps(r,indent=2))
 people.append(dict(image=f.name,nohelmet=sum(d['model_id']=='nohelmet' for d in r['detections'])))
summary=dict(negative_views=negatives,real_flame_frames=flames,people=people)
(root/'final_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
