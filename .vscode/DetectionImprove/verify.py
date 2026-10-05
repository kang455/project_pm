import json
from pathlib import Path
from server import Bridge
root=Path(__file__).parent
c=json.loads((root/'models.json').read_text())
for m in c['models']:m['weights']=str((Path('../Tools/YoloBridge')/m['weights']).resolve())
(root/'test_models.json').write_text(json.dumps(c),encoding='utf-8')
bridge=Bridge(root/'test_models.json','cpu')
rows=[]
images=list(Path('FireAudit').glob('view_*.jpg'))+list(Path('SceneSafetyValidation').glob('Comparison_*.jpg'))+[Path('ModelAdditionValidation/YellowHelmetPair.jpg'),Path('ModelAdditionValidation/BareheadFrontal.jpg'),root/'user_game_0.jpg',root/'user_game_1.jpg']
for f in images:
 for n in range(2):
  r=bridge.predict(f.read_bytes(),n+1,[],f.name)
 (root/(f.stem+'_improved.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
 rows.append({'image':str(f),'accepted':[(d['model_id'],d['class_name'],d.get('source','full')) for d in r['detections']],'reject':[(d['model_id'],d['class_name'],d.get('rejection_reason')) for d in r['raw_detections'] if d.get('rejection_reason')]})
 print(rows[-1],flush=True)
# Actual VFX changes between the two rendered flame frames, same camera/background.
r1=bridge.predict(Path('SceneSafetyValidation/Comparison_1.jpg').read_bytes(),1,[],'fire-animation')
r2=bridge.predict(Path('SceneSafetyValidation/Comparison_3.jpg').read_bytes(),2,[],'fire-animation')
(root/'changing_flame.json').write_text(json.dumps(r2,indent=2),encoding='utf-8')
print('changing_flame',[(d['model_id'],d['class_name']) for d in r2['detections']],flush=True)
(root/'verification.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
