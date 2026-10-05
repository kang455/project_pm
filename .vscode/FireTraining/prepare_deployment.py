import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cal=json.loads((ROOT/'calibration.json').read_text())['chosen']
evaluation=json.loads((ROOT/'evaluation.json').read_text())
def at(rows,t):return next(r for r in rows if abs(r['threshold']-t)<1e-6)
original=at(evaluation['original']['metrics'],.25)
trained=at(evaluation['trained']['metrics'],cal['threshold'])
if trained['false_positive_images']>original['false_positive_images'] or trained['detected_positive_images']<original['detected_positive_images']:
 raise RuntimeError('Held-out test failed to improve; do not deploy')
if trained['detected_positive_images']<.6*trained['positive_images']:
 raise RuntimeError('Held-out detection recall below acceptance')
config=json.loads((ROOT/'models.original.json').read_text(encoding='utf-8-sig'))
fire=next(m for m in config['models'] if m['model_id']=='fire')
fire.update(weights='fire_warehouse.pt',confidence=cal['threshold'],inference_size=640,
 class_min_confidence={'0':cal['threshold']},verify_flame_motion=False,multi_scale_fire=False,
 accepted_class_names=['fire'])
(ROOT/'models.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
report=dict(original=original,trained=trained,threshold=cal['threshold'],
 candidate_sha256=hashlib.sha256((ROOT/'fire_warehouse_candidate.pt').read_bytes()).hexdigest(),
 classes={'0':'fire','1':'smoke'},scope='Unity warehouse VFX only; smoke not trained/validated',
 collection='visible on/off image difference, UI excluded; grouped split; labels are synthetic estimates')
(ROOT/'deployment_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

