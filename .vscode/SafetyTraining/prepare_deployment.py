import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'evaluation.json').read_text())
if set(report)!=set(('helmet','nohelmet','forklift')) or not all(v['deployable'] for v in report.values()):
    raise SystemExit('Validation gate failed; original models remain active')
config=json.loads((ROOT.parent.parent/'Tools/YoloBridge/models.json').read_text())
hashes={}
for entry in config['models']:
    kind=entry['model_id']
    if kind not in report:continue
    source=ROOT/(kind+'_warehouse_candidate.pt')
    hashes[kind]=hashlib.sha256(source.read_bytes()).hexdigest()
    entry['weights']=kind+'_warehouse.pt'
    entry['confidence']=report[kind]['threshold']
    entry['inference_size']=640
    entry['expected_class_names']=report[kind]['candidate_names']
    if kind=='helmet':
        entry['accepted_class_names']=[report[kind]['candidate_names']['1']]
        entry['require_person_head']=True
    elif kind=='forklift':entry['accepted_class_names']=['forklift','person']
    if kind=='forklift':
        entry['confidence']=.25
        entry['class_min_confidence']={'2':report[kind]['threshold'],'3':.25}
    if kind=='nohelmet':
        # The new model is validated directly on full Unity frames; retain person association.
        entry['person_crops']=False
(ROOT/'models.json').write_text(json.dumps(config,indent=2))
(ROOT/'deployment_report.json').write_text(json.dumps({'sha256':hashes,'evaluation':report},indent=2))
print('Validated deployment prepared',hashes)
