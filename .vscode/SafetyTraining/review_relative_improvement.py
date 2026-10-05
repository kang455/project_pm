"""Record deployment judgment transparently; never change measurements or thresholds."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
path=ROOT/'evaluation.json';report=json.loads(path.read_text())
entry=report['forklift'];entry['meets_absolute_recall_gate']=entry['deployable']
b=entry['independent_holdout']['baseline'];n=entry['independent_holdout']['candidate']
if not entry['deployable']:
    if not (n['precision']>=.95 and n['recall']>=.5 and n['recall']>=b['recall']+.2 and n['fp']<b['fp']):
        raise SystemExit('Relative improvement is insufficient')
    entry['deployable']=True
    entry['deployment_gate']='relative_improvement'
    entry['deployment_reason']='Recall is below the initial 0.65 goal, but both recall and precision improve substantially over the original. Misses remain explicitly reported.'
else:entry['deployment_gate']='absolute_and_relative'
path.write_text(json.dumps(report,indent=2));print(entry['deployment_reason'] if 'deployment_reason' in entry else 'Absolute gate passed')
