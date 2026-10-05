import json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
models=json.load(urllib.request.urlopen('http://127.0.0.1:8765/models'))
(ROOT/'loaded_models.json').write_text(json.dumps(models,indent=2,ensure_ascii=False),encoding='utf-8')
for index,case in enumerate(('helmet','nohelmet','forklift','fire')):
    image=ROOT/('live_'+case+'.jpg')
    request=urllib.request.Request('http://127.0.0.1:8765/predict',data=image.read_bytes(),
        headers={'Content-Type':'image/jpeg','X-Frame-Id':str(100000+index),'X-Stream-Id':'safety-validation'})
    result=json.load(urllib.request.urlopen(request,timeout=30))
    (ROOT/('verified_'+case+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    accepted=[d['model_id'] for d in result['detections']]
    assert case in accepted,(case,accepted)
    if case=='helmet':assert 'nohelmet' not in accepted and 'fire' not in accepted
    if case=='nohelmet':
        assert 'helmet' not in accepted and 'fire' not in accepted
        assert next(d for d in result['detections'] if d['model_id']=='nohelmet').get('person_xyxy')
    if case=='fire':assert 'helmet' not in accepted
    print(case,'accepted:',accepted,'processing_ms:',round(result['processing_ms']))
print('Unity-rendered image regression checks passed')
