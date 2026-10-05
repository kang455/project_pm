import json, shutil
from pathlib import Path
root=Path(__file__).resolve().parent
bridge=root.parent.parent/'Tools/YoloBridge'
config=json.loads((bridge/'models.json').read_text(encoding='utf-8-sig'))
person=next(x for x in config['models'] if x['model_id']=='person')
person.update(weights='person_warehouse.pt',diagnostic_only=False,expected_class_names={'0':'person'},accepted_class_names=['person'],inference_size=640)
config['models'].remove(person);config['models'].insert(0,person)
(root/'models.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
