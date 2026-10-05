import json
from pathlib import Path
p = Path(__file__).parent
c = json.loads((p/'models.json').read_text())
c['models'] += [dict(model_id='fire2', weights='fire (2).pt', enabled=True, confidence=.25), dict(model_id='nohelmet', weights='nohelmet.pt', enabled=True, confidence=.5, expected_class_names={'0':'nohelmet'})]
(p/'models.json').write_text(json.dumps(c, indent=2), encoding='utf-8')
s = (p/'server.py').read_text(encoding='utf-8')
pos = s.index('        response = dict(')
association = '''        # Associate direct nohelmet detections with explicitly named person boxes only.
        # Numeric classes and Unity clothing states are not person evidence.
        people = [d for d in detections if d["class_name"].strip().lower() == "person"]
        for d in detections:
            if d["model_id"] != "nohelmet" or d["class_name"] != "nohelmet":
                continue
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            matches = []
            for person in people:
                px1, py1, px2, py2 = person["xyxy"]
                overlap = max(0, min(x2,px2)-max(x1,px1))*max(0, min(y2,py2)-max(y1,py1))
                if px1 <= cx <= px2 and py1 <= cy <= py2 and overlap/max(1,(x2-x1)*(y2-y1)) >= .6:
                    matches.append(person)
            if matches:
                person = min(matches, key=lambda p:(p["xyxy"][2]-p["xyxy"][0])*(p["xyxy"][3]-p["xyxy"][1]))
                d["person_model_id"] = person["model_id"]
                d["person_class_id"] = person["class_id"]
                d["person_xyxy"] = person["xyxy"]
'''
s = s[:pos] + association + s[pos:]
s = s.replace('except (BrokenPipeError, ConnectionResetError):', 'except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):')
(p/'server.py').write_text(s, encoding='utf-8')
