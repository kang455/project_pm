import json
from pathlib import Path
p=Path(__file__).parent
c=json.loads((p/'models.json').read_text())
c['models']=[m for m in c['models'] if m['model_id']!='lying_person']
for m in c['models']:
 if m['model_id']=='fire':m['class_min_confidence']={'0':.3};m['verify_flame_motion']=True
 if m['model_id']=='nohelmet':m['confidence']=.35;m['person_crops']=True
 if m['model_id'] in ('fire2','person'):m['diagnostic_only']=True
(p/'models.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
s=(p/'server.py').read_text(encoding='utf-8')
s=s.replace('import threading','import threading\nimport cv2\nimport numpy as np\nfrom collections import OrderedDict')
s=s.replace('        self.models = {}','        self.models = {}\n        self.previous_images = OrderedDict()')
s=s.replace('                "class_min_confidence":', '                "diagnostic_only": bool(entry.get("diagnostic_only", False)),\n                "person_crops": bool(entry.get("person_crops", False)),\n                "verify_flame_motion": bool(entry.get("verify_flame_motion", False)),\n                "class_min_confidence":')
s=s.replace('def predict(self, jpeg: bytes, frame_id: int, overrides: list):','def predict(self, jpeg: bytes, frame_id: int, overrides: list, stream_id="default"):')
s=s.replace('        settings = {}','        rgb = np.asarray(image)\n        previous = self.previous_images.get(stream_id)\n        aligned = self.align_previous(rgb, previous, frame_id)\n        settings = {}')
s=s.replace('            result = entry["model"].predict(image, imgsz=640, conf=confidence,','            result = entry["model"].predict(image, imgsz=640, conf=confidence,')
s=s.replace('                    raw_detections.append(row)','                    raw_detections.append(dict(row))')
s=s.replace('                    if score >= minimum:', '''                    reason = None
                    if entry["diagnostic_only"]:
                        reason = "unverified_class_semantics"
                    elif key == "nohelmet" and not self.head_match(row, detections):
                        reason = "no_matching_person_head"
                    elif entry["verify_flame_motion"] and str(names[cid]) == "fire":
                        reason = self.flame_reason(rgb, aligned, xyxy)
                    if reason: row["rejection_reason"] = reason
                    if score >= minimum and reason is None:''')
insert=s.index('            timings.append(')
s=s[:insert]+'''            if entry["person_crops"]:
                people = [d for d in detections if d["class_name"] == "person"]
                crop_inference_ms = 0
                for person in sorted(people, key=lambda d:d["confidence"], reverse=True)[:4]:
                    px1, py1, px2, py2 = person["xyxy"]
                    pw, ph = px2-px1, py2-py1
                    rect = (max(0,int(px1-.25*pw)), max(0,int(py1-.25*ph)),
                            min(image.width,int(px2+.25*pw)), min(image.height,int(py1+.35*ph)))
                    if rect[2]-rect[0] < 24 or rect[3]-rect[1] < 24: continue
                    crop = entry["model"].predict(image.crop(rect), imgsz=640, conf=confidence,
                                                device=self.device, verbose=False)[0]
                    crop_inference_ms += float(crop.speed.get("inference",0))
                    for box, cid, score in zip(crop.boxes.xyxy.tolist(),crop.boxes.cls.tolist(),crop.boxes.conf.tolist()):
                        row = dict(model_id=key, task=entry["task"], class_id=int(cid),
                                   class_name=str(crop.names[int(cid)]),confidence=float(score),
                                   xyxy=[box[0]+rect[0],box[1]+rect[1],box[2]+rect[0],box[3]+rect[1]],
                                   keypoints=[],source="person_crop")
                        raw_detections.append(dict(row))
                        # Crop-edge/trunk hallucinations are not head evidence.
                        if box[3] >= rect[3]-rect[1]-2 or not self.head_match(row,[person]): continue
                        if not any(d["model_id"]==key and self.iou(d["xyxy"],row["xyxy"])>.25 for d in detections):
                            detections.append(row); count += 1
                elapsed = (time.perf_counter()-model_start)*1000
                result.speed["inference"] = float(result.speed.get("inference",0))+crop_inference_ms
''' +s[insert:]
s=s.replace('        response = dict(','''        self.previous_images[stream_id] = (rgb.copy(), frame_id, time.perf_counter())
        self.previous_images.move_to_end(stream_id)
        while len(self.previous_images)>8: self.previous_images.popitem(last=False)
        response = dict(''')
s=s.replace('                d["person_xyxy"] = person["xyxy"]','                d["person_xyxy"] = person["xyxy"]')
pos=s.index('\n\nclass Handler')
helpers='''
    @staticmethod
    def iou(a,b):
        overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return overlap/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)

    @staticmethod
    def head_match(row, people):
        x1,y1,x2,y2=row["xyxy"]; cx,cy=(x1+x2)/2,(y1+y2)/2
        for p in people:
            if p["class_name"]!="person": continue
            a,b,c,d=p["xyxy"]; w,h=c-a,d-b
            if a-.2*w<=cx<=c+.2*w and b-.25*h<=cy<=b+.35*h and y2-y1<=.5*h and x2-x1<=1.5*w:
                return True
        return False

    @staticmethod
    def align_previous(rgb, previous, frame_id):
        if previous is None:return None
        old, old_id, timestamp=previous
        if old.shape!=rgb.shape or frame_id<=old_id or time.perf_counter()-timestamp>8:return None
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);before=cv2.cvtColor(old,cv2.COLOR_RGB2GRAY)
        orb=cv2.ORB_create(nfeatures=800)
        k1,d1=orb.detectAndCompute(before,None);k2,d2=orb.detectAndCompute(gray,None)
        if d1 is None or d2 is None:return None
        matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
        good=[m[0] for m in matches if len(m)==2 and m[0].distance<.7*m[1].distance]
        if len(good)<20:return None
        source=np.float32([k1[m.queryIdx].pt for m in good]);target=np.float32([k2[m.trainIdx].pt for m in good])
        matrix,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=2)
        if matrix is None or float(inliers.mean())<.65:return None
        valid=cv2.warpAffine(np.ones(old.shape[:2],np.uint8),matrix,(rgb.shape[1],rgb.shape[0]))
        return cv2.warpAffine(old,matrix,(rgb.shape[1],rgb.shape[0])),valid

    @staticmethod
    def flame_reason(rgb, aligned, box):
        x1,y1,x2,y2=map(int,box);x1=max(0,x1);y1=max(0,y1);x2=min(rgb.shape[1],x2);y2=min(rgb.shape[0],y2)
        if x2<=x1 or y2<=y1 or not .25<=(x2-x1)/(y2-y1)<=2.5:return "not_flame_shape"
        crop=rgb[y1:y2,x1:x2];mask=cv2.inRange(cv2.cvtColor(crop,cv2.COLOR_RGB2HSV),(0,75,100),(40,255,255))>0
        if mask.mean()<.04:return "no_warm_flame_pixels"
        if aligned is None:return "awaiting_camera_aligned_motion"
        old,valid=aligned
        if valid[y1:y2,x1:x2].mean()<.95:return "motion_region_outside_previous_view"
        before=old[y1:y2,x1:x2]
        warm_before=cv2.inRange(cv2.cvtColor(before,cv2.COLOR_RGB2HSV),(0,75,100),(40,255,255))>0
        union=mask|warm_before
        change=np.abs(crop.astype(np.int16)-before.astype(np.int16)).mean(axis=2)>25
        if float(change[union].mean())<.12:return "static_warm_object"
        return None
'''
s=s[:pos]+helpers+s[pos:]
s=s.replace('self.bridge.predict(data, frame_id, overrides)','self.bridge.predict(data, frame_id, overrides, self.headers.get("X-Stream-Id", "default")[:80])')
(p/'server.py').write_text(s,encoding='utf-8')
