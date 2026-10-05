"""Local Unity JPEG -> multi-model Ultralytics bridge. No cross-model NMS."""
from __future__ import annotations

import argparse
import io
import json
import os
import threading
import cv2
import numpy as np
from collections import OrderedDict
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / ".cache").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache"))
from PIL import Image
import torch
from ultralytics import YOLO


class Bridge:
    def __init__(self, config_path: Path, device: str):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.device = device
        self.lock = threading.Lock()
        self.models = {}
        self.previous_images = OrderedDict()
        self.requests = 0
        self.last_response = None
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        for entry in config["models"]:
            model_id = entry["model_id"]
            if model_id in self.models:
                raise ValueError(f"Duplicate model_id: {model_id}")
            path = (config_path.parent / entry["weights"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            model = YOLO(str(path))  # Loaded once, never inside request processing.
            for cid, expected in entry.get("expected_class_names", {}).items():
                if str(model.names[int(cid)]) != expected:
                    raise ValueError(f"Class mapping changed for {model_id}, class {cid}")
            self.models[model_id] = {
                "model": model, "path": str(path), "task": model.task,
                "model_type": type(model.model).__name__, "names": model.names,
                "architecture": getattr(model.model, "yaml", {}).get("yaml_file"),
                "enabled": bool(entry.get("enabled", True)),
                "confidence": float(entry.get("confidence", 0.25)),
                "diagnostic_only": bool(entry.get("diagnostic_only", False)),
                "person_crops": bool(entry.get("person_crops", False)),
                "verify_flame_motion": bool(entry.get("verify_flame_motion", False)),
                "inference_size": int(entry.get("inference_size", 640)),
                "multi_scale_fire": bool(entry.get("multi_scale_fire", False)),
                "class_min_confidence": {str(k): float(v) for k, v in
                                         entry.get("class_min_confidence", {}).items()},
            }
            print(f"Loaded {model_id}: {model.task}, {model.names}", flush=True)

    def metadata(self):
        return [dict(model_id=k, **{f: v for f, v in m.items() if f != "model"})
                for k, m in self.models.items()]

    def predict(self, jpeg: bytes, frame_id: int, overrides: list, stream_id="default"):
        started = time.perf_counter()
        if not jpeg.startswith(b"\xff\xd8"):
            raise ValueError("Expected JPEG bytes")
        image = Image.open(io.BytesIO(jpeg))
        if image.format != "JPEG" or image.width * image.height > 16_000_000:
            raise ValueError("Invalid JPEG or image too large")
        image = image.convert("RGB")
        rgb = np.asarray(image)
        previous = self.previous_images.get(stream_id)
        aligned = self.align_previous(rgb, previous, frame_id)
        settings = {}
        for item in overrides:
            key = item["model_id"]
            if key not in self.models or key in settings:
                raise ValueError(f"Unknown or repeated model_id: {key}")
            conf = float(item.get("confidence", self.models[key]["confidence"]))
            if not 0 <= conf <= 1:
                raise ValueError("confidence must be in [0, 1]")
            settings[key] = (bool(item.get("enabled", True)), conf)
        detections, raw_detections, classifications, timings = [], [], [], []
        for key, entry in self.models.items():
            enabled, confidence = settings.get(key, (entry["enabled"], entry["confidence"]))
            if not enabled:
                continue
            model_start = time.perf_counter()
            result = entry["model"].predict(image, imgsz=entry["inference_size"], conf=confidence,
                                          device=self.device, verbose=False)[0]
            if entry["multi_scale_fire"] and result.boxes is not None:
                fire_scores = [float(score) for cid, score in zip(result.boxes.cls, result.boxes.conf)
                               if str(result.names[int(cid)]) == "fire"]
                if max(fire_scores, default=0) < .5:
                    detail = entry["model"].predict(image, imgsz=960, conf=confidence,
                                                   device=self.device, verbose=False)[0]
                    combined = torch.cat((result.boxes.data, detail.boxes.data))
                    if len(combined): combined = combined[torch.argsort(combined[:, 4], descending=True)]
                    result.boxes = type(result.boxes)(combined, result.orig_shape)
                    result.speed["inference"] += float(detail.speed.get("inference", 0))
            elapsed = (time.perf_counter() - model_start) * 1000
            names = result.names
            count = 0
            if result.probs is not None:  # Classification is image-level, never a fake box.
                for cid in result.probs.top5:
                    score = float(result.probs.data[cid])
                    if score >= confidence:
                        classifications.append(dict(model_id=key, task=entry["task"],
                                                    class_id=int(cid), class_name=str(names[cid]),
                                                    confidence=score))
                        count += 1
            boxes = result.boxes if result.boxes is not None else result.obb
            if boxes is not None:
                coords = boxes.xyxy.cpu().tolist()
                ids = boxes.cls.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                for i, (xyxy, cid, score) in enumerate(zip(coords, ids, scores)):
                    cid = int(cid)
                    row = dict(model_id=key, task=entry["task"], class_id=cid,
                               class_name=str(names[cid]), confidence=float(score),
                               xyxy=[float(x) for x in xyxy], keypoints=[])
                    if result.keypoints is not None:
                        points = result.keypoints.xy[i].cpu().tolist()
                        confidences = (result.keypoints.conf[i].cpu().tolist()
                                       if result.keypoints.conf is not None else [None] * len(points))
                        row["keypoints"] = [dict(x=float(p[0]), y=float(p[1]), confidence=c)
                                            for p, c in zip(points, confidences)]
                    raw_detections.append(dict(row))
                    # Server-side acceptance, shared by UI and ML-Agents. Preserve raw IDs/names.
                    minimum = max(confidence, entry["class_min_confidence"].get(str(cid), 0))
                    reason = None
                    if entry["diagnostic_only"]:
                        reason = "unverified_class_semantics"
                    elif key == "nohelmet" and not self.head_match(row, detections):
                        reason = "no_matching_person_head"
                    elif entry["verify_flame_motion"] and str(names[cid]) == "fire":
                        reason = self.flame_reason(rgb, aligned, xyxy)
                    if score < minimum: reason = "below_acceptance_threshold"
                    if reason: raw_detections[-1]["rejection_reason"] = reason
                    if score >= minimum and reason is None and not any(
                        d["model_id"] == key and d["class_id"] == cid and self.iou(d["xyxy"],row["xyxy"]) > .4
                        for d in detections):
                        detections.append(row)
                        count += 1
            if entry["person_crops"]:
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
            timings.append(dict(model_id=key, task=entry["task"], processing_ms=elapsed,
                                inference_ms=float(result.speed.get("inference", 0)), count=count))
        # Associate direct nohelmet detections with explicitly named person boxes only.
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
        self.previous_images[stream_id] = (rgb.copy(), frame_id, time.perf_counter())
        self.previous_images.move_to_end(stream_id)
        while len(self.previous_images)>8: self.previous_images.popitem(last=False)
        response = dict(frame_id=frame_id, width=image.width, height=image.height,
                        processing_ms=(time.perf_counter() - started) * 1000,
                        inference_ms=sum(t["inference_ms"] for t in timings),
                        coordinate_origin="top_left", detections=detections,
                        raw_detections=raw_detections,
                        classifications=classifications, model_results=timings)
        self.requests += 1
        self.last_response = response
        return response

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


class Handler(BaseHTTPRequestHandler):
    bridge: Bridge

    def log_message(self, *_):
        pass  # No per-frame logs.

    def send_json(self, code, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        if self.path == "/models":
            self.send_json(200, {"models": self.bridge.metadata()})
        elif self.path == "/video":
            self.send_json(200, self.video.metrics())
        elif self.path == "/health":
            last = self.bridge.last_response
            self.send_json(200, dict(status="ready", requests=self.bridge.requests,
                                     device=self.bridge.device,
                                     last_frame_id=last["frame_id"] if last else None,
                                     last_processing_ms=last["processing_ms"] if last else None,
                                     last_detection_count=len(last["detections"]) if last else None))
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/predict":
            return self.send_json(404, {"error": "Not found"})
        if not self.bridge.lock.acquire(blocking=False):
            return self.send_json(429, {"error": "Inference already in progress"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 6_000_000:
                raise ValueError("Invalid Content-Length")
            self.connection.settimeout(15)
            frame_id = int(self.headers.get("X-Frame-Id", "0"))
            overrides = json.loads(self.headers.get("X-Model-Settings", "{\"models\":[]}"))["models"]
            data = self.rfile.read(size)
            if len(data) != size:
                raise ValueError("Incomplete JPEG")
            self.send_json(200, self.bridge.predict(data, frame_id, overrides, self.headers.get("X-Stream-Id", "default")[:80]))
        except (ValueError, KeyError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            print(f"Inference failed: {type(exc).__name__}: {exc}", flush=True)
            self.send_json(500, {"error": str(exc)})
        finally:
            self.bridge.lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "models.json")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    args = parser.parse_args()
    Handler.bridge = Bridge(args.config.resolve(), args.device)
    from video_stream import VideoServer
    Handler.video = VideoServer(Handler.bridge)
    threading.Thread(target=Handler.video.serve_forever, daemon=True).start()
    print("Latest-frame video ready at tcp://127.0.0.1:8766", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    server.daemon_threads = True
    print("YOLO bridge ready at http://127.0.0.1:8765 (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        Handler.video.shutdown()
        Handler.video.server_close()

