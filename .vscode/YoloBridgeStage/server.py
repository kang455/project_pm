"""Local Unity JPEG -> multi-model Ultralytics bridge. No cross-model NMS."""
from __future__ import annotations

import argparse
import io
import json
import os
import threading
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
                "class_min_confidence": {str(k): float(v) for k, v in
                                         entry.get("class_min_confidence", {}).items()},
            }
            print(f"Loaded {model_id}: {model.task}, {model.names}", flush=True)

    def metadata(self):
        return [dict(model_id=k, **{f: v for f, v in m.items() if f != "model"})
                for k, m in self.models.items()]

    def predict(self, jpeg: bytes, frame_id: int, overrides: list):
        started = time.perf_counter()
        if not jpeg.startswith(b"\xff\xd8"):
            raise ValueError("Expected JPEG bytes")
        image = Image.open(io.BytesIO(jpeg))
        if image.format != "JPEG" or image.width * image.height > 16_000_000:
            raise ValueError("Invalid JPEG or image too large")
        image = image.convert("RGB")
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
            result = entry["model"].predict(image, imgsz=640, conf=confidence,
                                          device=self.device, verbose=False)[0]
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
                    raw_detections.append(row)
                    # Server-side acceptance, shared by UI and ML-Agents. Preserve raw IDs/names.
                    minimum = max(confidence, entry["class_min_confidence"].get(str(cid), 0))
                    if score >= minimum:
                        detections.append(row)
                        count += 1
            timings.append(dict(model_id=key, task=entry["task"], processing_ms=elapsed,
                                inference_ms=float(result.speed.get("inference", 0)), count=count))
        response = dict(frame_id=frame_id, width=image.width, height=image.height,
                        processing_ms=(time.perf_counter() - started) * 1000,
                        inference_ms=sum(t["inference_ms"] for t in timings),
                        coordinate_origin="top_left", detections=detections,
                        raw_detections=raw_detections,
                        classifications=classifications, model_results=timings)
        self.requests += 1
        self.last_response = response
        return response


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
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/models":
            self.send_json(200, {"models": self.bridge.metadata()})
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
            self.send_json(200, self.bridge.predict(data, frame_id, overrides))
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
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    server.daemon_threads = True
    print("YOLO bridge ready at http://127.0.0.1:8765 (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
