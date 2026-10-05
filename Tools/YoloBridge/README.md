# Unity multi-model YOLO bridge

Scene: `Assets/Scenes/VR_Warehouse.unity`.

From PowerShell:

```powershell
Set-Location 'D:\새 폴더\unity project\project_pm'
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu
```

The existing weights were found directly in Tools/YoloBridge, rather than its models subfolder. models.json points at those files without moving them. Set weights paths there to change models. Models load once at startup. Dependencies are pinned in requirements.txt; the supplied .venv uses the existing PC Python 3.10 / CPU PyTorch installation.

GET http://127.0.0.1:8765/models gives actual loaded model types, tasks and original class names. All five current weights are detection models. person has classes `0`, `1`, `2`; lying_person has `0`, `1`, `2`, `Fall-Detected`. These names are preserved without guessing their meanings. helmet contains `object` and `safety_helmet_detection_0321 - v3 2024-03-27 8-32pm`.

POST /predict receives raw JPEG, X-Frame-Id and optional X-Model-Settings JSON: {"models":[{"model_id":"fire","enabled":true,"confidence":0.25}]}. Response coordinates use original image pixels with top-left origin. Detections retain model_id, class_id and class_name. Classification results have no boxes; pose keypoints are returned without inventing a lying-state judgement. No cross-model suppression is performed. A busy server rejects another inference with HTTP 429.

Inspector on YOLO Multi-Model Bridge: Models controls enabled/confidence/colors; Show Detections toggles the world-space names panel. It contains names only and hides when there are no approved detections; confidence, percentages, image previews, frame counts and timing are never shown in user UI. The separate mono camera follows the XR camera; UI layer is excluded. Original JPEGs remain internal. The UI shows model/class names from accepted responses only. Async GPU readback orientation was verified upright on this D3D11/URP project with Flip Async Readback Y disabled. Maximum Requests Per Second is 5, but measured processing time increases the interval. One request is in flight; failures retry while VR continues.

ML-Agents can read bridge.LatestResult or subscribe to bridge.ResultReceived. Results are observations, not ground-truth labels. Class identity is the pair (model_id, class_id).

Validation artifacts in .vscode: yolo_first_frame.jpg / yolo_first_response.json, yolo_runtime_frame.jpg, model_inventory.json. The latest fire investigation and names-only UI evidence are in .vscode/FireAudit/README.md. Initial Unity frame returned forklift class 2, confidence 0.71765; other models returned empty detections. Repeated Play requests, UI, original-image orientation, per-model disabling and connection failure/recovery were checked. CPU warm processing is about 0.9–1.1 seconds for all models. Physical Quest display and controller/head tracking must be checked wearing the headset. Play is stopped after validation.

Fire mitigation (2026-10-05): server-side class_min_confidence 0.90 for fire model class 0. Original model candidates are returned in raw_detections; accepted detections also feed ML-Agents. Mapping is guarded at startup. Identical no-fire 12-view comparison: 5 false boxes before, 0 accepted after; the 5 raw false boxes remain identical. This is threshold mitigation, not retraining. Actual fire recall and detection performance remain untested. Raising the threshold may miss real fires. Weights are unchanged.
