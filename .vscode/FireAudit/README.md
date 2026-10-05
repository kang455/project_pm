# Fire false-positive investigation (2026-10-05)

The active VR_Warehouse scene contains no intentional fire object/effect, as confirmed by the user. Actual fire.pt metadata: YOLO26m, DetectionModel, task detect, class 0 fire, class 1 smoke. The server maps model result IDs through result.names without renaming classes. SHA256: B18C28292517A356BD49DDB6FEA4ADA311F8CD857DCE92CF0F42741EC7905487. Weights were not modified.

Captured 12 upright 640x360 JPEG views at the current XR position, varying yaw. The original raw model returned 5 false fire boxes across 3 views. Examples include a yellow warehouse column in view_09.jpg and other non-fire structure/props in view_00.jpg and view_11.jpg. Mapping was correct; this is model-level false detection, not a UI label swap. The exact training cause cannot be determined from the checkpoint alone.

Mitigation: models.json sets a server-side class_min_confidence floor of 0.90 for fire model class 0 only. Original inference still runs at the request's candidate threshold (default 0.25). raw_detections preserves those candidates, including their original class IDs, class names, confidences and boxes. Only candidates meeting both requested and class-specific acceptance thresholds enter detections, timing counts, Unity LatestResult.detections and ResultReceived's approved observations. This is a conservative decision threshold change, not retraining or proof of improved underlying model accuracy. The startup expected_class_names guard prevents silent checkpoint label changes.

The identical 12 input images yielded 0 approved fire boxes after the change, while all 5 original raw false positives and their values remained identical. baseline.json, after.json, summary.json and per-image before/after JSON files record the comparison. The 12 views served as calibration examples; they are not a held-out accuracy benchmark. Runtime/ contains 20 additional real Play image-response pairs: 0 approved fire boxes, 0 raw fire boxes, 5 other approved detections. This run verifies transmission and live operation, not an additional reduction against a positive baseline.

User UI now contains only colored model/class names. RawImage preview, coordinate boxes, confidence, percentages, timing and frame/count diagnostics were removed from the canvas. With no approved names the whole panel is disabled; network failures clear it. Confidence and original images remain internal for inference/diagnostics. Other model class names, including numeric names, remain unchanged.

Actual fire detection has not been tested because this scene has no fire. Raising the threshold may miss actual fires. This change does not eliminate all possible false positives; reliable performance will require representative virtual fire positives and hard-negative warehouse images for evaluation and possibly retraining. Physical Quest display/interaction needs headset validation.

PowerShell server command from project root:

```powershell
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu
```
