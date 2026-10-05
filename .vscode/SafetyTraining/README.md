# Unity warehouse helmet, nohelmet and forklift training

The offline `WarehouseSafetyDataset` component is connected in `Assets/Scenes/VR_Warehouse.unity`. Collection starts only through its Play-mode context menu or `Begin()`, never during normal VR play. It disables inference and random fire while collecting and restores both afterwards. Temporary helmets are removed; original worker appearance is restored. Runtime inference does not read scene clothing, renderer names or offline labels.

Dataset: 316 distinct rendered camera views shared by three datasets (948 JPEG files), each with 234 training, 42 validation and 40 test images. Paired helmet/bare views stay in the same split. The four worker subjects reserved for validation/test are excluded from the worker-centered training views. Other workers can still appear in the background, so these are scene-specific tests, not independent real-world tests. Forklift views use separate randomized camera positions, but the same vehicle asset and warehouse. Visibility labels use the difference between normal rendering and rendering with only the target's renderers disabled. This handles occlusion approximately; tiny and heavily occluded objects can be omitted. Reviewed overlays: `label_review.jpg`.

Training uses the existing official COCO-pretrained YOLO26n checkpoint, separately fine-tuned for each dataset. It is not an imitation of the previous predictions. All original class IDs and names are retained in the saved models:

- Helmet: 0 `object` (not trained/accepted), 1 `safety_helmet_detection_0321 - v3 2024-03-27 8-32pm` (rendered scooter helmets). UI alias for class 1 is exactly `helmet detected`.
- Nohelmet: 0 `nohelmet`, labeled from visibly bare heads, not helmet detection absence.
- Forklift: 2 `forklift`, 3 `person`; other original class mappings remain but are not trained/accepted. Person boxes support nohelmet association internally; only forklift is shown for the forklift model.

Forklift refinement: the initial collector labeled only the counterbalance vehicle and omitted the reach truck, which is another forklift in this warehouse. Its renderer groups are now connected separately, excluding driver renderers and ground shadows. The v2 dataset contains 432 camera images at 90-degree FOV (312 train, 62 validation, 58 test), including views of both vehicles and paired renders with vehicles/drivers hidden. It teaches the distinction between forklifts and shelves/pallets. Only the forklift candidate is fine-tuned on v2. A new 188-image final holdout (seed 42527) is reserved after investigating the initial holdout; it is never used for training or calibration. All three models are assessed on that final set before deployment.

Final forklift judgment: the initial absolute recall target of 0.65 was not met. The candidate finds 38/72 objects with zero unmatched detections; the original finds 18/72 with 22 unmatched detections. Deployment is accepted as a clear relative improvement (higher precision and recall), not as complete recognition. The report records the failed absolute criterion and the alternative judgment. Helmet finds 71/98 (4 unmatched), nohelmet 94/114 (9 unmatched). These are model-level measurements; runtime person association can reject additional nohelmet results. Small and occluded targets remain a limitation.

Runtime regression: the raw helmet model also classified a flame as a helmet. Worn-helmet acceptance now requires a YOLO-detected person's head region, just like nohelmet. Rejected raw results keep `no_matching_person_head`; accepted results are also what ML-Agents receive. This uses image inference only, not Unity fire/clothing state. Loose helmets without a detected wearer are not confirmed by this acceptance rule. The underlying raw model's flame confusion remains a model limitation.

Candidate confidence is selected using validation images only. Test detection matches require IoU >= 0.3. Reports compare the original and newly trained models using the same images and the server's per-class duplicate suppression (IoU > 0.4). Initial collection conditioned helmet/person annotations on visible face pixels; that omitted some visible helmets. This was corrected in the collector. A separate 158-view, 90-degree FOV holdout was collected with seed 20527 and complete renderer visibility labels. These images are never used for training or threshold selection; deployment tests use this corrected independent set. Original test metrics are retained for transparency. Deployment requires adequate test precision/recall and improvement over baseline. Original weight files and the previously trained fire model are preserved. See `evaluation.json` and `deployment_report.json` for actual measurements.

Run from the project's `.vscode` folder:

```powershell
& '../Tools/YoloBridge/.venv/Scripts/python.exe' SafetyTraining/pipeline.py
& '../Tools/YoloBridge/.venv/Scripts/python.exe' SafetyTraining/forklift_refine.py
& '../Tools/YoloBridge/.venv/Scripts/python.exe' SafetyTraining/evaluate.py
& '../Tools/YoloBridge/.venv/Scripts/python.exe' SafetyTraining/review_relative_improvement.py
& '../Tools/YoloBridge/.venv/Scripts/python.exe' SafetyTraining/prepare_deployment.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File './SafetyTraining/deploy.ps1'
```

Normal server startup, from the project root:

```powershell
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu
```

Confidence values and evaluation percentages are diagnostic data only, never user-facing VR UI. Headset display and physical controller response require the user's Quest verification. These weights have been evaluated only in this synthetic warehouse. Small or occluded objects can still be missed.

Completed runtime checks: Python server metadata confirms all three new `*_warehouse.pt` files as YOLO26n detection models with unchanged class maps. Actual Unity-rendered JPEGs were sent to the restarted server: helmet accepted alone; bare head accepted as nohelmet with a person box; forklift plus its bare-headed driver accepted; fire accepted while the raw helmet/flame confusion was rejected with head evidence. In Play, the Game View showed exactly `helmet detected`, without a date, confidence, percentage or image preview. The last sampled session had 40 successful responses and 0 failures; the preceding session reached 334 successful responses and 0 failures. No Console errors or missing scripts were found. Play ended, the scene was saved, tracking is enabled, random fire spawning is enabled, the main camera returned to (31, 1.39, 52), and only one Audio Listener is active. The original fire weights' SHA256 is unchanged. Quest display and physical controller response were not directly verified.
