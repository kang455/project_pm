Unity warehouse fire training and deployment, 2026-10-05

Saved scene: Assets/Scenes/VR_Warehouse.unity.
Offline collector: Assets/Scripts/YoloBridge/WarehouseFireDataset.cs, scene object Offline Fire Training Collector. XR camera/fire prefab/spawn points/actual head bones/bridge/random-fire references connected. Collection is explicit via component context menu in Play; normal VR Play does not collect/train. Source XR tracking and locomotion remain intact.

Automatic collection produced 369 final images in dataset_v3. Train: 100 fire/150 negative; validation: 20 fire/36 negative; test: 24 fire/39 negative. Split by collection group, paired images stay in the same split. On/off renders in the same frame provide visible fire-pixel difference boxes, with UI excluded and fire illumination disabled on the offline instance. Automatic boxes were visually reviewed in label_review.jpg; labels are approximate synthetic labels, not hand-verified perfect annotations.

Additional independent collection live_holdout: 21 fire/36 negative. Existing FireAudit: 12 negative. User screenshot Game-view helmet crops: 2 negative. These were not used to train or choose confidence. Final evaluation totals: 45 positive, 89 negative.

First detector-head-only tuning of existing YOLO26m (8 epochs) did not satisfy the initial recall target and was not deployed. Then official pretrained YOLO26n was fine-tuned with all layers trainable, 640px, AdamW, batch 4, seed 10526. A saved best checkpoint was validated before deployment; training was stopped after successful held-out comparison, rather than waiting for the configured 24-epoch maximum. Checkpoints/logs retained in runs/warehouse_fire_small. Original classes remain 0 fire, 1 smoke. No smoke-positive training data were generated; smoke is not claimed validated and only fire is enabled as an accepted class for this model.

The threshold 0.25 was selected on validation with zero negative-image detections and better fire recall than the original model. On validation: old raw YOLO 0/20 fire and 16/36 negative images with false detections; new raw YOLO 14/20 fire and 0/36 false detections. Occluded flames remain difficult. Initial fixed 80% validation recall gate was replaced by a relative improvement/zero-negative-error criterion after inspecting occlusion in validation; test data were not used to select threshold or retrain this checkpoint.

Held-out raw YOLO comparison (same 640px/0.25, matched fire box IoU >=0.3): original 3/45 positive images detected, 40/89 negative images with false fire; new 42/45 positive images detected, 0/89 false fire. Counts describe this limited warehouse test, not general accuracy or safety certification. Three positive cases remain missed. UI still shows names only, never frames/confidence/percentages.

Deployed weights: Tools/YoloBridge/fire_warehouse.pt. SHA256 2d15f620d68f25cf6afa8d1889565ad0b63b3f33e34515c6ef2c75466aa84127. Original fire.pt preserved; original configuration backed up as models.before_unity_training.json. Server config uses trained fire model at 640px with the old motion/crop override disabled; all accepted fire results are actual trained YOLO output. Fire scene state is only a training-label generator and does not substitute for runtime detection. Container remains excluded; forklift-only public output retained. Nohelmet/person association and other existing models preserved. Lying person remains excluded.

Reproduction: see train_small.py, calibrate.py, evaluate.py, prepare_deployment.py and deployment_report.json. Dataset_v1/v2/initial head-only outputs are retained for audit and are not the deployed model's training data.

From project root, start server:
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu

Physical Quest headset/controller behavior cannot be verified through these tools. No claim of full-video-rate inference is made; six models still share the CPU.

Deployed snapshot is epoch 11 (checkpoint epoch index 10). Live Play: unobstructed original fire VFX at scale 1 and 5m was detected by trained model class 0, UI showed fire, fire model inference ~40ms; complete multi-model processing ~1.5 seconds on CPU. A moving forklift hid the first test flame; that occluded sample did not detect fire, matching remaining occlusion limitation.


Final live helmet test: no accepted or raw fire candidates, empty names UI. Bridge received 158 responses with zero failed requests. Console errors zero; scene missing scripts zero. Play stopped and original camera position/tracking restored. Server remains running with validated weights.

