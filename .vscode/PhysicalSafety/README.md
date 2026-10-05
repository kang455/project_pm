# Warehouse safety interactions

## Current vision-based collision warning

The added `Training Forklift`, its two path markers, and the scene's `WarehouseTrainingScenario` component have been removed. Original Forklift and Reachlift remain, and B still controls their stop/resume. The added pedestrian remains available.

`WarehouseVisionCollisionRisk` consumes accepted YOLO `person` and `forklift` boxes from the same image. Normalized box feet, screen-space proximity, and short-term relative approach supply warning candidates; two consecutive inference results confirm a candidate. A cabin/feet geometry heuristic rejects likely drivers. Missing classes or stale results clear the warning. No actor transforms, scene person lists, colliders, or world distances are inputs to this warning. The geometric pedestrian lists on the remaining actuator components are empty and automatic braking stays disabled.

This is a 2D possible-collision warning, not verified metric distance, a reconstructed 3D path, or a safety guarantee. Projection overlap, camera motion and detection errors can cause false warnings or missed hazards. Recorder schema 2 marks `collision_risk_source: yolo_image_boxes`; observation slot 5 is now a normalized image-space gap (`-1` when unavailable), not world distance. Keep old geometric-source recordings separate from schema-2 demonstrations.

Validation: synthetic accepted boxes triggered for nearby pedestrians, not distant pedestrians, cabin drivers or person-only images; live Play inference remained connected with zero request/Console errors. Physical collision prediction accuracy across real headset movement still needs evaluation.

## Manual VR demonstrations (current scene defaults)

- Right A: issue the helmet warning for four seconds.
- Right B: toggle stop/resume for all three forklifts.
- Left X: toggle right-hand extinguisher equip/stow, independent of YOLO detection.
- Right trigger: hold to spray; aim at the flame for two seconds to extinguish it.

`manualDemonstrationMode` is enabled. Automatic tool acquisition is disabled and `automaticSafetyStop` is disabled on the three forklift components so a human can demonstrate braking. Danger remains observable and warns the user to press B. A nohelmet detection is a perception cue; the human's A warning is a separate action. Do not leave the training forklift unattended in this manual scenario. Inspector can re-enable automatic safety braking for assisted operation.

The right A jump binding is temporarily overridden while the action component is enabled and restored on disable. Walking, turning and head/controller tracking remain in the existing XR configuration.

Recorder samples now separate `collision_risk` from `collision_stop` (manual command), and include actual `helmet_warning`, `extinguisher_equipped`, `control_mode`, and controller availability. Immediate `kind: human_action` records preserve short button presses between 10 Hz samples. Synthetic-controller validation confirmed A/B/X/trigger callbacks, stop/resume, equip/stow and actual flame suppression; this does not verify physical Quest hardware. Older validation sessions recorded perception flags as action labels and must not be used as new human demonstrations.

Scene: `Assets/Scenes/VR_Warehouse.unity`.

`WarehouseSafetyActions` subscribes to accepted YOLO results. Explicit `fire` creates/shows the right-controller extinguisher. Right-hand trigger sprays along `Extinguisher Aim Nozzle`; sustained aim for two seconds suppresses the actual fire. Walls block the spray. Explicit `nohelmet` must have an associated YOLO person box before issuing `Wear a helmet`. Helmet absence alone is not a violation.

`WarehouseForkliftSafety` predicts pedestrians inside a forward corridor and stops its independent PlayableDirector or waypoint driver. This is a geometric safety interlock, not a claim that 2D YOLO boxes provide 3D collision prediction. Existing forklift and reachlift animation bindings have separate directors so one vehicle can stop independently. A third training forklift encounters a helmetless pedestrian on the central aisle. Moving that pedestrian aside releases the interlock.

`WarehouseTrainingScenario` supports resetting/repeating episodes. `WarehouseSafetyActions` exposes `LatestResult` through its bridge, `PerceptionReceived`, `ActionResult`, `GetActionResults`, `GetTrainingObservations`, `SetAgentSpray`, `SetAgentAim`, and `SetForkliftStop`. Observations mix vision flags with explicitly geometric safety state; do not treat geometric state as a vision prediction.

`WarehouseActionRecorder` stores 10 Hz JSONL demonstrations in Unity's persistent data path under `SafetyDemonstrations`. It records head/right-controller poses, accepted and raw perception, spraying, braking, warning, and extinguishing outcomes. Enable recording during actual human demonstrations for imitation learning. The initial validation sessions contain synthetic test actions and must not be treated as skilled human demonstrations. ML-Agents is not installed and no learned motion policy has been trained/deployed by this change.

Person training reuses 312 training / 62 validation warehouse images. Explicit original dataset label `3: person` is converted into a NEW single-class training dataset `0: person`; the old numeric-only person model is not semantically relabeled. All 188 final-holdout images remain outside training. Original weights remain intact. The renderer-based bounding boxes are approximate and distant/occluded persons can still be missed.

Start the server from the project root:

```powershell
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu
```

Editor validation: geometric stop/resume, YOLO-triggered extinguisher visibility, nohelmet warning, sustained spray destroying fire, `fire_extinguished` event, recording and compilation. Actual Quest controller trigger response, attachment fit and hand/head tracking must be checked with the headset connected.
