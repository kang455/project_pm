# Applied warehouse changes, 2026-10-05

Saved scene: Assets/Scenes/VR_Warehouse.unity. Play stopped after verification.

The XR TunnelingVignetteController and its dedicated mask GameObject are disabled. DynamicMoveProvider, SnapTurnProvider, ContinuousTurnProvider and active tracked-pose drivers remain enabled. No Volume vignette was found. Physical input and display require Quest verification.

The replaced Tools/YoloBridge/helmet.pt was loaded directly, then the Python server was restarted. Old in-memory helmet architecture was yolo26s.yaml; loaded server now reports yolo26m.yaml, DetectionModel, detect. New file SHA256 AC7F0E39DCED071CED6E60991CCC66711DD07FE7F37AFC39B3D7F9AAFCD59631. Class 0 object, class 1 safety_helmet_detection_0321 - v3 2024-03-27 8-32pm. No helmet/head/no_helmet class exists. Names are not remapped. LoadedModels.json records the server state. HelmetPair_response.json has no helmet detections. Missing helmet detection never means confirmed non-compliance.

All 17 worker helmet SkinnedMeshRenderers are separate from their heads and disabled in this scene. All 17 heads remain enabled. Nine blue Scooter helmet instances are parented to Bip001 Head and aligned over their head bounds; eight workers are bareheaded. Only Assets/Scooter helmet/VR/HelmetBlueURP.mat was created/colored; original body materials are unchanged. HelmetPair.jpg shows a bareheaded and a blue-helmeted worker together. Local helmet attachment offsets remain fixed as the head bones animate. HelmetGroundTruth_ValidationOnly.json is validation metadata only and is not read by YOLO.

WarehouseRandomFire.cs uses the imported VFX_Fire_01_Medium_Simple.prefab. Six floor points are connected with a reference to the XR head. Inspector controls interval, duration, maximum fires, scale and clearance. Spawn-time checks include upward-facing floor, floor height, obstacle capsule, distance to the player and other fires, and animated body renderer bounds. Validation observed nine created and nine expired effects over a live run, with a configured maximum of two. Validation overrides were runtime-only and did not change saved defaults (8–16 second interval, 6–12 second lifetime).

The imported fire shader's depth-fade needs Depth Texture. It was initially invisible despite live particles; Depth Texture is now enabled on the XR camera and separately in YOLO camera configuration. FireDepthFixed.png is the XR Game screen and FireDepthFixed.jpg is the actual detection image, both showing flames.

Comparison_0/2.jpg have no validation fire; Comparison_1/3.jpg have the real imported fire VFX at two sizes/distances. They were sent as JPEGs to the running server after pausing the competing Unity requester. Neither negative image produced raw/accepted fire. Both fire images produced raw fire candidates but no approved fire under the prior 0.90 class-0 floor. This demonstrates missed displayed detections; it does not establish reliable fire recognition. The earlier 12-view no-fire hard-negative comparison in .vscode/FireAudit still provides the recorded yellow-object false-positive evidence. The actual positive candidates are weaker than some negative candidates, so a confidence-only threshold cannot solve both issues on these examples. Representative rendered fire positives and warehouse hard negatives are needed for model evaluation/retraining. Fire-spawner states do not influence server or UI decisions.

User UI remains names only, with long original names shortened solely for display; raw full names remain in results. No input-image preview, confidence, percentages or diagnostics are shown. Empty results hide the panel. Final Console errors: zero. Missing scene object references: zero.

Server remains running on 127.0.0.1:8765. Restart command from project root, after stopping the existing process:

```powershell
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu
```
