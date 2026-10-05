Continuous latest-frame video validation (2026-10-05)

Scene: Assets/Scenes/VR_Warehouse.unity. Continuous Video Stream enabled; capture target 10 FPS, 960x540. TCP 127.0.0.1:8766 for persistent video; HTTP 8765 retained for diagnostics. Server loads the same six models once. Lying person remains excluded. Model mappings and fire/nohelmet filters preserved.

Unity Play: at 31.39 seconds, 164 frames sent, eight accepted responses, zero failures. Warm-up affected early startup. Later 129 frames were sent over 16.03 seconds (~8 FPS); response processing ~1.8-1.9 seconds on CPU (~0.5 inference FPS). Server had received 497 frames, processed 31, discarded 466 superseded frames. Only one waiting frame per client; no FIFO inference backlog. Transport is on background threads; AsyncGPUReadback remains enabled.

matched.jpg and matched.json are the same completed frame 471. Actual nohelmet class 0 associated with forklift model person class 3. Names-only UI showed No helmet/person. Raw fire candidate was retained in diagnostics and rejected for not_flame_shape. This limited negative scene check does not establish general fire accuracy. No image/confidence/percent displayed to users.

Scene missing scripts: zero; Console error entries: zero. Imported prefab warnings and XR audio/eye-tracking warnings also exist; no physical Quest interaction verified. Runtime camera move was for verification only and reverted on exiting Play.

PowerShell from project root:
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu

Video is still decoded/inferred frame by frame. This change reduces stale-frame waiting and separates capture from inference; it does not imply 30 FPS detection or fix model accuracy by itself. Installed Torch is CPU-only; no GPU package/environment changes were made.
