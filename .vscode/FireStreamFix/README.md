Fire/forklift change, 2026-10-05

Cause reproduced using original imported medium fire prefab at scale 1, 5m from capture camera. YOLO fire.pt detected original fire class 0, but motion verifier rejected it as static_warm_object. Pixel motion was below the original >25 intensity / 12% warm-pixel criterion.

Server now uses >8 / 1% for weak temporal changes and allows a second YOLO confirmation on a padded crop when temporal evidence is absent/weak. Crop confirmation requires actual fire class, confidence >=0.55, box IoU >0.3, and candidate aspect 0.25..1.2. Original shape/warm-color gates remain. Original class IDs/names and raw candidates preserved. Scene fire state is never fed into detector.

Final saved-JPEG replay: both newly captured scale-1 flame frames accepted; previous six positive frames accepted 5/6 (one had no raw YOLO fire at all). Twelve saved no-fire views all zero accepted fire. This limited replay is not a general accuracy guarantee; distant/occluded flames and camera motion still require further testing.

Forklift model final public detections limited to class_name forklift. Internal person detections remain available during nohelmet crop/association steps; container/crane/person/ship/stacker/truck from that model are not exposed as final detections/UI. Other models remain enabled. Original raw results remain diagnostic.

Server command unchanged: from project root
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\Tools\YoloBridge\server.py' --device cpu

Final detail fallback changed from 960 to 1280 inference size. Live Play confirmed fire label; after removing flame confirmed no fire and forklift label. Console error entries zero. Temporary viewpoint/flame reverted on Play stop. Final 12 negative replays still zero. Expanded confirmation adds CPU work; real-time high-FPS inference is not achieved.

