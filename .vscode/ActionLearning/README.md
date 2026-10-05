# VR demonstration behavioral cloning

Input: the three user-supplied ZIP archives in Unity's `SafetyDemonstrations` directory. Originals are read without extraction or modification. The archive audit and training report preserve source hashes and per-file splits. Seventeen sessions contain 5 helmet warning presses, 9 stop / 2 resume presses, 5 extinguisher equip presses and 7 spray-start / 7 spray-stop presses.

Training is offline behavioral cloning, not reinforcement learning or a prewritten response rule substituted for a trained model. Four small neural networks predict helmet-warning, vehicle-stop, equip and spray states from accepted YOLO box features plus the existing image-space risk/gap. A fifth regresses head-relative right-controller position and directional residual from actual spraying demonstrations. No Unity actor positions, fire spawn state, clothing configuration, or physics ground truth are inputs to these networks.

Session files, rather than adjacent frames, are split into training, validation and test. Each topic's last file is held out for test and next-to-last for validation. Thresholds/early stopping are chosen only on validation. Metrics count frames, not independent events or task success. Only one test session per topic is available, so estimates are unstable.

The first model's test frame results:

| Network | True positive | False positive | False negative |
|---|---:|---:|---:|
| Helmet warning | 12 | 0 | 4 |
| Forklift stop | 5 | 5 | 17 |
| Equip | 45 | 25 | 0 |
| Spray | 19 | 24 | 0 |

Forklift decision generalization is weak; equip/spray tend to act earlier than the human. The absence of dedicated normal-scene demonstrations and very few completed episodes limit generalization. These models are experimental and have not passed an autonomous safety acceptance criterion. The scene defaults to human control (`Run AI` off).

`WarehouseLearnedAgent` loads `Assets/Scripts/YoloBridge/Models/WarehouseActionPolicy.asset`. Inspector toggle `Run AI` enables the trained policy, while individual task toggles allow isolated trials. Runtime availability guards require the relevant fresh accepted detections; the learned probabilities still determine the action. Stop hold/cooldown and stale-perception cancellation are control safeguards. B human inputs are ignored in AI mode; turn `Run AI` off or stop Play to return to human control.

The extinguisher tool moves as an AI-controlled proxy using the learned head-relative hand pose; this does not animate a full humanoid skeleton or simulate a physical robot arm. During AI mode, the tool is parented to the XR head so hardware controller availability does not hide it. Controller tracking itself remains untouched and manual parenting is restored when AI mode ends. Aim reconstruction assumes approximately 90-degree capture vertical field of view because original demonstrations did not record projection metadata. The existing physical spray/occlusion simulation determines actual extinguishing outcomes.

The pure C# network evaluator matches PyTorch on held-out fixtures (maximum error about 6e-8). Play testing confirmed learned helmet warnings and one actual fire extinguished by the learned equip/aim/spray actions without controller button injection. A live-camera trial using a training-session viewpoint also produced the learned stop command and paused both original vehicle directors. No recorded detection response was injected into those live trials. These are specific scene trials, not evidence that every hazard can be resolved.

Stop commands are latched by default because there are only two resume demonstrations. `Release Stop When Clear` is off; disable AI mode to return to manual control and release AI braking. Automatic release can be explicitly tested via its Inspector toggle but is not validated as a safe resume policy. AI warnings omit human button instructions.

New recordings distinguish `human_action` from `agent_action`, and sample `control_mode` remains separate. Training filters out agent-mode samples, inactive controller samples and records without schema-2 perception. Head/controller samples and perception were recorded at different latency points; the original files contain no images or precise capture poses, so this pipeline is not an end-to-end image policy and moving-head aim can be inaccurate.

Retrain from project root:

```powershell
& '.\Tools\YoloBridge\.venv\Scripts\python.exe' '.\.vscode\ActionLearning\train.py'
```

The Python script exports a candidate `policy.json` and reports locally; it does not silently overwrite the Unity asset. Import/verify the candidate before replacing the scene's trained policy. Current Python requirements are the already installed NumPy and PyTorch.

## Autonomous patrol
Scene: Assets/Scenes/VR_Warehouse.unity. Warehouse Patrol AI uses baked NavMesh and nine patrol points. Warehouse Safety Actions > Warehouse Learned Agent > Run AI enables the separate robot and its image source; disabling restores manual VR perception and stops the robot. AI Patrol Camera outputs Display 2. Navigation is programmed, safety actions use the existing learned policy. Physical geometry is used for navigation and suppression, not for replacing YOLO danger results.
Play validation: 5.75 m walked, one waypoint reached, 27 image responses, zero failed requests, two helmet warning commands and one forklift stop command. Manual camera restored and zero navigation velocity verified. Console errors and missing scripts: zero. Extinguishing during this patrol run and physical Quest viewing were not verified.


## Additional scripted teacher training (2026-10-05)
Completed 15 live Unity/YOLO sessions: five helmet, five forklift, five fire; each 18 distinct detection-response frames. Stored separately in persistentDataPath/SafetyDemonstrations/AutomaticTeacher/20261005_204635. These are scripted-teacher examples, not human-controller demonstrations or self-labeled AI outputs. Accepted YOLO boxes drive teacher decisions; known fire positions only demonstrate the desired hand pose. No ground-truth positions enter deployed neural-network features. Last automated session of each topic excluded from fitting. Four small feature-noise variants generated for classifier fitting only. Human validation/test session split preserved.
Only the stop network passed validation and frozen-test regression acceptance and was deployed. Frozen human stop test: TP 5 to 9, FP unchanged at 5, FN 17 to 13, F1 .3125 to .50. Helmet, equip, spray and aim candidates did not improve or regressed and original heads are retained. This limited frame-based test does not establish broad task success; detector misses still prevent action. See reinforced/comparison.json. Original policy retained for rollback.
New scene policy: Assets/Scripts/YoloBridge/Models/WarehouseActionPolicyReinforced.asset. Play validation: native export error 5.96e-8, separate AI walked 5.60 m and issued one stop and two warnings. No Console errors. Play stopped. Offline Action Teacher is wired but Run On Play is false so ordinary Play does not generate teacher sessions.


## Patrol tool and approach correction
Right Hand Tool Mount is now a fixed body child. Patrol updates the tool pose every LateUpdate, including patrol/stow states; cached world-space aim must not freeze the tool while the parent walks. Arm visual endpoints use the fixed mount and maximum half-length .25 m, never unconstrained predicted controller translations. YOLO bridge stores bounded per-frame camera position/rotation/FOV/aspect so image-derived floor goals use capture-time projection. NavMesh goals update at .8 second intervals. Fire approach standoff 1.8 m; helmet warning approach standoff 2.2 m. Fire spray and helmet warning are gated until close. Existing learned policy remains loaded, with explicit Execute Confirmed Nearby Actions fallback after three distinct accepted YOLO frames near a target. This fallback and geometric tool aiming are programmed controllers, not learned behavior or scene-ground-truth detection. Forklift braking remains immediate rather than waiting for approach.
Live validation: a helmet warning after approaching to 1.89 m; two random fires extinguished in the final run, 20.65 m travelled, resumed patrol. Tool/mount distance 0.0000038 m during approach and after extinguishing, arm scale (.11,.20,.11). No Console errors and zero missing scene scripts. Physical Quest viewing not checked. Scene VR_Warehouse saved; Play stopped.

