using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
public class WarehouseSafetyActions : MonoBehaviour {
 public YoloMultiBridge bridge;
 public Transform rightController,head,toolRoot,nozzle;
 public GameObject extinguisher;
 public Text warningText;
 public WarehouseForkliftSafety[] forklifts;
 public WarehouseVisionCollisionRisk visionCollision;
 public float resultLifetime=3,sprayRange=7,sprayRadius=.45f,triggerThreshold=.2f;
 public bool aiControl,aiSpray,showWarnings=true;
 public bool manualDemonstrationMode=true;
 public float helmetWarningSeconds=4;
 public bool ManualForkliftStopped {get;private set;}
 public bool HelmetWarningActive=>Time.time<helmetWarningUntil;
 public bool ExtinguisherEquipped=>extinguisher&&extinguisher.activeSelf;
 public event Action<string> HumanAction;
 float helmetWarningUntil;
 class SuppressedBinding {public InputAction action;public int index;public string path;}
 readonly List<SuppressedBinding> suppressedJump=new List<SuppressedBinding>();
 void ReserveAButton(){
  foreach(var asset in Resources.FindObjectsOfTypeAll<InputActionAsset>())foreach(var map in asset.actionMaps)foreach(var action in map.actions){
   if(action.name!="Jump")continue;
   for(int i=0;i<action.bindings.Count;i++){
    var binding=action.bindings[i];if(binding.path==null||!binding.path.Contains("RightHand")||binding.path.IndexOf("PrimaryButton",StringComparison.OrdinalIgnoreCase)<0)continue;
    suppressedJump.Add(new SuppressedBinding{action=action,index=i,path=binding.overridePath});action.ApplyBindingOverride(i,string.Empty);
   }
  }
 }
 void RestoreAButton(){foreach(var b in suppressedJump){if(b.path==null)b.action.RemoveBindingOverride(b.index);else b.action.ApplyBindingOverride(b.index,b.path);}suppressedJump.Clear();}
 InputAction helmetButton,stopButton,equipButton;
 void RecordHuman(string command){if(aiControl){Publish(command.Replace("human_","agent_"));return;}HumanAction?.Invoke(command);Publish(command);}
 public void WarnHelmet(){helmetWarningUntil=Time.time+helmetWarningSeconds;RecordHuman("human_helmet_warning");}
 public void ToggleForkliftStop(){ManualForkliftStopped=!ManualForkliftStopped;SetForkliftStop(ManualForkliftStopped);RecordHuman(ManualForkliftStopped?"human_forklift_stop":"human_forklift_resume");}
 public void ToggleExtinguisher(){if(!extinguisher)return;extinguisher.SetActive(!extinguisher.activeSelf);RecordHuman(extinguisher.activeSelf?"human_extinguisher_equip":"human_extinguisher_stow");}
 bool previousSpraying;
 public ParticleSystem sprayParticles;
 public bool FireDetected {get;private set;}
 public bool NoHelmetDetected {get;private set;}
 public bool CollisionRisk {get;private set;}
 public bool Spraying {get;private set;}
 public bool PersonDetected {get;private set;}
 public bool HasFreshPerception=>Time.time-lastResult<=resultLifetime;
 public event Action<string> ActionResult;
 public event Action<YoloResponse> PerceptionReceived;
 public int ExtinguishedCount=>WarehouseExtinguishableFire.TotalExtinguished;
 float lastResult=-100;
 bool lastCollision,lastHelmet,lastFire;
 Vector3 agentAimPosition;Quaternion agentAimRotation;bool agentAimSet;
 public void SetAgentAim(Vector3 position,Quaternion rotation){agentAimPosition=position;agentAimRotation=rotation;agentAimSet=true;}
 void LateUpdate(){if(aiControl&&agentAimSet&&toolRoot)toolRoot.SetPositionAndRotation(agentAimPosition,agentAimRotation);}
 InputAction trigger;
 readonly Queue<string> results=new Queue<string>();
 public string[] GetActionResults()=>results.ToArray();
 public void SetAgentSpray(bool value){aiSpray=value;}
 public void SetAgentEquip(bool equipped){if(ExtinguisherEquipped!=equipped)ToggleExtinguisher();}
 public void SetAgentForkliftStop(bool stop){if(ManualForkliftStopped==stop)return;ManualForkliftStopped=stop;SetForkliftStop(stop);Publish(stop?"agent_forklift_stop":"agent_forklift_resume");}
 public void SetForkliftStop(bool value){foreach(var f in forklifts)if(f)f.SetStop(value);}
 void OnEnable(){
  if(bridge)bridge.ResultReceived+=Receive;
  ReserveAButton();
  trigger=new InputAction("Extinguisher Trigger",InputActionType.Value,"<XRController>{RightHand}/trigger");
  trigger.Enable();
  helmetButton=new InputAction("Helmet warning A",InputActionType.Button,"<XRController>{RightHand}/primaryButton");
  stopButton=new InputAction("Forklift stop B",InputActionType.Button,"<XRController>{RightHand}/secondaryButton");
  equipButton=new InputAction("Extinguisher X",InputActionType.Button,"<XRController>{LeftHand}/primaryButton");
  helmetButton.performed+=c=>{if(!aiControl)WarnHelmet();};
  stopButton.performed+=c=>{if(!aiControl)ToggleForkliftStop();};
  equipButton.performed+=c=>{if(!aiControl)ToggleExtinguisher();};
  helmetButton.Enable();stopButton.Enable();equipButton.Enable();
 }
 void OnDisable(){RestoreAButton();if(bridge)bridge.ResultReceived-=Receive;trigger?.Dispose();helmetButton?.Dispose();stopButton?.Dispose();equipButton?.Dispose();if(sprayParticles)sprayParticles.Stop();}
 public void ResetPerception(){lastResult=-100;FireDetected=NoHelmetDetected=PersonDetected=CollisionRisk=false;if(visionCollision)visionCollision.Clear();}
 void Receive(YoloResponse response){
  lastResult=Time.time;FireDetected=false;NoHelmetDetected=false;PersonDetected=false;
  foreach(var d in response.detections){
   if(d.class_name=="fire")FireDetected=true;
   if(d.class_name=="nohelmet"&&d.person_xyxy!=null&&d.person_xyxy.Length==4)NoHelmetDetected=true;
   if(d.class_name=="person")PersonDetected=true;
  }
  CollisionRisk=visionCollision&&visionCollision.Evaluate(response);
  PerceptionReceived?.Invoke(response);
 }
 void Update(){
  if(Time.time-lastResult>resultLifetime){FireDetected=NoHelmetDetected=PersonDetected=CollisionRisk=false;if(visionCollision)visionCollision.Clear();}
  // Keep the acquired tool available through brief detector dropouts.
  if(!manualDemonstrationMode&&FireDetected&&extinguisher&&!extinguisher.activeSelf)extinguisher.SetActive(true);
  Spraying=extinguisher&&extinguisher.activeInHierarchy&&(aiControl?aiSpray:trigger!=null&&trigger.ReadValue<float>()>triggerThreshold);
  if(Spraying!=previousSpraying)RecordHuman(Spraying?"human_spray_start":"human_spray_stop");
  previousSpraying=Spraying;
  if(sprayParticles){if(Spraying&&!sprayParticles.isPlaying)sprayParticles.Play();if(!Spraying&&sprayParticles.isPlaying)sprayParticles.Stop();}
  if(Spraying&&nozzle){
   foreach(var fire in FindObjectsByType<WarehouseExtinguishableFire>(FindObjectsSortMode.None)){
    if(fire.Extinguished)continue;
    var target=fire.transform.position+Vector3.up*.6f;var offset=target-nozzle.position;
    float along=Vector3.Dot(offset,nozzle.forward);
    if(along<=0||along>sprayRange||(offset-nozzle.forward*along).magnitude>sprayRadius+along*.07f)continue;
    RaycastHit hit;
    if(Physics.Raycast(nozzle.position,offset.normalized,out hit,offset.magnitude,~(1<<5),QueryTriggerInteraction.Ignore)&&hit.distance<offset.magnitude-.35f)continue;
    bool before=fire.Extinguished;fire.Suppress(Time.deltaTime);
    if(!before&&fire.Extinguished)Publish("fire_extinguished");
   }
  }
  if(CollisionRisk&&!lastCollision)Publish("collision_risk_detected");
  if(NoHelmetDetected&&!lastHelmet)Publish("nohelmet_detected");
  if(FireDetected&&!lastFire)Publish("fire_detected");
  lastCollision=CollisionRisk;lastHelmet=NoHelmetDetected;lastFire=FireDetected;
  if(warningText){
   warningText.text=(CollisionRisk?(ManualForkliftStopped?"Danger - forklift stopped\n":aiControl?"Possible collision\n":"Possible collision - press B to stop\n"):ManualForkliftStopped?"Forklift stopped\n":"")+(HelmetWarningActive?"Wear a helmet\n":NoHelmetDetected?(aiControl?"No helmet detected\n":"No helmet - press A to warn\n"):"")+(FireDetected?(aiControl?"Fire detected":"Fire - X equip / trigger spray"):"");
   warningText.gameObject.SetActive(showWarnings&&warningText.text.Length>0);
  }
 }
 void Publish(string result){results.Enqueue(result);while(results.Count>64)results.Dequeue();ActionResult?.Invoke(result);}
 public float[] GetTrainingObservations(){
  float nearest=visionCollision?visionCollision.ClosestImageGap:-1;
  return new[]{PersonDetected?1f:0,FireDetected?1f:0,NoHelmetDetected?1f:0,CollisionRisk?1f:0,Spraying?1f:0,nearest,ExtinguishedCount};
 }
}