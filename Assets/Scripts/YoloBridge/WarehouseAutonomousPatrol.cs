using UnityEngine;
using UnityEngine.AI;
using UnityEngine.UI;
[DefaultExecutionOrder(-200)]
public class WarehouseAutonomousPatrol:MonoBehaviour {
 public WarehouseLearnedAgent learnedAgent;public WarehouseSafetyActions actions;public YoloMultiBridge bridge;
 public NavMeshAgent navigator;public NavMeshData navigationData;public Camera aiCamera;
 public Transform[] patrolPoints;public Transform leftLeg,rightLeg,rightArm,rightHand;
 public Canvas aiCanvas;public Text aiNames,aiWarnings;
 public bool automaticPatrol=true,recordingViewOnPC;
 public float walkingSpeed=1.2f,inspectionSeconds=3,lookAroundSeconds=2;
 public float fireStandOff=1.8f,warningStandOff=2.2f,maximumFireHandlingSeconds=35;
 public string State {get;private set;}="waiting";
 public float DistanceWalked {get;private set;} public int WaypointsVisited {get;private set;}
 public bool Active {get;private set;} public int WaypointIndex {get;private set;}
 public bool ReadyForFire {get;private set;} public bool ReadyForHelmet {get;private set;}
 public float TargetDistance {get;private set;}=-1;
 NavMeshDataInstance meshInstance;Camera humanCamera;Transform humanHead;
 bool humanFovMatch;float nextPathUpdate;float scanUntil,seenAt=-100,targetStarted,nextWarningVisit;
 Vector3 previousPosition,target;string targetKind;long lastFrame=-1;
 void Awake(){humanCamera=bridge.xrCamera;humanHead=actions.head;humanFovMatch=bridge.matchXrFieldOfView;previousPosition=transform.position;}
 void OnEnable(){
  if(!Application.isPlaying)return;
  if(navigationData)meshInstance=NavMesh.AddNavMeshData(navigationData);
  NavMeshHit hit;
  if(!meshInstance.valid||!NavMesh.SamplePosition(transform.position,out hit,1,NavMesh.AllAreas)){State="no walkable start";Debug.LogWarning("AI patrol: no valid baked navigation start.",this);enabled=false;return;}
  navigator.enabled=true;navigator.Warp(hit.position);navigator.speed=walkingSpeed;navigator.isStopped=true;
 }
 void Source(bool active){
  Active=active;bridge.matchXrFieldOfView=active?false:humanFovMatch;bridge.SetSourceCamera(active?aiCamera:humanCamera);
  actions.head=active?aiCamera.transform:humanHead;actions.ResetPerception();aiCamera.enabled=active;
  if(aiCanvas)aiCanvas.enabled=false;
  targetKind=null;ReadyForFire=ReadyForHelmet=false;
  if(!active&&navigator&&navigator.isOnNavMesh){navigator.ResetPath();navigator.isStopped=true;}
  if(active){scanUntil=0;lastFrame=-1;previousPosition=transform.position;GoToWaypoint();}
 }
 bool Reachable(Vector3 point,out Vector3 sampled){
  NavMeshHit hit;sampled=point;if(!NavMesh.SamplePosition(point,out hit,1.5f,NavMesh.AllAreas))return false;
  var path=new NavMeshPath();if(!navigator.CalculatePath(hit.position,path)||path.status!=NavMeshPathStatus.PathComplete)return false;
  sampled=hit.position;return true;
 }
 void GoToWaypoint(){
  if(patrolPoints==null||patrolPoints.Length==0)return;
  for(int i=0;i<patrolPoints.Length;i++){
   Vector3 destination;var p=patrolPoints[WaypointIndex];
   if(p&&Reachable(p.position,out destination)){navigator.stoppingDistance=.15f;navigator.SetDestination(destination);return;}
   WaypointIndex=(WaypointIndex+1)%patrolPoints.Length;
  }State="no reachable patrol point";
 }
 YoloDetection Find(string name){
  if(!actions.HasFreshPerception||bridge.LatestResult==null||bridge.LatestResult.detections==null)return null;
  YoloDetection best=null;
  foreach(var d in bridge.LatestResult.detections)if(d.class_name==name&&d.xyxy!=null&&d.xyxy.Length==4&&(best==null||d.confidence>best.confidence))best=d;
  return best;
 }
 bool Locate(YoloDetection detection,out Vector3 ground){
  ground=Vector3.zero;var response=bridge.LatestResult;
  var box=detection.class_name=="nohelmet"&&detection.person_xyxy!=null&&detection.person_xyxy.Length==4?detection.person_xyxy:detection.xyxy;
  Ray ray;float x=(box[0]+box[2])*.5f/response.width,y=1-box[3]/response.height;
  if(!bridge.TryGetFrameRay(response.frame_id,new Vector2(x,y),out ray)||ray.direction.y>=-.015f)return false;
  // Project detected feet/base onto local floor, then verify real floor and a complete path.
  float distance=(transform.position.y-ray.origin.y)/ray.direction.y;
  if(distance<0||distance>25)return false;
  var estimate=ray.GetPoint(distance);RaycastHit hit;
  int mask=~((1<<5)|(1<<gameObject.layer));
  if(!Physics.Raycast(estimate+Vector3.up*1.5f,Vector3.down,out hit,3,mask,QueryTriggerInteraction.Ignore)||hit.normal.y<.65f)return false;
  return Reachable(hit.point,out ground);
 }
 void ObserveTarget(YoloDetection detection,string kind){
  Vector3 ground;if(!Locate(detection,out ground))return;
  if(targetKind!=kind){targetKind=kind;targetStarted=Time.time;nextPathUpdate=0;target=ground;}
  else if(Vector3.Distance(target,ground)<3)target=Vector3.Lerp(target,ground,.3f);
  seenAt=Time.time;
 }
 void FaceTarget(){
  var flat=target-transform.position;flat.y=0;
  if(flat.sqrMagnitude>.01f)transform.rotation=Quaternion.RotateTowards(transform.rotation,Quaternion.LookRotation(flat),90*Time.deltaTime);
  var look=target+Vector3.up*(targetKind=="fire"?.65f:1.4f)-aiCamera.transform.position;
  if(look.sqrMagnitude>.01f)aiCamera.transform.rotation=Quaternion.RotateTowards(aiCamera.transform.rotation,Quaternion.LookRotation(look),55*Time.deltaTime);
 }
 public void GetToolPose(out Vector3 position,out Quaternion rotation){
  position=rightHand?rightHand.position:transform.TransformPoint(new Vector3(.30f,1.04f,.38f));
  var aim=targetKind=="fire"?target+Vector3.up*.6f:position+transform.forward*3;
  rotation=Quaternion.LookRotation(aim-position,Vector3.up);
 }
 void ClearTarget(){targetKind=null;TargetDistance=-1;ReadyForFire=ReadyForHelmet=false;GoToWaypoint();}
 void Update(){
  if(!navigator||!navigator.enabled||!navigator.isOnNavMesh)return;
  bool requested=automaticPatrol&&learnedAgent&&learnedAgent.runAI;if(requested!=Active)Source(requested);
  aiCamera.targetDisplay=recordingViewOnPC?0:1;if(aiCanvas)aiCanvas.targetDisplay=aiCamera.targetDisplay;
  if(!Active){navigator.isStopped=true;State="manual VR control";return;}
  DistanceWalked+=Vector3.Distance(transform.position,previousPosition);previousPosition=transform.position;
  navigator.speed=walkingSpeed;ReadyForFire=ReadyForHelmet=false;
  if(!actions.HasFreshPerception){navigator.isStopped=true;actions.SetAgentSpray(false);State="waiting for AI vision";return;}
  var response=bridge.LatestResult;
  if(response!=null&&response.frame_id!=lastFrame){
   lastFrame=response.frame_id;var fire=Find("fire");var helmet=Find("nohelmet");
   if(fire!=null)ObserveTarget(fire,"fire");
   else if(Time.time>=nextWarningVisit&&helmet!=null&&targetKind!="fire")ObserveTarget(helmet,"helmet");
  }
  if(targetKind!=null){
   if(Time.time-seenAt>3||Time.time-targetStarted>maximumFireHandlingSeconds){nextWarningVisit=Time.time+8;ClearTarget();}
   else{
    TargetDistance=Vector3.Distance(new Vector3(target.x,transform.position.y,target.z),transform.position);
    float standOff=targetKind=="fire"?fireStandOff:warningStandOff;
    bool near=TargetDistance<=standOff+.25f;
    navigator.updateRotation=!near;navigator.stoppingDistance=standOff;
    if(!near){if(Time.time>=nextPathUpdate){navigator.SetDestination(target);nextPathUpdate=Time.time+.8f;}navigator.isStopped=false;State=targetKind=="fire"?"walking to detected fire":"walking to detected person";}
    else{navigator.isStopped=true;FaceTarget();State=targetKind=="fire"?"extinguishing nearby fire":"warning nearby person";}
    ReadyForFire=near&&targetKind=="fire"&&actions.FireDetected;
    ReadyForHelmet=near&&targetKind=="helmet"&&actions.NoHelmetDetected;
    if(!ReadyForFire)actions.SetAgentSpray(false);
    if(ReadyForHelmet&&actions.HelmetWarningActive){nextWarningVisit=Time.time+10;targetKind=null;scanUntil=Time.time+inspectionSeconds;GoToWaypoint();}
    return;
   }
  }
  navigator.updateRotation=true;
  if(Time.time<scanUntil){navigator.isStopped=true;State="looking around";float yaw=Mathf.Sin(Time.time*1.8f)*45;aiCamera.transform.localRotation=Quaternion.Slerp(aiCamera.transform.localRotation,Quaternion.Euler(0,yaw,0),Time.deltaTime*2);return;}
  navigator.isStopped=false;State="patrolling";aiCamera.transform.localRotation=Quaternion.Slerp(aiCamera.transform.localRotation,Quaternion.identity,Time.deltaTime*2);
  if(!navigator.pathPending&&navigator.hasPath&&navigator.remainingDistance<.4f){WaypointsVisited++;WaypointIndex=(WaypointIndex+1)%patrolPoints.Length;scanUntil=Time.time+lookAroundSeconds;GoToWaypoint();}
  else if(!navigator.hasPath)GoToWaypoint();
 }
 void LateUpdate(){
  if(Active&&actions.aiControl){Vector3 position;Quaternion rotation;GetToolPose(out position,out rotation);actions.SetAgentAim(position,rotation);}

  if(aiCanvas){if(aiNames)aiNames.text=bridge.resultText.text;if(aiWarnings)aiWarnings.text=actions.warningText.text;aiCanvas.enabled=Active&&(!string.IsNullOrEmpty(aiNames.text)||!string.IsNullOrEmpty(aiWarnings.text));}
  float step=navigator&&navigator.enabled&&navigator.isOnNavMesh?navigator.velocity.magnitude:0;
  if(leftLeg)leftLeg.localRotation=Quaternion.Euler(Mathf.Sin(Time.time*7)*25*Mathf.Clamp01(step),0,0);
  if(rightLeg)rightLeg.localRotation=Quaternion.Euler(-Mathf.Sin(Time.time*7)*25*Mathf.Clamp01(step),0,0);
  if(rightArm){
   var shoulder=transform.TransformPoint(new Vector3(.25f,1.15f,0));var hand=rightHand?rightHand.position:transform.TransformPoint(new Vector3(.30f,1.04f,.38f));
   // Fixed hand mount bounds the arm length independently of inferred pose and camera motion.
   rightArm.position=(shoulder+hand)*.5f;rightArm.rotation=Quaternion.FromToRotation(Vector3.up,hand-shoulder);
   rightArm.localScale=new Vector3(.11f,Mathf.Min(.25f,Vector3.Distance(shoulder,hand)*.5f),.11f);
  }
 }
 void OnDisable(){if(Active&&bridge&&actions)Source(false);if(navigator)navigator.enabled=false;if(meshInstance.valid)meshInstance.Remove();}
}