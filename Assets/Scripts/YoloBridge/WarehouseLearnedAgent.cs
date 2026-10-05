using System;
using UnityEngine;
// Behavioral cloning from the supplied VR demonstrations; inference uses YOLO boxes only.
public class WarehouseLearnedAgent : MonoBehaviour {
 public WarehouseSafetyActions actions;
 public WarehouseAutonomousPatrol patrol;
 public bool executeConfirmedNearbyActions=true;
 long nearbyFrame=-1;int nearbyHelmetFrames,nearbyFireFrames;
 public TextAsset trainedPolicy;
 public bool runAI;
 public bool enableHelmet=true,enableForklift=true,enableFire=true;
 public float decisionsPerSecond=5,warningCooldown=6,minimumStopSeconds=3;
 public bool releaseStopWhenClear=false;
 public float fireFieldOfView=90;
 public string LastDecision {get;private set;}="manual";
 public int Decisions {get;private set;}
 public int WarningsIssued {get;private set;}
 public int StopCommands {get;private set;}
 public int EquipCommands {get;private set;}
 public int SprayCommands {get;private set;}
 public float[] LastProbabilities {get;private set;}=new float[4];
 [Serializable] public class Layer {public int inputs,outputs;public float[] weights,bias;}
 [Serializable] public class Model {public string name;public bool motion;public float threshold;public float[] inputMean,inputScale,outputMean,outputScale;public Layer[] layers;}
 [Serializable] public class Fixture {public string name;public float[] features,outputs;}
 [Serializable] public class Policy {public int version;public string[] featureNames;public Model[] models;public Fixture[] fixtures;}
 Policy policy;
 float nextDecision,nextWarning,stopUntil,lastFireSeen;
 bool wasAI,originalToolEquipped;
 Transform originalToolParent;Vector3 originalToolPosition;Quaternion originalToolRotation;
 void Awake(){
  if(trainedPolicy)policy=JsonUtility.FromJson<Policy>(trainedPolicy.text);
  if(actions&&actions.toolRoot){originalToolParent=actions.toolRoot.parent;originalToolPosition=actions.toolRoot.localPosition;originalToolRotation=actions.toolRoot.localRotation;}
 }
 Model Find(string name){if(policy==null)return null;foreach(var model in policy.models)if(model.name==name)return model;return null;}
 public float[] Predict(string name,float[] inputs){
  var model=Find(name);if(model==null)return null;
  var value=new float[inputs.Length];for(int i=0;i<value.Length;i++)value[i]=(inputs[i]-model.inputMean[i])/model.inputScale[i];
  for(int l=0;l<model.layers.Length;l++){
   var layer=model.layers[l];var output=new float[layer.outputs];
   for(int o=0;o<output.Length;o++){float sum=layer.bias[o];for(int i=0;i<value.Length;i++)sum+=value[i]*layer.weights[o*layer.inputs+i];output[o]=l==model.layers.Length-1?sum:Mathf.Max(0,sum);}
   value=output;
  }
  if(model.motion)for(int i=0;i<value.Length;i++)value[i]=value[i]*model.outputScale[i]+model.outputMean[i];
  else for(int i=0;i<value.Length;i++)value[i]=1/(1+Mathf.Exp(-Mathf.Clamp(value[i],-40,40)));
  return value;
 }
 public float ValidateExport(){
  if(policy==null)policy=JsonUtility.FromJson<Policy>(trainedPolicy.text);
  float maximum=0;foreach(var fixture in policy.fixtures){var outputs=Predict(fixture.name,fixture.features);for(int i=0;i<outputs.Length;i++)maximum=Mathf.Max(maximum,Mathf.Abs(outputs[i]-fixture.outputs[i]));}return maximum;
 }
 public float[] Features(YoloResponse response){
  var result=new float[26];if(response==null||response.width<=0||response.height<=0)return result;
  string[] names={"person","forklift","fire","nohelmet"};
  for(int n=0;n<names.Length;n++){
   YoloDetection best=null;
   foreach(var d in response.detections){
    if(d.class_name!=names[n]||d.xyxy==null||d.xyxy.Length!=4)continue;
    if(n==3&&(d.person_xyxy==null||d.person_xyxy.Length!=4))continue;
    if(best==null||d.confidence>best.confidence)best=d;
   }
   if(best==null)continue;int k=n*6;var p=best.xyxy;
   result[k]=1;result[k+1]=best.confidence;result[k+2]=(p[0]+p[2])*.5f/response.width;result[k+3]=(p[1]+p[3])*.5f/response.height;result[k+4]=(p[2]-p[0])/response.width;result[k+5]=(p[3]-p[1])/response.height;
  }
  result[24]=actions.CollisionRisk?1:0;result[25]=actions.visionCollision?Mathf.Clamp(actions.visionCollision.ClosestImageGap,-1,1):-1;return result;
 }
 void Mode(bool value){
  actions.aiControl=value;wasAI=value;
  if(value){
   originalToolEquipped=actions.ExtinguisherEquipped;
   if(actions.toolRoot)actions.toolRoot.SetParent(patrol&&patrol.Active&&patrol.rightHand?patrol.rightHand:actions.head,true);
   Debug.Log("Learned warehouse AI enabled; experimental policy from VR demonstrations.");
  }else{
   actions.SetAgentSpray(false);actions.SetAgentForkliftStop(false);
   if(actions.toolRoot){actions.toolRoot.SetParent(originalToolParent,false);actions.toolRoot.localPosition=originalToolPosition;actions.toolRoot.localRotation=originalToolRotation;}
   if(actions.extinguisher)actions.extinguisher.SetActive(originalToolEquipped);
   LastDecision="manual";
  }
 }
 void Update(){
  if(!actions||policy==null)return;
  if(runAI!=wasAI)Mode(runAI);
  if(!runAI)return;
  if(!actions.HasFreshPerception){actions.SetAgentSpray(false);LastDecision="waiting for perception";return;}
  var response=actions.bridge.LatestResult;
  if(response==null||(!actions.PersonDetected&&!actions.FireDetected&&!actions.NoHelmetDetected)){actions.SetAgentSpray(false);}
  if(Time.time<nextDecision)return;nextDecision=Time.time+1/Mathf.Max(1,decisionsPerSecond);
  bool closeHelmet=patrol&&patrol.Active&&patrol.ReadyForHelmet&&actions.NoHelmetDetected;
  bool closeFire=patrol&&patrol.Active&&patrol.ReadyForFire&&actions.FireDetected;
  if(!closeHelmet)nearbyHelmetFrames=0;if(!closeFire)nearbyFireFrames=0;
  if(response!=null&&response.frame_id!=nearbyFrame){nearbyFrame=response.frame_id;if(closeHelmet)nearbyHelmetFrames++;if(closeFire)nearbyFireFrames++;}
  bool confirmedHelmet=executeConfirmedNearbyActions&&nearbyHelmetFrames>=3;
  bool confirmedFire=executeConfirmedNearbyActions&&nearbyFireFrames>=3;
  var x=Features(response);string[] modelNames={"helmet","stop","equip","spray"};
  for(int i=0;i<4;i++){var output=Predict(modelNames[i],x);LastProbabilities[i]=output==null?0:output[0];}
  Decisions++;LastDecision="observe";
  if(enableHelmet&&(!patrol||!patrol.Active||patrol.ReadyForHelmet)&&actions.NoHelmetDetected&&(LastProbabilities[0]>=Find("helmet").threshold||confirmedHelmet)&&Time.time>=nextWarning){
   actions.WarnHelmet();WarningsIssued++;nextWarning=Time.time+warningCooldown;LastDecision="helmet warning";
  }
  bool stop=enableForklift&&x[0]>0&&x[6]>0&&LastProbabilities[1]>=Find("stop").threshold;
  if(stop){
   stopUntil=Time.time+minimumStopSeconds;
   if(!actions.ManualForkliftStopped){actions.SetAgentForkliftStop(true);StopCommands++;}
   LastDecision="forklift stopped";
  }else if(releaseStopWhenClear&&actions.ManualForkliftStopped&&Time.time>=stopUntil)actions.SetAgentForkliftStop(false);
  if(enableFire&&actions.FireDetected){
   lastFireSeen=Time.time;
   if((LastProbabilities[2]>=Find("equip").threshold||confirmedFire)&&!actions.ExtinguisherEquipped){actions.SetAgentEquip(true);EquipCommands++;LastDecision="extinguisher equipped";}
   if(actions.ExtinguisherEquipped){
    var pose=Predict("aim",x);var relativePosition=Vector3.ClampMagnitude(new Vector3(pose[0],pose[1],pose[2]),1.2f);
    float tangent=Mathf.Tan(fireFieldOfView*Mathf.Deg2Rad*.5f);
    var ray=new Vector3((2*x[14]-1)*(float)response.width/response.height*tangent,(1-2*x[15])*tangent,1).normalized;
    var direction=(ray+new Vector3(pose[3],pose[4],pose[5])).normalized;
    var position=actions.head.TransformPoint(relativePosition);var rotation=Quaternion.LookRotation(actions.head.TransformDirection(direction),actions.head.up);
    if(patrol&&patrol.Active)patrol.GetToolPose(out position,out rotation);
    actions.SetAgentAim(position,rotation);
   }
   bool spray=(!patrol||!patrol.Active||patrol.ReadyForFire)&&actions.ExtinguisherEquipped&&(LastProbabilities[3]>=Find("spray").threshold||confirmedFire);
   if(spray&&!actions.aiSpray)SprayCommands++;
   actions.SetAgentSpray(spray);if(spray)LastDecision="spraying";
  }else{
   actions.SetAgentSpray(false);
   if(Time.time-lastFireSeen>5&&actions.ExtinguisherEquipped)actions.SetAgentEquip(false);
  }
 }
 void OnDisable(){if(actions&&wasAI)Mode(false);}
}