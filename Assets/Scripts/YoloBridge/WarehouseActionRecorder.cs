using System;
using System.IO;
using UnityEngine;
public class WarehouseActionRecorder : MonoBehaviour {
 public WarehouseSafetyActions actions;
 public bool recordDemonstrations=true;
 [Range(1,30)] public float samplesPerSecond=10;
 public string OutputPath {get;private set;}
 StreamWriter writer;float next;
 [Serializable] class Sample {
  public int schema_version=2;
  public string collision_risk_source="yolo_image_boxes";
  public string[] observation_names={"person_detected","fire_detected","nohelmet_detected","image_collision_risk","spraying","normalized_image_foot_gap","extinguished_count"};
  public float time;public float[] observations;public Vector3 head_position,right_position;
  public Quaternion head_rotation,right_rotation;
  public bool spray,collision_stop,helmet_warning,extinguisher_equipped,collision_risk;
  public string control_mode;
  public bool right_controller_active;
  public int extinguished;public YoloResponse perception;
 }
 [Serializable] class Command {public string kind="human_action";public float time;public string action;}
 void OnHumanAction(string action){if(writer!=null)writer.WriteLine(JsonUtility.ToJson(new Command{time=Time.time,action=action}));}
 void OnAgentResult(string action){if(writer!=null&&actions.aiControl&&(action.StartsWith("agent_")||action=="fire_extinguished"))writer.WriteLine(JsonUtility.ToJson(new Command{kind="agent_action",time=Time.time,action=action}));}
 void OnEnable(){
  if(!recordDemonstrations||!actions)return;
  try{
   var directory=Path.Combine(Application.persistentDataPath,"SafetyDemonstrations");Directory.CreateDirectory(directory);
   OutputPath=Path.Combine(directory,DateTime.Now.ToString("yyyyMMdd_HHmmss_fff")+".jsonl");
   writer=new StreamWriter(OutputPath);actions.HumanAction+=OnHumanAction;actions.ActionResult+=OnAgentResult;Debug.Log("Safety action recording: "+OutputPath);
  }catch(Exception error){Debug.LogWarning("Safety recording unavailable: "+error.Message);}
 }
 void Update(){
  if(writer==null||Time.time<next)return;next=Time.time+1/Mathf.Max(1,samplesPerSecond);
  var sample=new Sample{time=Time.time,observations=actions.GetTrainingObservations(),head_position=actions.head.position,head_rotation=actions.head.rotation,right_position=actions.rightController.position,right_rotation=actions.rightController.rotation,spray=actions.Spraying,collision_stop=actions.ManualForkliftStopped,helmet_warning=actions.HelmetWarningActive,extinguisher_equipped=actions.ExtinguisherEquipped,collision_risk=actions.CollisionRisk,control_mode=actions.aiControl?"agent":"human",right_controller_active=actions.rightController.gameObject.activeInHierarchy,extinguished=actions.ExtinguishedCount,perception=actions.bridge.LatestResult};
  writer.WriteLine(JsonUtility.ToJson(sample));
 }
 void OnDisable(){if(actions){actions.HumanAction-=OnHumanAction;actions.ActionResult-=OnAgentResult;}writer?.Dispose();writer=null;}
}