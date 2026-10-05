using System;
using System.IO;
using System.Collections;
using UnityEngine;
// Explicit scripted teacher demonstrations, never mislabeled as human VR input.
public class WarehouseTeacherCollection : MonoBehaviour {
 public WarehouseSafetyActions actions;public WarehouseLearnedAgent agent;
 public WarehouseAutonomousPatrol patrol;public WarehouseRandomFire fires;
 [TextArea] public string planJson;public bool runOnPlay;
 public int repetitions=5,framesPerSession=18;
 public int CompletedSessions {get;private set;} public int PositiveHelmet,PositiveStop,PositiveFire;
 public string Status {get;private set;}="idle";
 [Serializable] public class Case {public string topic;public Vector3 position;public Quaternion rotation;}
 [Serializable] class Plan {public Case[] cases;}
 [Serializable] class Sample {
  public int schema_version=2;public string control_mode="scripted_teacher",data_source="live_unity_yolo_scripted_teacher";
  public string topic;public long frame_id;public float time;public float[] observations;
  public Vector3 head_position,right_position;public Quaternion head_rotation,right_rotation;
  public bool spray,collision_stop,helmet_warning,extinguisher_equipped,collision_risk,right_controller_active=true;
  public YoloResponse perception;
 }
 IEnumerator Start(){if(!runOnPlay)yield break;
  var plan=JsonUtility.FromJson<Plan>(planJson);
  agent.runAI=false;patrol.automaticPatrol=false;yield return null;
  agent.enabled=false;patrol.enabled=false;
  var human=actions.head;var source=actions.bridge.xrCamera;
  var teacher=new GameObject("Temporary Teacher Camera").AddComponent<Camera>();
  teacher.CopyFrom(patrol.aiCamera);teacher.enabled=false;teacher.targetTexture=null;
  actions.head=teacher.transform;actions.aiControl=true;actions.toolRoot.SetParent(teacher.transform,true);
  actions.bridge.matchXrFieldOfView=false;actions.bridge.SetSourceCamera(teacher);
  fires.spawningEnabled=false;
  foreach(var f in FindObjectsByType<WarehouseExtinguishableFire>(FindObjectsSortMode.None))Destroy(f.gameObject);
  var directory=Path.Combine(Application.persistentDataPath,"SafetyDemonstrations","AutomaticTeacher",DateTime.Now.ToString("yyyyMMdd_HHmmss"));
  Directory.CreateDirectory(directory);
  for(int topic=0;topic<3;topic++)for(int trial=0;trial<repetitions;trial++){
   string name=topic==0?"helmet":topic==1?"forklift":"fire";
   var candidates=Array.FindAll(plan.cases,c=>c.topic==name);
   var view=candidates[trial%candidates.Length];
   teacher.transform.SetPositionAndRotation(view.position+new Vector3((trial%3-1)*.15f,0,0),view.rotation);
   if(topic==2)teacher.transform.SetPositionAndRotation(new Vector3(31+(trial%3-1)*.35f,1.5f,49),Quaternion.Euler(0,180,0));
   actions.ResetPerception();actions.SetAgentEquip(false);actions.SetAgentSpray(false);actions.SetAgentForkliftStop(false);
   GameObject fireObject=null;long last=-1;int frames=0;
   Status=name+" "+(trial+1);
   using(var writer=new StreamWriter(Path.Combine(directory,name+"_"+trial+".jsonl"))){
    float deadline=Time.realtimeSinceStartup+45;
    while(frames<framesPerSession&&Time.realtimeSinceStartup<deadline){
     yield return null;var response=actions.bridge.LatestResult;
     if(response==null||response.frame_id==last||!actions.HasFreshPerception)continue;
     last=response.frame_id;
     if(topic==2&&frames==6){
      fireObject=Instantiate(fires.firePrefab,new Vector3(31+(trial%3-1)*.4f,.03f,45-(trial%2)*.5f),Quaternion.identity);
      fireObject.AddComponent<WarehouseExtinguishableFire>();yield return new WaitForSeconds(.5f);
     }
     bool helmet=topic==0&&actions.NoHelmetDetected;
     bool stop=topic==1&&actions.PersonDetected&&actions.CollisionRisk;
     bool fire=topic==2&&actions.FireDetected&&frames>6;
     if(helmet){actions.WarnHelmet();PositiveHelmet++;}
     actions.SetAgentForkliftStop(stop);
     if(stop)PositiveStop++;
     actions.SetAgentEquip(fire);
     Vector3 hand=teacher.transform.TransformPoint(new Vector3(.25f,-.35f,.35f));
     Quaternion aim=teacher.transform.rotation;
     if(fireObject)aim=Quaternion.LookRotation(fireObject.transform.position+Vector3.up*.6f-hand);
     // Known target is allowed only for teaching the desired hand pose, not detector input.
     actions.SetAgentAim(hand,aim);actions.SetAgentSpray(fire);
     if(fire)PositiveFire++;
     var sample=new Sample{topic=name,frame_id=last,time=Time.time,observations=actions.GetTrainingObservations(),head_position=teacher.transform.position,head_rotation=teacher.transform.rotation,right_position=hand,right_rotation=aim,spray=fire,collision_stop=stop,helmet_warning=helmet,extinguisher_equipped=fire,collision_risk=actions.CollisionRisk,perception=response};
     writer.WriteLine(JsonUtility.ToJson(sample));frames++;
    }
   }
   if(fireObject)Destroy(fireObject);
   actions.SetAgentSpray(false);CompletedSessions++;
  }
  Status="complete";actions.SetAgentSpray(false);actions.SetAgentEquip(false);actions.SetAgentForkliftStop(false);
  actions.head=human;actions.bridge.SetSourceCamera(source);actions.aiControl=false;
  actions.toolRoot.SetParent(actions.rightController,true);
  Destroy(teacher.gameObject);agent.enabled=true;patrol.enabled=true;patrol.automaticPatrol=true;agent.runAI=true;fires.spawningEnabled=true;
  Debug.Log("Automatic teacher collection complete: "+directory);
 }
}