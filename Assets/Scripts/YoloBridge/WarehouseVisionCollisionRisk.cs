using System.Collections.Generic;
using UnityEngine;
// Image-space warning only: no scene actors, colliders, world positions or clothing state.
public class WarehouseVisionCollisionRisk : MonoBehaviour {
 [Range(.01f,.3f)] public float proximityMargin=.065f;
 public float predictionSeconds=1;
 [Min(1)] public int confirmationFrames=2;
 public bool Risk {get;private set;}
 public float ClosestImageGap {get;private set;}=-1;
 public int VisiblePeople {get;private set;}
 public int VisibleForklifts {get;private set;}
 public string Evidence {get;private set;}
 class Box {public Rect rect;public Vector2 foot;}
 readonly List<Box> previousPeople=new List<Box>(),previousForks=new List<Box>();
 float previousTime;int confirmations;long previousFrame=-1;
 public void Clear(){Risk=false;confirmations=0;ClosestImageGap=-1;previousPeople.Clear();previousForks.Clear();previousFrame=-1;Evidence="no_fresh_visual_evidence";}
 List<Box> Read(YoloResponse response,string name){
  var boxes=new List<Box>();
  if(response.detections==null)return boxes;
  foreach(var detection in response.detections){
   if(detection.class_name!=name||detection.xyxy==null||detection.xyxy.Length!=4)continue;
   var p=detection.xyxy;float x=p[0]/response.width,y=p[1]/response.height,w=(p[2]-p[0])/response.width,h=(p[3]-p[1])/response.height;
   if(w<=0||h<=0)continue;
   boxes.Add(new Box{rect=new Rect(x,y,w,h),foot=new Vector2(x+w*.5f,y+h)});
  }return boxes;
 }
 Vector2 Velocity(Box current,List<Box> previous,float dt){
  Box match=null;float best=.12f;
  foreach(var box in previous){
   float distance=Vector2.Distance(current.rect.center,box.rect.center);
   if(distance<best&&current.rect.width/box.rect.width>.4f&&current.rect.width/box.rect.width<2.5f){best=distance;match=box;}
  }return match==null||dt<.02f||dt>2?Vector2.zero:Vector2.ClampMagnitude((current.foot-match.foot)/dt,.5f);
 }
 public bool Evaluate(YoloResponse response){
  if(response==null||response.width<=0||response.height<=0){Clear();return false;}
  if(response.frame_id<=previousFrame)return Risk;
  var people=Read(response,"person");var forks=Read(response,"forklift");VisiblePeople=people.Count;VisibleForklifts=forks.Count;
  float dt=Time.time-previousTime;bool candidate=false;ClosestImageGap=-1;
  foreach(var person in people)foreach(var fork in forks){
   // A driver visible in the cabin is not pedestrian collision evidence.
   if(fork.rect.Contains(person.rect.center)&&person.foot.y<fork.rect.yMax-fork.rect.height*.18f)continue;
   float horizontal=Mathf.Max(0,Mathf.Abs(person.foot.x-fork.foot.x)-(person.rect.width+fork.rect.width)*.5f);
   float vertical=Mathf.Abs(person.foot.y-fork.foot.y);
   float gap=Mathf.Sqrt(horizontal*horizontal+vertical*vertical);
   if(ClosestImageGap<0||gap<ClosestImageGap)ClosestImageGap=gap;
   Vector2 relativeVelocity=Velocity(person,previousPeople,dt)-Velocity(fork,previousForks,dt);
   Vector2 offset=person.foot-fork.foot;
   float t=relativeVelocity.sqrMagnitude<.0001f?0:Mathf.Clamp(-Vector2.Dot(offset,relativeVelocity)/relativeVelocity.sqrMagnitude,0,predictionSeconds);
   Vector2 predicted=offset+relativeVelocity*t;
   bool near=horizontal<proximityMargin&&vertical<proximityMargin;
   bool approaching=t>0&&predicted.magnitude<offset.magnitude&&Mathf.Abs(predicted.x)<(person.rect.width+fork.rect.width)*.5f+proximityMargin&&Mathf.Abs(predicted.y)<proximityMargin;
   if(near||approaching)candidate=true;
  }
  confirmations=candidate?(dt>2?1:confirmations+1):0;Risk=confirmations>=confirmationFrames;
  Evidence=Risk?"yolo_image_proximity_or_approach":"no_confirmed_image_risk";
  previousPeople.Clear();previousPeople.AddRange(people);previousForks.Clear();previousForks.AddRange(forks);previousTime=Time.time;previousFrame=response.frame_id;
  return Risk;
 }
}