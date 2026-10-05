using UnityEngine;
using UnityEngine.Playables;
public class WarehouseForkliftSafety : MonoBehaviour {
 public PlayableDirector motionDirector;
 public Transform[] pedestrians;
 public Transform[] waypoints;
 public float speed=1.5f, lookAheadSeconds=3, corridorRadius=1.3f;
 public bool waypointMode;
 public bool automaticSafetyStop=true;
 public bool Danger {get;private set;}
 public bool Stopped {get;private set;}
 public float ClosestDistance {get;private set;}
 public Vector3 Velocity {get;private set;}
 public event System.Action<bool> DangerChanged;
 int waypoint; Vector3 previous; bool commandedStop;
 void OnEnable(){previous=transform.position;}
 public void SetStop(bool stop){commandedStop=stop;}
 void Update(){
  var delta=transform.position-previous; previous=transform.position;
  Velocity=delta/Mathf.Max(.001f,Time.deltaTime);
  Vector3 direction=waypointMode&&waypoints!=null&&waypoints.Length>0?(waypoints[waypoint].position-transform.position):Velocity;
  direction.y=0;
  if(direction.sqrMagnitude<.01f)direction=transform.forward;
  direction.Normalize(); ClosestDistance=float.PositiveInfinity; bool danger=false;
  foreach(var person in pedestrians){
   if(!person||!person.gameObject.activeInHierarchy)continue;
   var offset=person.position-transform.position;offset.y=0;
   float along=Vector3.Dot(offset,direction),side=(offset-direction*along).magnitude;
   ClosestDistance=Mathf.Min(ClosestDistance,offset.magnitude);
   if(along>-.5f&&along<Mathf.Max(3,speed*lookAheadSeconds+1.5f)&&side<corridorRadius)danger=true;
  }
  if(danger!=Danger){Danger=danger;DangerChanged?.Invoke(danger);}
  Stopped=(automaticSafetyStop&&Danger)||commandedStop;
  if(motionDirector){
   if(Stopped&&motionDirector.state==PlayState.Playing)motionDirector.Pause();
   else if(!Stopped&&motionDirector.state!=PlayState.Playing)motionDirector.Resume();
  }
  if(waypointMode&&!Stopped&&waypoints!=null&&waypoints.Length>0){
   var target=waypoints[waypoint].position;target.y=transform.position.y;
   transform.position=Vector3.MoveTowards(transform.position,target,speed*Time.deltaTime);
   if(direction.sqrMagnitude>.01f)transform.rotation=Quaternion.RotateTowards(transform.rotation,Quaternion.LookRotation(direction),90*Time.deltaTime);
   if(Vector3.Distance(transform.position,target)<.1f)waypoint=(waypoint+1)%waypoints.Length;
  }
 }
}