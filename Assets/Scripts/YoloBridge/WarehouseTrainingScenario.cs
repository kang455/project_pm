using UnityEngine;
public class WarehouseTrainingScenario : MonoBehaviour {
 public WarehouseForkliftSafety trainingForklift;
 public Transform pedestrian;
 public WarehouseRandomFire randomFire;
 public WarehouseSafetyActions actions;
 public Vector3 forkliftStart=new Vector3(31,0,35),pedestrianStart=new Vector3(31,0,42);
 public bool repeatEpisodes;
 public float episodeSeconds=30;
 public int Episode {get;private set;}
 public event System.Action<int> EpisodeReset;
 float nextReset;
 void Start(){nextReset=Time.time+episodeSeconds;}
 void Update(){if(repeatEpisodes&&Time.time>=nextReset)ResetEpisode();}
 [ContextMenu("Reset Training Episode")]
 public void ResetEpisode(){
  if(!Application.isPlaying)return;
  trainingForklift.transform.SetPositionAndRotation(forkliftStart,Quaternion.identity);
  pedestrian.position=pedestrianStart;
  trainingForklift.SetStop(false);actions.SetAgentSpray(false);randomFire.ClearFires();
  Episode++;nextReset=Time.time+Mathf.Max(5,episodeSeconds);EpisodeReset?.Invoke(Episode);
 }
}