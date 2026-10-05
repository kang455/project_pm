using UnityEngine;
public class WarehouseExtinguishableFire : MonoBehaviour {
 public float requiredSpraySeconds=2;
 public float remaining=1;
 public static int TotalExtinguished;
 public bool Extinguished {get;private set;}
 public event System.Action ExtinguishedEvent;
 public void Suppress(float seconds) {
  if(Extinguished||seconds<=0)return;
  remaining=Mathf.Max(0,remaining-seconds/Mathf.Max(.1f,requiredSpraySeconds));
  if(remaining>0)return;
  Extinguished=true; TotalExtinguished++; ExtinguishedEvent?.Invoke();
  foreach(var p in GetComponentsInChildren<ParticleSystem>())p.Stop(true,ParticleSystemStopBehavior.StopEmittingAndClear);
  Destroy(gameObject);
 }
}