using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;
public static class WarehouseSafetyInputVerification {
 static InputDevice right,left;
 public static string Begin(){
  InputSystem.RegisterLayout("{\"name\":\"SafetyVerificationController\",\"extend\":\"XRController\",\"controls\":[{\"name\":\"primaryButton\",\"layout\":\"Button\",\"offset\":0,\"bit\":0},{\"name\":\"secondaryButton\",\"layout\":\"Button\",\"offset\":0,\"bit\":1},{\"name\":\"trigger\",\"layout\":\"Axis\",\"offset\":4,\"format\":\"FLT\"}]}");
  right=InputSystem.AddDevice("SafetyVerificationController");left=InputSystem.AddDevice("SafetyVerificationController");
  InputSystem.SetDeviceUsage(right,CommonUsages.RightHand);InputSystem.SetDeviceUsage(left,CommonUsages.LeftHand);
  return "Synthetic controllers only; not Quest hardware validation";
 }
 public static void Press(string button,float value){
  var device=button=="X"?left:right;string control=button=="B"?"secondaryButton":button=="Trigger"?"trigger":"primaryButton";
  using(StateEvent.From(device,out var eventPtr)){((AxisControl)device[control]).WriteValueIntoEvent(value,eventPtr);InputSystem.QueueEvent(eventPtr);}InputSystem.Update();
 }
 public static void End(){if(right!=null)InputSystem.RemoveDevice(right);if(left!=null)InputSystem.RemoveDevice(left);right=left=null;InputSystem.RemoveLayout("SafetyVerificationController");}
}