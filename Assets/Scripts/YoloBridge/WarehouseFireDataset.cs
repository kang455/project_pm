using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// <summary>Offline synthetic dataset collection. Never used by runtime detection.</summary>
public class WarehouseFireDataset : MonoBehaviour
{
    public Camera sourceCamera;
    public YoloMultiBridge detectionBridge;
    public WarehouseRandomFire randomFire;
    public GameObject firePrefab;
    public Transform[] firePoints;
    public Transform[] heads;
    public string outputDirectory;
    public int groups = 75;
    public int seed = 10526;
    public bool collecting;
    public int savedImages, positiveImages, rejectedPositives;
    public string status = "Idle";
    Camera captureCamera;
    RenderTexture target;
    Texture2D pixels;
    GameObject flame;
    System.Random rng;
    [ContextMenu("Collect fire dataset (Play mode only)")]
    public void Begin()
    {
        if(!Application.isPlaying || !sourceCamera || !firePrefab || firePoints==null || firePoints.Length==0)
        { Debug.LogWarning("Dataset collection needs Play mode and connected references.",this); return; }
        if(!collecting) StartCoroutine(Collect());
    }
    float R(float a,float b) { return a+(float)rng.NextDouble()*(b-a); }
    Color32[] Render(bool showFire)
    {
        if(flame) flame.SetActive(showFire);
        var request=new UniversalRenderPipeline.SingleCameraRequest { destination=target };
        RenderPipeline.SubmitRenderRequest(captureCamera,request);
        var old=RenderTexture.active;RenderTexture.active=target;
        pixels.ReadPixels(new Rect(0,0,target.width,target.height),0,0,false);
        pixels.Apply(false,false);RenderTexture.active=old;
        return pixels.GetPixels32();
    }
    void Save(string split,string id,Color32[] data,string label)
    {
        string dir=Path.Combine(outputDirectory,"images",split);Directory.CreateDirectory(dir);
        string labels=Path.Combine(outputDirectory,"labels",split);Directory.CreateDirectory(labels);
        pixels.SetPixels32(data);pixels.Apply(false,false);
        File.WriteAllBytes(Path.Combine(dir,id+".jpg"),pixels.EncodeToJPG(90));
        File.WriteAllText(Path.Combine(labels,id+".txt"),label);
        savedImages++;
    }
    string Label(Color32[] off,Color32[] on)
    {
        int w=target.width,h=target.height,minX=w,minY=h,maxX=-1,maxY=-1,count=0;
        for(int y=0;y<h;y++)for(int x=0;x<w;x++)
        {
            int i=y*w+x;var a=on[i];var b=off[i];
            int delta=Math.Abs(a.r-b.r)+Math.Abs(a.g-b.g)+Math.Abs(a.b-b.b);
            // Visible fire-only difference; illumination is disabled on this offline instance.
            if(delta>35 && a.r>70 && a.g>55 && a.r>a.b*1.12f && a.g>a.b*1.12f)
            {minX=Math.Min(minX,x);minY=Math.Min(minY,y);maxX=Math.Max(maxX,x);maxY=Math.Max(maxY,y);count++;}
        }
        if(count<35 || maxX-minX<5 || maxY-minY<8) return null;
        minX=Math.Max(0,minX-2);maxX=Math.Min(w-1,maxX+2);
        minY=Math.Max(0,minY-2);maxY=Math.Min(h-1,maxY+2);
        // Texture pixels are bottom-left; JPEG/YOLO coordinates are top-left.
        return string.Format(CultureInfo.InvariantCulture,"0 {0:F6} {1:F6} {2:F6} {3:F6}\n",
            (minX+maxX+1f)/(2*w),1-(minY+maxY+1f)/(2*h),(maxX-minX+1f)/w,(maxY-minY+1f)/h);
    }
    bool Ground(Vector3 position,out Vector3 floor)
    {
        RaycastHit hit;floor=position;
        if(!Physics.Raycast(position+Vector3.up*3,Vector3.down,out hit,6,~0,QueryTriggerInteraction.Ignore) || hit.normal.y<.95f)return false;
        floor=hit.point;
        if(Physics.CheckCapsule(floor+Vector3.up*.5f,floor+Vector3.up*1.5f,.3f,~0,QueryTriggerInteraction.Ignore))return false;
        return true;
    }
    IEnumerator Collect()
    {
        collecting=true;rng=new System.Random(seed);savedImages=positiveImages=rejectedPositives=0;
        bool bridgeEnabled=detectionBridge && detectionBridge.enabled;
        bool fireEnabled=randomFire && randomFire.spawningEnabled;
        if(detectionBridge)detectionBridge.enabled=false;
        if(randomFire){randomFire.spawningEnabled=false;randomFire.ClearFires();}
        Directory.CreateDirectory(outputDirectory);
        target=new RenderTexture(640,360,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);
        target.Create();pixels=new Texture2D(640,360,TextureFormat.RGB24,false);
        captureCamera=new GameObject("Offline Fire Dataset Camera").AddComponent<Camera>();
        captureCamera.CopyFrom(sourceCamera);captureCamera.enabled=false;captureCamera.targetTexture=target;
        captureCamera.aspect=640f/360;captureCamera.fieldOfView=90;captureCamera.cullingMask&=~(1<<5);
        captureCamera.ResetProjectionMatrix();captureCamera.ResetWorldToCameraMatrix();
        var additional=captureCamera.gameObject.AddComponent<UniversalAdditionalCameraData>();
        additional.allowXRRendering=false;additional.requiresDepthTexture=true;additional.renderPostProcessing=false;
        var manifest=new System.Text.StringBuilder("group,split,fire_x,fire_z,camera_x,camera_z\n");
        try
        {
            for(int g=0;g<groups;g++)
            {
                string split=g<50?"train":g<62?"val":"test";
                status="Collecting "+g+"/"+groups;
                Vector3 floor=Vector3.zero,view=Vector3.zero;bool found=false;
                for(int trial=0;trial<60;trial++)
                {
                    var anchor=firePoints[rng.Next(firePoints.Length)].position;
                    if(!Ground(anchor+new Vector3(R(-1.8f,1.8f),0,R(-1.8f,1.8f)),out floor))continue;
                    float angle=R(0,Mathf.PI*2),distance=R(2.5f,8);
                    if(!Ground(floor+new Vector3(Mathf.Sin(angle)*distance,0,Mathf.Cos(angle)*distance),out view))continue;
                    found=true;break;
                }
                if(!found) { status="No safe placement group "+g; continue; }
                captureCamera.transform.position=view+Vector3.up*R(1.35f,1.85f);
                captureCamera.transform.LookAt(floor+Vector3.up*R(.6f,1.1f));
                captureCamera.transform.Rotate(R(-12,12),R(-25,25),0,Space.Self);
                flame=Instantiate(firePrefab,floor+Vector3.up*.03f,Quaternion.identity);
                flame.name="Offline Training Flame";flame.transform.localScale=Vector3.one*R(.8f,1.6f);
                foreach(var l in flame.GetComponentsInChildren<Light>())l.enabled=false;
                foreach(var p in flame.GetComponentsInChildren<ParticleSystem>())
                {var lights=p.lights;lights.enabled=false;p.Play(true);p.Simulate(R(.5f,2),false,true);}
                yield return null;
                for(int phase=0;phase<2;phase++)
                {
                    var off=Render(false);var on=Render(true);string label=Label(off,on);
                    string id="g"+g.ToString("D3")+"_p"+phase;
                    Save(split,id+"_negative",off,"");
                    if(label!=null) {Save(split,id+"_fire",on,label);positiveImages++;}
                    else rejectedPositives++;
                    yield return new WaitForSecondsRealtime(.08f);
                }
                flame.SetActive(false);
                if(heads!=null && heads.Length>0)
                {
                    var head=heads[rng.Next(heads.Length)];
                    Vector3 direction=new Vector3(R(-1,1),0,R(-1,1)).normalized;
                    captureCamera.transform.position=head.position+direction*R(1.2f,3.5f)+Vector3.up*R(-.2f,.2f);
                    captureCamera.transform.LookAt(head.position+Vector3.down*.35f);
                    Save(split,"g"+g.ToString("D3")+"_worker",Render(false),"");
                }
                manifest.AppendFormat(CultureInfo.InvariantCulture,"{0},{1},{2:F3},{3:F3},{4:F3},{5:F3}\n",g,split,floor.x,floor.z,view.x,view.z);
                Destroy(flame);flame=null;
                yield return null;
            }
            File.WriteAllText(Path.Combine(outputDirectory,"groups.csv"),manifest.ToString());
            File.WriteAllText(Path.Combine(outputDirectory,"data.yaml"),
                "path: '"+outputDirectory.Replace("\\","/")+"'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fire\n  1: smoke\n");
            status="Complete: "+savedImages+" images, "+positiveImages+" fire";
        }
        finally
        {
            if(flame)Destroy(flame);
            if(captureCamera)Destroy(captureCamera.gameObject);
            if(target){target.Release();Destroy(target);}
            if(pixels)Destroy(pixels);
            if(detectionBridge)detectionBridge.enabled=bridgeEnabled;
            if(randomFire)randomFire.spawningEnabled=fireEnabled;
            collecting=false;
        }
    }
}
