using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

[Serializable]
public class WarehouseVehicleLabelGroup
{
    public string name;
    public Renderer[] renderers;
}

// Offline labels come from renderer visibility. Runtime detection never reads these labels.
public class WarehouseSafetyDataset : MonoBehaviour
{
    public Camera sourceCamera;
    public YoloMultiBridge detectionBridge;
    public WarehouseRandomFire randomFire;
    public SkinnedMeshRenderer[] heads;
    public Renderer[] forkliftRenderers;
    public WarehouseVehicleLabelGroup[] forkliftGroups;
    public Renderer helmetTemplate;
    public string outputDirectory;
    public int viewsPerWorker = 12, forkliftViews = 90, seed = 10527;
    public float captureFieldOfView = 70;
    public bool includeForkliftNegatives;
    public bool collecting;
    public int savedImages;
    public string status = "Idle";
    Camera cameraRig;
    RenderTexture target;
    Texture2D texture;
    System.Random rng;
    readonly List<GameObject> temporaryHelmets = new List<GameObject>();
    Renderer[] helmets;
    bool[] originalHelmetState;
    float R(float a,float b) { return a+(float)rng.NextDouble()*(b-a); }
    [ContextMenu("Collect safety datasets (Play mode only)")]
    public void Begin()
    {
        if(!Application.isPlaying || !sourceCamera || !helmetTemplate || heads==null || heads.Length<4)
        {Debug.LogWarning("Safety collection requires connected references and Play mode.",this);return;}
        if(!collecting)StartCoroutine(Collect());
    }
    Color32[] Render()
    {
        RenderPipeline.SubmitRenderRequest(cameraRig,new UniversalRenderPipeline.SingleCameraRequest {destination=target});
        var old=RenderTexture.active;RenderTexture.active=target;
        texture.ReadPixels(new Rect(0,0,target.width,target.height),0,0,false);texture.Apply(false,false);
        RenderTexture.active=old;return texture.GetPixels32();
    }
    string VisibleLabel(Color32[] image,Renderer[] parts,int classId)
    {
        parts=parts.Where(x=>x && x.enabled && x.gameObject.activeInHierarchy).ToArray();
        if(parts.Length==0)return "";
        var bounds=parts[0].bounds;foreach(var part in parts)bounds.Encapsulate(part.bounds);
        var planes=GeometryUtility.CalculateFrustumPlanes(cameraRig);
        if(!GeometryUtility.TestPlanesAABB(planes,bounds))return "";
        int w=target.width,h=target.height;
        float left=w,right=0,bottom=h,top=0;
        for(int i=0;i<8;i++)
        {
            Vector3 c=bounds.center+Vector3.Scale(bounds.extents,new Vector3((i&1)==0?-1:1,(i&2)==0?-1:1,(i&4)==0?-1:1));
            var p=cameraRig.WorldToViewportPoint(c);if(p.z<=cameraRig.nearClipPlane)return "";
            left=Mathf.Min(left,p.x*w);right=Mathf.Max(right,p.x*w);bottom=Mathf.Min(bottom,p.y*h);top=Mathf.Max(top,p.y*h);
        }
        int l=Mathf.Clamp(Mathf.FloorToInt(left),0,w-1),r=Mathf.Clamp(Mathf.CeilToInt(right),0,w-1);
        int b=Mathf.Clamp(Mathf.FloorToInt(bottom),0,h-1),t=Mathf.Clamp(Mathf.CeilToInt(top),0,h-1);
        if(r-l<5 || t-b<5)return "";
        Color32[] without;
        foreach(var part in parts)part.enabled=false;
        try {without=Render();}finally{foreach(var part in parts)part.enabled=true;}
        int minX=w,minY=h,maxX=-1,maxY=-1,count=0;
        for(int y=b;y<=t;y++)for(int x=l;x<=r;x++)
        {
            int i=y*w+x;var a=image[i];var z=without[i];
            if(Math.Abs(a.r-z.r)+Math.Abs(a.g-z.g)+Math.Abs(a.b-z.b)<45)continue;
            minX=Math.Min(minX,x);maxX=Math.Max(maxX,x);minY=Math.Min(minY,y);maxY=Math.Max(maxY,y);count++;
        }
        if(count<15 || maxX-minX<5 || maxY-minY<5)return "";
        return string.Format(CultureInfo.InvariantCulture,"{0} {1:F6} {2:F6} {3:F6} {4:F6}\n",classId,
            (minX+maxX+1f)/(2*w),1-(minY+maxY+1f)/(2*h),(maxX-minX+1f)/w,(maxY-minY+1f)/h);
    }
    void Save(string model,string split,string id,byte[] jpeg,string labels)
    {
        string root=Path.Combine(outputDirectory,model);
        Directory.CreateDirectory(Path.Combine(root,"images",split));Directory.CreateDirectory(Path.Combine(root,"labels",split));
        File.WriteAllBytes(Path.Combine(root,"images",split,id+".jpg"),jpeg);
        File.WriteAllText(Path.Combine(root,"labels",split,id+".txt"),labels);savedImages++;
    }
    void Capture(string split,string id)
    {
        var image=Render();var helmetLabels=new StringBuilder();var bareLabels=new StringBuilder();var vehicleLabels=new StringBuilder();
        for(int i=0;i<heads.Length;i++)
        {
            if(!heads[i] || !heads[i].enabled || !heads[i].gameObject.activeInHierarchy)continue;
            bool equipped=helmets[i] && helmets[i].enabled && helmets[i].gameObject.activeInHierarchy;
            if(equipped)helmetLabels.Append(VisibleLabel(image,new Renderer[]{helmets[i]},1));
            else bareLabels.Append(VisibleLabel(image,new Renderer[]{heads[i]},0));
            var personParts=heads[i].transform.parent.GetComponentsInChildren<Renderer>().Where(x=>x.name!="Helmet").ToArray();
            vehicleLabels.Append(VisibleLabel(image,personParts,3));
        }
        foreach(var vehicle in Vehicles())vehicleLabels.Append(VisibleLabel(image,vehicle,2));
        texture.SetPixels32(image);texture.Apply(false,false);var jpeg=texture.EncodeToJPG(90);
        Save("helmet",split,id,jpeg,helmetLabels.ToString());Save("nohelmet",split,id,jpeg,bareLabels.ToString());
        Save("forklift",split,id,jpeg,vehicleLabels.ToString());
    }
    Renderer[][] Vehicles()
    {
        if(forkliftGroups!=null && forkliftGroups.Length>0)return forkliftGroups.Where(g=>g!=null && g.renderers!=null && g.renderers.Length>0).Select(g=>g.renderers).ToArray();
        return new[]{forkliftRenderers};
    }
    void CaptureWithoutForklift(string split,string id)
    {
        var parts=FindObjectsByType<Renderer>(FindObjectsSortMode.None).Where(r=>r.enabled && r.transform.GetComponentsInParent<Transform>().Any(p=>p.name=="Forklift" || p.name=="Reachlift")).ToArray();
        foreach(var part in parts)part.enabled=false;
        try {Capture(split,id+"_empty");}finally{foreach(var part in parts)part.enabled=true;}
    }
    IEnumerator Collect()
    {
        collecting=true;savedImages=0;rng=new System.Random(seed);
        bool bridgeOn=detectionBridge && detectionBridge.enabled,fireOn=randomFire && randomFire.spawningEnabled;
        if(detectionBridge)detectionBridge.enabled=false;
        if(randomFire){randomFire.spawningEnabled=false;randomFire.ClearFires();}
        target=new RenderTexture(640,360,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);target.Create();
        texture=new Texture2D(640,360,TextureFormat.RGB24,false);
        cameraRig=new GameObject("Offline Safety Dataset Camera").AddComponent<Camera>();cameraRig.CopyFrom(sourceCamera);
        cameraRig.enabled=false;cameraRig.targetTexture=target;cameraRig.aspect=640f/360;cameraRig.fieldOfView=captureFieldOfView;
        cameraRig.cullingMask&=~(1<<5);cameraRig.ResetProjectionMatrix();cameraRig.ResetWorldToCameraMatrix();
        var urp=cameraRig.gameObject.AddComponent<UniversalAdditionalCameraData>();urp.allowXRRendering=false;urp.requiresDepthTexture=true;
        helmets=new Renderer[heads.Length];originalHelmetState=new bool[heads.Length];
        try
        {
            for(int i=0;i<heads.Length;i++)
            {
                var bone=heads[i].bones.FirstOrDefault(x=>x && x.name=="Bip001 Head");
                if(!bone)continue;
                helmets[i]=bone.GetComponentsInChildren<Renderer>().FirstOrDefault(x=>x.name=="Yellow Scooter Helmet");
                if(helmets[i])originalHelmetState[i]=helmets[i].enabled;
                else
                {
                    var copy=Instantiate(helmetTemplate.gameObject,bone,false);
                    copy.transform.localPosition=helmetTemplate.transform.localPosition;
                    copy.transform.localRotation=helmetTemplate.transform.localRotation;
                    copy.transform.localScale=helmetTemplate.transform.localScale;
                    temporaryHelmets.Add(copy);helmets[i]=copy.GetComponent<Renderer>();helmets[i].enabled=false;
                }
            }
            int trainCount=Math.Max(2,heads.Length-4);
            for(int i=0;i<heads.Length;i++)for(int v=0;v<viewsPerWorker;v++)
            {
                string split=i<trainCount?"train":i<trainCount+2?"val":"test";
                status="Worker "+i+" view "+v+"/"+viewsPerWorker;
                Vector3 center=heads[i].bounds.center;
                float angle=R(0,Mathf.PI*2),distance=R(1.1f,3.6f);
                cameraRig.transform.position=center+new Vector3(Mathf.Sin(angle)*distance,R(-.3f,.25f),Mathf.Cos(angle)*distance);
                cameraRig.transform.LookAt(center+Vector3.down*R(.05f,.4f));cameraRig.transform.Rotate(R(-5,5),R(-12,12),0,Space.Self);
                if(helmets[i])helmets[i].enabled=true;
                Capture(split,"w"+i.ToString("D2")+"_v"+v.ToString("D2")+"_helmet");
                if(helmets[i])helmets[i].enabled=false;
                Capture(split,"w"+i.ToString("D2")+"_v"+v.ToString("D2")+"_bare");
                if(helmets[i])helmets[i].enabled=originalHelmetState[i];
                yield return null;
            }
            for(int v=0;v<forkliftViews;v++)
            {
                var vehicle=Vehicles()[v%Vehicles().Length];
                var vehicleBounds=vehicle[0].bounds;
                foreach(var part in vehicle)vehicleBounds.Encapsulate(part.bounds);
                string split=v<forkliftViews*.7f?"train":v<forkliftViews*.85f?"val":"test";
                status="Forklift view "+v+"/"+forkliftViews;
                float angle=R(0,Mathf.PI*2),distance=R(3.5f,8.5f);
                cameraRig.transform.position=vehicleBounds.center+new Vector3(Mathf.Sin(angle)*distance,R(-.1f,1),Mathf.Cos(angle)*distance);
                cameraRig.transform.LookAt(vehicleBounds.center);cameraRig.transform.Rotate(R(-5,5),R(-15,15),0,Space.Self);
                Capture(split,"f"+v.ToString("D3"));
                if(includeForkliftNegatives)CaptureWithoutForklift(split,"f"+v.ToString("D3"));
                yield return null;
            }
            string[] modelNames={"helmet","nohelmet","forklift"};
            string[] yamlNames={"  0: object\n  1: safety_helmet_detection_0321 - v3 2024-03-27 8-32pm\n","  0: nohelmet\n","  0: container\n  1: crane\n  2: forklift\n  3: person\n  4: ship\n  5: stacker\n  6: truck\n"};
            for(int i=0;i<modelNames.Length;i++)
            {
                string dir=Path.Combine(outputDirectory,modelNames[i]);
                File.WriteAllText(Path.Combine(dir,"data.yaml"),"path: '"+dir.Replace("\\","/")+"'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"+yamlNames[i]);
            }
            status="Complete: "+savedImages+" labeled images";
        }
        finally
        {
            if(helmets!=null)for(int i=0;i<helmets.Length;i++)if(helmets[i])helmets[i].enabled=originalHelmetState[i];
            foreach(var copy in temporaryHelmets)if(copy)Destroy(copy);temporaryHelmets.Clear();
            if(cameraRig)Destroy(cameraRig.gameObject);if(target){target.Release();Destroy(target);}if(texture)Destroy(texture);
            if(detectionBridge)detectionBridge.enabled=bridgeOn;if(randomFire)randomFire.spawningEnabled=fireOn;collecting=false;
        }
    }
}
