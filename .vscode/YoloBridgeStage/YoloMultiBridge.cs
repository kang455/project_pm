using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.UI;

[Serializable] public class YoloModelSetting
{
    public string model_id;
    public bool enabled = true;
    [Range(0, 1)] public float confidence = 0.25f;
    public Color color = Color.green;
}
[Serializable] public class YoloKeypoint { public float x, y, confidence; }
[Serializable] public class YoloDetection
{
    public string model_id, task, class_name;
    public int class_id;
    public float confidence;
    public float[] xyxy;
    public YoloKeypoint[] keypoints;
}
[Serializable] public class YoloClassification
{
    public string model_id, task, class_name;
    public int class_id;
    public float confidence;
}
[Serializable] public class YoloModelTiming
{
    public string model_id, task;
    public float processing_ms, inference_ms;
    public int count;
}
[Serializable] public class YoloResponse
{
    public long frame_id;
    public int width, height;
    public float processing_ms, inference_ms;
    public string coordinate_origin;
    public YoloDetection[] detections;
    public YoloDetection[] raw_detections; // Diagnostic candidates, not approved observations.
    public YoloClassification[] classifications;
    public YoloModelTiming[] model_results;
}

/// <summary>One in-flight JPEG, one matching preview/result. Accessible to ML-Agents.</summary>
public class YoloMultiBridge : MonoBehaviour
{
    public Camera xrCamera;
    public Camera detectionCamera;
    public Canvas previewCanvas;
    public Text resultText;
    public string serverUrl = "http://127.0.0.1:8765/predict";
    public int imageWidth = 640, imageHeight = 360;
    [Range(1, 100)] public int jpegQuality = 80;
    [Range(0.1f, 5)] public float maximumRequestsPerSecond = 5;
    public bool useAsyncGPUReadback = true;
    public bool flipAsyncReadbackY = false; // Verified upright on this project's D3D11 URP capture.
    [UnityEngine.Serialization.FormerlySerializedAs("showPreview")]
    public bool showDetections = true;
    public int requestTimeoutSeconds = 30;
    public YoloModelSetting[] models;

    public YoloResponse LatestResult { get; private set; }
    public event Action<YoloResponse> ResultReceived;
    public bool RequestInFlight { get; private set; }
    public int SuccessfulResponses { get; private set; }
    public int FailedRequests { get; private set; }
    public string ConnectionStatus { get; private set; } = "Waiting for capture";
    public float LastRoundTripMs { get; private set; }
    // Useful for a reproducible single-frame test, diagnostics, or an observation recorder.
    public byte[] LastSubmittedJpeg { get; private set; }
    public long LastSubmittedFrameId { get; private set; }

    RenderTexture captureTarget;
    Texture2D capturePixels;
    UnityWebRequest activeRequest;
    bool warned;
    long nextFrameId;

    void Start()
    {
        if (!xrCamera || !detectionCamera || !previewCanvas || !resultText)
        { Failure("Missing camera or preview references"); enabled = false; return; }
        imageWidth = Mathf.Clamp(imageWidth, 64, 1920);
        imageHeight = Mathf.Clamp(imageHeight, 64, 1080);
        captureTarget = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32,
                                         RenderTextureReadWrite.sRGB);
        captureTarget.name = "YOLO monocular capture";
        captureTarget.antiAliasing = 1;
        captureTarget.Create();
        capturePixels = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false, false);
        ConfigureCamera();
        resultText.text = "";
        previewCanvas.enabled = false;
        Debug.Log("YOLO bridge started: monocular 640x360, one in-flight request.", this);
        StartCoroutine(CaptureLoop());
    }

    void Update()
    {
        if (previewCanvas) previewCanvas.enabled = showDetections && !warned && resultText && !string.IsNullOrEmpty(resultText.text);
    }

    void OnEnable()
    {
        if (captureTarget) StartCoroutine(CaptureLoop());
    }

    void ConfigureCamera()
    {
        detectionCamera.CopyFrom(xrCamera);
        detectionCamera.enabled = false;
        if (GraphicsSettings.currentRenderPipeline == null)
            detectionCamera.stereoTargetEye = StereoTargetEyeMask.None;
        detectionCamera.targetTexture = captureTarget;
        detectionCamera.aspect = (float)imageWidth / imageHeight;
        detectionCamera.cullingMask = xrCamera.cullingMask & ~(1 << 5); // UI layer excluded.
        detectionCamera.ResetProjectionMatrix();
        detectionCamera.ResetWorldToCameraMatrix();
        var data = detectionCamera.GetComponent<UniversalAdditionalCameraData>();
        if (!data) data = detectionCamera.gameObject.AddComponent<UniversalAdditionalCameraData>();
        data.renderType = CameraRenderType.Base;
        data.allowXRRendering = false;
        data.renderPostProcessing = false;
        data.cameraStack.Clear();
    }

    bool RenderCapture()
    {
        try
        {
            ConfigureCamera();
            detectionCamera.transform.SetPositionAndRotation(xrCamera.transform.position, xrCamera.transform.rotation);
            if (GraphicsSettings.currentRenderPipeline == null)
                detectionCamera.Render();
            else
            {
                var request = new UniversalRenderPipeline.SingleCameraRequest { destination = captureTarget };
                if (!RenderPipeline.SupportsRenderRequest(detectionCamera, request))
                    throw new InvalidOperationException("Current pipeline does not support SingleCameraRequest");
                RenderPipeline.SubmitRenderRequest(detectionCamera, request);
            }
            return true;
        }
        catch (Exception ex) { Failure("Capture failed: " + ex.Message); return false; }
    }

    IEnumerator CaptureLoop()
    {
        while (enabled)
        {
            float started = Time.realtimeSinceStartup;
            yield return new WaitForEndOfFrame();
            if (!RenderCapture()) { yield return new WaitForSecondsRealtime(2); continue; }
            if (useAsyncGPUReadback && SystemInfo.supportsAsyncGPUReadback)
            {
                var readback = AsyncGPUReadback.Request(captureTarget, 0, TextureFormat.RGBA32);
                while (!readback.done) yield return null;
                if (readback.hasError)
                {
                    // Keep VR running, and use the supported synchronous fallback next time.
                    useAsyncGPUReadback = false;
                    Failure("GPU readback failed; switching to ReadPixels");
                    continue;
                }
                byte[] pixels = readback.GetData<byte>().ToArray();
                if (flipAsyncReadbackY) FlipRows(pixels, imageWidth * 4, imageHeight);
                capturePixels.LoadRawTextureData(pixels);
                capturePixels.Apply(false, false);
            }
            else
            {
                var previous = RenderTexture.active;
                RenderTexture.active = captureTarget;
                capturePixels.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0, false);
                capturePixels.Apply(false, false);
                RenderTexture.active = previous;
            }
            byte[] jpeg = capturePixels.EncodeToJPG(jpegQuality);
            long frameId = ++nextFrameId;
            LastSubmittedJpeg = jpeg;
            LastSubmittedFrameId = frameId;
            using (var request = new UnityWebRequest(serverUrl, "POST"))
            {
                activeRequest = request;
                request.uploadHandler = new UploadHandlerRaw(jpeg);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.timeout = Mathf.Max(1, requestTimeoutSeconds);
                request.SetRequestHeader("Content-Type", "image/jpeg");
                request.SetRequestHeader("X-Frame-Id", frameId.ToString(CultureInfo.InvariantCulture));
                request.SetRequestHeader("X-Model-Settings", ModelSettingsJson());
                RequestInFlight = true;
                float sentAt = Time.realtimeSinceStartup;
                yield return request.SendWebRequest(); // Does not block the main thread.
                LastRoundTripMs = (Time.realtimeSinceStartup - sentAt) * 1000;
                RequestInFlight = false;
                activeRequest = null;
                if (request.result != UnityWebRequest.Result.Success)
                {
                    FailedRequests++;
                    Failure("Server request failed: " + request.error);
                }
                else
                {
                    try
                    {
                        var result = JsonUtility.FromJson<YoloResponse>(request.downloadHandler.text);
                        if (result == null || result.frame_id != frameId || result.width != imageWidth ||
                            result.height != imageHeight || result.coordinate_origin != "top_left" || result.detections == null)
                            throw new InvalidOperationException("Frame ID, image size or response schema mismatch");
                        if (warned) { Debug.Log("YOLO server connection restored.", this); warned = false; }
                        LatestResult = result;
                        SuccessfulResponses++;
                        ConnectionStatus = "Connected";
                        DrawResult(result);
                        ResultReceived?.Invoke(result);
                    }
                    catch (Exception ex) { FailedRequests++; Failure("Response failed: " + ex.Message); }
                }
            }
            float period = Mathf.Max(1f / Mathf.Clamp(maximumRequestsPerSecond, 0.1f, 5),
                                      LatestResult == null ? 0 : LatestResult.processing_ms / 1000f * 1.1f);
            if (warned) period = Mathf.Max(period, 2);
            float remaining = period - (Time.realtimeSinceStartup - started);
            if (remaining > 0) yield return new WaitForSecondsRealtime(remaining);
        }
    }

    string ModelSettingsJson()
    {
        var sb = new StringBuilder("{\"models\":[");
        if (models != null)
            for (int i = 0; i < models.Length; i++)
            {
                if (i > 0) sb.Append(',');
                var m = models[i];
                sb.Append("{\"model_id\":\"").Append(m.model_id).Append("\",\"enabled\":")
                  .Append(m.enabled ? "true" : "false").Append(",\"confidence\":")
                  .Append(Mathf.Clamp01(m.confidence).ToString(CultureInfo.InvariantCulture)).Append('}');
            }
        return sb.Append("]}").ToString();
    }

    static void FlipRows(byte[] bytes, int stride, int height)
    {
        var row = new byte[stride];
        for (int y = 0; y < height / 2; y++)
        {
            int a = y * stride, b = (height - 1 - y) * stride;
            Buffer.BlockCopy(bytes, a, row, 0, stride);
            Buffer.BlockCopy(bytes, b, bytes, a, stride);
            Buffer.BlockCopy(row, 0, bytes, b, stride);
        }
    }

    void Failure(string message)
    {
        ConnectionStatus = message;
        if (previewCanvas) previewCanvas.enabled = false;
        if (resultText) resultText.text = "";
        if (!warned) { Debug.LogWarning("YOLO: " + message, this); warned = true; }
    }

    Color ModelColor(string modelId)
    {
        if (models != null) foreach (var m in models) if (m.model_id == modelId) return m.color;
        return Color.white;
    }

    void DrawResult(YoloResponse result)
    {
        var names = new StringBuilder();
        var seen = new HashSet<string>();
        if (result.detections != null)
            foreach (var d in result.detections)
                AppendName(names, seen, d.model_id, d.class_id, d.class_name);
        if (result.classifications != null)
            foreach (var c in result.classifications)
                AppendName(names, seen, c.model_id, c.class_id, c.class_name);
        resultText.supportRichText = true;
        resultText.text = names.ToString();
    }

    void AppendName(StringBuilder names, HashSet<string> seen, string modelId, int classId, string className)
    {
        if (!seen.Add(modelId + ":" + classId)) return;
        if (names.Length > 0) names.Append('\n');
        var color = ColorUtility.ToHtmlStringRGB(ModelColor(modelId));
        names.Append("<color=#").Append(color).Append('>')
             .Append(modelId).Append(": ").Append(className.Replace("<", "?").Replace(">", "?"))
             .Append("</color>");
    }

    void OnDisable()
    {
        if (previewCanvas) previewCanvas.enabled = false;
        if (resultText) resultText.text = "";
        StopAllCoroutines();
        if (activeRequest != null) { activeRequest.Abort(); activeRequest.Dispose(); activeRequest = null; }
        RequestInFlight = false;
    }

    void OnDestroy()
    {
        if (captureTarget) { captureTarget.Release(); Destroy(captureTarget); }
        if (capturePixels) Destroy(capturePixels);
    }
}
