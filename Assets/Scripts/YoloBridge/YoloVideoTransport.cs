using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;

/// <summary>Background video transport, bounded to one unsent frame and one result.</summary>
public sealed class YoloVideoTransport : IDisposable
{
    readonly object gate = new object();
    readonly AutoResetEvent wake = new AutoResetEvent(false);
    readonly string host;
    readonly int port;
    volatile bool stopped;
    TcpClient client;
    byte[] pending;
    string latestResult;
    public string Status { get; private set; } = "Connecting video";
    public int SentFrames { get; private set; }
    public int DroppedFrames { get; private set; }
    public YoloVideoTransport(string host, int port)
    {
        this.host = host; this.port = port;
        new Thread(Run) { IsBackground = true, Name = "YOLO video sender" }.Start();
    }
    static byte[] Size(int n) { return new[] { (byte)(n >> 24), (byte)(n >> 16), (byte)(n >> 8), (byte)n }; }
    public void Submit(byte[] jpeg, string metadata)
    {
        byte[] header = Encoding.UTF8.GetBytes(metadata);
        byte[] packet = new byte[8 + header.Length + jpeg.Length];
        Buffer.BlockCopy(Size(header.Length),0,packet,0,4);
        Buffer.BlockCopy(header,0,packet,4,header.Length);
        Buffer.BlockCopy(Size(jpeg.Length),0,packet,4+header.Length,4);
        Buffer.BlockCopy(jpeg,0,packet,8+header.Length,jpeg.Length);
        lock(gate) { if(pending != null) DroppedFrames++; pending = packet; }
        wake.Set();
    }
    public string TakeResult() { lock(gate) { string result=latestResult; latestResult=null; return result; } }
    static byte[] Read(NetworkStream stream, int count)
    {
        byte[] data=new byte[count]; int offset=0;
        while(offset<count) { int n=stream.Read(data,offset,count-offset); if(n==0) throw new System.IO.IOException("Video closed"); offset+=n; }
        return data;
    }
    void Run()
    {
        while(!stopped)
        {
            TcpClient connection=null;
            try
            {
                connection=new TcpClient(); client=connection;
                connection.NoDelay=true;
                var connect=connection.ConnectAsync(host,port);
                if(!connect.Wait(3000)) throw new TimeoutException("Video connect timeout");
                connect.GetAwaiter().GetResult();
                var stream=connection.GetStream();
                stream.WriteTimeout=3000; stream.ReadTimeout=30000;
                Status="Video connected";
                new Thread(() => Receive(connection,stream)) { IsBackground=true,Name="YOLO video results" }.Start();
                while(!stopped && connection.Connected)
                {
                    byte[] packet;
                    lock(gate) { packet=pending; pending=null; }
                    if(packet==null) { wake.WaitOne(100); continue; }
                    stream.Write(packet,0,packet.Length); SentFrames++;
                }
            }
            catch(Exception ex) { if(!stopped) Status="Video unavailable: "+ex.GetBaseException().Message; }
            finally { if(connection!=null) connection.Close(); }
            if(!stopped) wake.WaitOne(1000);
        }
    }
    void Receive(TcpClient connection, NetworkStream stream)
    {
        try
        {
            while(!stopped)
            {
                byte[] h=Read(stream,4);
                int count=(h[0]<<24)|(h[1]<<16)|(h[2]<<8)|h[3];
                if(count<=0 || count>6000000) throw new System.IO.IOException("Invalid video response");
                string json=Encoding.UTF8.GetString(Read(stream,count));
                lock(gate) latestResult=json;
            }
        }
        catch(Exception ex) { if(!stopped) Status="Video unavailable: "+ex.Message; }
        finally { connection.Close(); wake.Set(); }
    }
    public void Dispose()
    {
        stopped=true;
        if(client!=null) client.Close();
        wake.Set();
    }
}
