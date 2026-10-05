"""Latest-frame video transport. Same loaded Bridge as HTTP diagnostics."""
import json, socket, socketserver, struct, threading, time

def read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part: raise ConnectionError("stream closed")
        data.extend(part)
    return bytes(data)

class VideoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        condition = threading.Condition()
        state = {"latest": None, "closed": False}
        def infer():
            while True:
                with condition:
                    condition.wait_for(lambda: state["closed"] or state["latest"] is not None)
                    if state["closed"]: return
                    meta, jpeg, received = state["latest"]
                    state["latest"] = None
                try:
                    with self.server.bridge.lock:
                        result = self.server.bridge.predict(jpeg, int(meta["frame_id"]),
                            meta.get("models", []), str(meta.get("stream_id","video"))[:80])
                    result["stream_age_ms"] = (time.monotonic()-received)*1000
                    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    self.request.sendall(struct.pack("!I",len(payload))+payload)
                    self.server.processed += 1
                except Exception as exc:
                    if not state["closed"]: print("Video inference/connection failed:", str(exc), flush=True)
                    with condition:
                        state["closed"] = True
                        condition.notify_all()
                    self.request.shutdown(socket.SHUT_RDWR)
                    return
        worker = threading.Thread(target=infer, daemon=True)
        worker.start()
        try:
            while not state["closed"]:
                count = struct.unpack("!I",read_exact(self.request,4))[0]
                if not 0 < count <= 65536: raise ValueError("Invalid video metadata")
                meta = json.loads(read_exact(self.request,count))
                count = struct.unpack("!I",read_exact(self.request,4))[0]
                if not 0 < count <= 6000000: raise ValueError("Invalid video JPEG")
                jpeg = read_exact(self.request,count)
                with condition:
                    if state["latest"] is not None: self.server.dropped += 1
                    state["latest"] = (meta,jpeg,time.monotonic())
                    self.server.received += 1
                    condition.notify()
        except (OSError,ValueError,ConnectionError):
            pass
        finally:
            with condition:
                state["closed"] = True
                condition.notify_all()

class VideoServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    def __init__(self, bridge, port=8766):
        self.bridge = bridge
        self.received = self.processed = self.dropped = 0
        super().__init__(("127.0.0.1",port),VideoHandler)
    def metrics(self):
        return dict(received=self.received,processed=self.processed,dropped=self.dropped,port=8766)

