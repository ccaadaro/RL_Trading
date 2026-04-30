import zmq
import json
import time

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5556")
sock.setsockopt(zmq.SUBSCRIBE, b"")

print("Listening for 10 seconds...")
start = time.time()
while time.time() - start < 10:
    try:
        topic, payload = sock.recv_multipart(flags=zmq.NOBLOCK)
        print(f"Received: {json.loads(payload)}")
    except zmq.Again:
        time.sleep(1)
