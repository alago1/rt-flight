import sys
import io
import pickle
from pprint import pprint
import time
import threading
from dataclasses import asdict

import zmq

import models.error as error
from models.bbox import BBox

context = zmq.Context()

#  Socket to talk to server
print("Connecting to Model Network")
model_socket = context.socket(zmq.REQ)
model_socket.connect("tcp://localhost:5555")

# img_path = "../data/dota_demo.jpg"
img_path = "../data/DJI_0007_copy.JPG"
# img_path = "../data_ignore/dji0007.out.tif"
# img_path = "../data/DJI_0007_copy.tif"

model_socket.send_string(img_path)
message = model_socket.recv()

try:
    result = pickle.load(io.BytesIO(message))

    if isinstance(result, list):
        # result is a list of BBox objects
        pprint(result)
    elif isinstance(result, error.DetectionError):
        print(f"Received Detection Error: {result.error_msg}")
    elif isinstance(result, error.HeaderError):
        print(f"Received Header Error: {result.error_msg}")
    else:
        print("Received unknown error")
except pickle.UnpicklingError:
    print("Could not unpickle message")
    sys.exit(1)
except Exception as e:
    print("Received unknown error while trying to parse message:")
    print(e)
    sys.exit(1)

print("Forwarding to Django UI")

PUBLISHER_PORT = 5556
SYNC_PORT = 5557

# number of subscribers to wait for before sending messages
NUM_SUBSCRIBERS = 1

connected_subs = 0


def publisher_worker():
    try:
        ui_publisher = context.socket(zmq.PUB)
        ui_publisher.sndhwm = 1100000  # set high water mark so messages aren't dropped
        ui_publisher.bind(f"tcp://*:{PUBLISHER_PORT}")

        while connected_subs < NUM_SUBSCRIBERS:
            ui_publisher.send(b'')
            time.sleep(0.1)
        
        print("Connected waiting 5 seconds")
        time.sleep(5)
        print("Sending...")
        ui_publisher.send_json([asdict(r) for r in result])
    except KeyboardInterrupt:
        pass
    
    ui_publisher.close()


def syncservice_worker():
    global connected_subs

    syncservice = context.socket(zmq.REP)
    syncservice.bind(f"tcp://*:{SYNC_PORT}")

    try:
        while True:
            syncservice.recv()
            connected_subs += 1
            print(f"Connected {connected_subs}/{NUM_SUBSCRIBERS} subscriber(s)")
            syncservice.send(b'')
    except KeyboardInterrupt:
        pass

    syncservice.close()


threads = []
for worker in (publisher_worker, syncservice_worker):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

model_socket.close()
context.term()
