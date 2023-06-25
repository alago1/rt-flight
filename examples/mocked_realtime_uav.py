import traceback
import pathlib
import sys
import io
import pickle
from pprint import pprint
import time
import threading
from dataclasses import asdict
from queue import SimpleQueue

import zmq
import numpy as np

from util.mock_flight_path import mock_flight_path

# symlinked from raspberry_pi_code
from models.error import DetectionError, HeaderError
from models.bbox import BBox


context = zmq.Context()

#  Socket to talk to server
print("Connecting to Model Network")
model_socket = context.socket(zmq.REQ)
model_socket.connect("tcp://localhost:5555")

PUBLISHER_PORT = 5556
SYNC_PORT = 5557

# number of subscribers to wait for before sending messages
NUM_SUBSCRIBERS = 1

connected_subs = 0

message_queue = SimpleQueue()


def publisher_worker():
    try:
        ui_publisher = context.socket(zmq.PUB)
        ui_publisher.sndhwm = 1100000  # set high water mark so messages aren't dropped
        ui_publisher.bind(f"tcp://*:{PUBLISHER_PORT}")

        while connected_subs < NUM_SUBSCRIBERS:
            ui_publisher.send(b'')
            time.sleep(0.1)
        
        while True:
            if not message_queue.empty():
                ui_publisher.send_json([asdict(r) for r in message_queue.get()])
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


DATASET_TOP_LEFT_GPS = np.array((12.86308254761559, 77.5151947517078))
DATASET_TOP_RIGHT_GPS = np.array((12.863010715187013, 77.52267023737696))
DATASET_BOT_LEFT_GPS = np.array((12.859008245256549, 77.5151541499705))
DATASET_BOT_RIGHT_GPS = np.array((12.858936436333265, 77.52262951527761))
DATASET_CORNER_GPS_COORDS = np.array([DATASET_TOP_LEFT_GPS, DATASET_TOP_RIGHT_GPS, DATASET_BOT_RIGHT_GPS, DATASET_BOT_LEFT_GPS])


for img_path in mock_flight_path("../data/Blore_Clean.jpg", 5, DATASET_CORNER_GPS_COORDS, 100.5):
    
    model_socket.send_string(img_path)
    message = model_socket.recv()

    try:
        result = pickle.load(io.BytesIO(message))
        message_queue.put(result)

        if isinstance(result, list):
            # result is a list of BBox objects
            pprint(result)
        elif isinstance(result, DetectionError):
            print(f"Received Detection Error: {result.error_msg}")
        elif isinstance(result, HeaderError):
            print(f"Received Header Error: {result.error_msg}")
        else:
            print("Received unknown error")
    except pickle.UnpicklingError:
        print("Could not unpickle message")
        sys.exit(1)
    except Exception as e:
        print("Received unknown error while trying to parse message:")
        print(traceback.format_exc())
        sys.exit(1)


for t in threads:
    t.join()


model_socket.close()
context.term()
